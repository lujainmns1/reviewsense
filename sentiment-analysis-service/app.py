from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from flask import Flask, jsonify, request, render_template

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Optional deps; server can start without them, but inference will require them.
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModel,
        pipeline,
    )
    import langdetect
    import yake
    from scipy.spatial.distance import cosine
    logger.info("All optional dependencies loaded successfully")
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
except ImportError as e:
    logger.error(f"Critical dependency not available: {e}")
    logger.error("Please ensure all requirements are installed: pip install -r requirements.txt")
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    pipeline = None  # type: ignore
    torch = None  # type: ignore
    langdetect = None  # type: ignore
    yake = None  # type: ignore
    cosine = None  # type: ignore

app = Flask(__name__)

# ---------------------------
# Arabic text utilities
# ---------------------------

_AR_ABBREV_POS = {"ممتاز", "رائع", "رائعة", "رائعًا", "جيد", "جيدة", "جداً", "جدًا", "مميز", "أنصح", "أوصي", "أوصي به"}
_AR_ABBREV_NEG = {"سيئ", "سيئة", "رديء", "رديئة", "سيئًا", "سئ", "سئية", "خيبة أمل", "لا أنصح", "مش جيد", "أسوأ", "كارثي"}
_AR_HEDGE_NEUTRAL = {"متوسط", "عادي", "لا بأس", "معقول", "قد", "ربما", "إلى حد ما"}

_AR_ASPECT_BOOST = {
    "المنتج", "السعر", "مرتفع", "رخيص", "الجودة", "الخدمة", "خدمة العملاء",
    "التوصيل", "الشحن", "العبوة", "التغليف", "الأداء", "النظافة",
    "التصميم", "المتانة", "السرعة", "الدعم", "النكهة", "الصحة", "الطعم",
    "الحجم", "اللون", "المادة", "الكمية"
}

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

def normalize_arabic(text: str) -> str:
    """Light Arabic normalization to stabilize both sentiment and topics."""
    t = text
    t = _AR_DIACRITICS.sub("", t)             # remove diacritics
    t = t.replace(_TATWEEL, "")               # remove tatweel
    # unify alef/hamza variants
    t = re.sub("[\u0622\u0623\u0625]", "ا", t)
    # taa marbuta -> ha in many preprocessors; here retain but unify spaces
    t = t.replace("ي", "ي").replace("ى", "ي")
    # punctuation & spaces
    t = re.sub(r"[^\w\u0600-\u06FF\s]", " ", t)  # keep Arabic letters/digits/underscore/space
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\r\n\t]", " ", text)
    return text

def detect_language(text: str) -> Optional[str]:
    if langdetect is None:
        return None
    try:
        return langdetect.detect(text)
    except Exception:
        return None

# ---------------------------
# Dialect Identification
# ---------------------------

class DialectIdentifier:
    """MARBERTv2 dialect classifier; returns one of MAGHREB, LEV, MSA, GLF, EGY."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._classifier = None

    def load(self) -> None:
        if self._classifier is None:
            if pipeline is None:
                raise RuntimeError("transformers not installed; dialect detection unavailable.")
            device = 0 if torch and torch.cuda.is_available() else -1
            self._classifier = pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                return_all_scores=False,
                device=device,
            )

    def predict(self, text: str) -> Optional[str]:
        try:
            self.load()
            result = self._classifier(text)[0]
            label = result.get("label")
            # Map to user-friendly names
            mapping = {
                "MAGHREB": "Maghrebi",
                "LEV": "Levantine",
                "MSA": "MSA",
                "GLF": "Gulf",
                "EGY": "Egyptian",
            }
            return mapping.get(label, label)
        except Exception:
            return None

# ---------------------------
# Sentiment Models
# ---------------------------

class SentimentModel:
    """Wrapper around a HuggingFace sentiment model with robust tokenizer handling."""
    def __init__(self, model_id: str, display_name: str, description: str):
        self.model_id = model_id
        self.display_name = display_name
        self.description = description
        self._classifier = None  # HF pipeline
        logger.info(f"Initialized sentiment model: {display_name} ({model_id})")

    def load(self) -> None:
        if self._classifier is not None:
            return
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            logger.error(f"Cannot load model {self.model_id}: transformers not available")
            raise RuntimeError("transformers not available; cannot load sentiment model.")

        logger.info(f"Loading sentiment model: {self.model_id}")
        start_time = datetime.now()

        device = 0 if torch and torch.cuda.is_available() else -1
        logger.info(f"Using device: {'cuda' if device == 0 else 'cpu'}")

        # Avoid SentencePiece fast-tokenizer conversion for XLM-R/Cardiff
        needs_slow = ("xlm-roberta" in self.model_id) or ("cardiffnlp" in self.model_id)
        if needs_slow:
            logger.info(f"Using slow tokenizer for {self.model_id}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._classifier = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=False,
                device=device,
            )
        else:
            self._classifier = pipeline(
                "sentiment-analysis",
                model=self.model_id,
                tokenizer=self.model_id,
                return_all_scores=False,
                device=device,
            )
        
        load_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model {self.model_id} loaded in {load_time:.2f}s")

    @staticmethod
    def _normalize_label(label_raw: str) -> str:
        l = (label_raw or "").strip().lower()
        if l in {"pos", "positive", "4 stars", "5 stars"}:
            return "positive"
        if l in {"neg", "negative", "1 star", "2 stars"}:
            return "negative"
        if l in {"neutral", "3 stars", "mixed", "objective"}:
            return "neutral"
        # fallback
        return "neutral"

    def predict(self, text: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        """Return {'label': 'positive|neutral|negative', 'score': float}."""
        logger.info(f"Processing text for prediction: {text[:100]}...")
        
        self.load()
        
        # Get tokenizer info if available
        if hasattr(self._classifier, "tokenizer"):
            tokens = self._classifier.tokenizer.tokenize(text)
            logger.info(f"Tokenized input ({len(tokens)} tokens): {tokens[:20]}...")
            
            # Log token IDs for debugging
            token_ids = self._classifier.tokenizer.encode(text)
            logger.info(f"Token IDs: {token_ids[:20]}...")
        
        # Run model inference
        logger.info("Running model inference...")
        out = self._classifier(text)[0]
        logger.info(f"Raw model output: {out}")
        
        label = self._normalize_label(out.get("label", "neutral"))
        score = float(out.get("score", 0.0))
        logger.info(f"Normalized prediction: {label} (score: {score})")

        # --------- Accuracy guards (light calibration) ----------
        # Hedge detection → pull extreme scores toward neutral
        t_norm = normalize_arabic(text) if (lang_hint == "ar") else text.lower()
        logger.info(f"Normalized text: {t_norm[:100]}...")
        
        contains_pos = any(w in t_norm for w in _AR_ABBREV_POS) if lang_hint == "ar" else False
        contains_neg = any(w in t_norm for w in _AR_ABBREV_NEG) if lang_hint == "ar" else False
        contains_hedge = any(w in t_norm for w in _AR_HEDGE_NEUTRAL) if lang_hint == "ar" else False
        
        # Log detected keywords
        if lang_hint == "ar":
            pos_words = [w for w in _AR_ABBREV_POS if w in t_norm]
            neg_words = [w for w in _AR_ABBREV_NEG if w in t_norm]
            hedge_words = [w for w in _AR_HEDGE_NEUTRAL if w in t_norm]
            logger.info(f"Detected positive words: {pos_words}")
            logger.info(f"Detected negative words: {neg_words}")
            logger.info(f"Detected hedge words: {hedge_words}")

        # If clearly mixed language (“جميل لكن… مرتفع”) enforce neutral band
        mixed_positive_negative = contains_pos and contains_neg

        # Pull toward neutral in hedged or mixed cases
        if lang_hint == "ar" and (contains_hedge or mixed_positive_negative):
            if label in {"positive", "negative"} and score > 0.85:
                score = 0.72  # dampen
            label = "neutral"  # centralize

        # If contradiction between label and strong polarity words, nudge
        if lang_hint == "ar":
            if label == "positive" and contains_neg:
                label, score = "neutral", min(score, 0.7)
            elif label == "negative" and contains_pos:
                label, score = "neutral", min(score, 0.7)

        # Keep score to 4 dp
        return {"label": label, "score": round(score, 4)}

# ---------------------------
# Arabic POS Tagger for Topic Candidates
# ---------------------------

class ArabicPOSTagger:
    """CAMeL BERT POS tagger to extract NOUN/ADJ phrases as topic candidates."""
    def __init__(self):
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is None:
            if pipeline is None:
                raise RuntimeError("transformers not available; POS tagging unavailable.")
            try:
                self._pipeline = pipeline(
                    "token-classification",
                    #  if it terminates
                    # model="CAMeL-Lab/bert-base-arabic-camelbert-mix-pos",
                    model="CAMeL-Lab/bert-base-arabic-camelbert-ca",
                    aggregation_strategy="simple",
                )
            except Exception as e:
                print(f"Warning: Failed to load POS model: {e}")
                self._pipeline = None

    def predict(self, text: str) -> List[str]:
        """Extract grouped NOUN/ADJ phrases."""
        if self._pipeline is None:
            return []
        try:
            results = self._pipeline(text)
            # Filter and group consecutive NOUN/ADJ
            candidates = []
            current_phrase = []
            for res in results:
                pos = res.get('entity', '')
                token = res.get('word', '').strip()
                if token and pos in {'NOUN', 'ADJ'}:
                    current_phrase.append(token)
                else:
                    if current_phrase:
                        candidates.append(' '.join(current_phrase))
                        current_phrase = []
            if current_phrase:
                candidates.append(' '.join(current_phrase))
            return [c for c in candidates if len(c) > 0]
        except Exception:
            return []

# ---------------------------
# Embedder for Re-ranking
# ---------------------------

ARABERT_EMBEDDER: Optional[Tuple[AutoTokenizer, AutoModel]] = None

def load_embedder() -> None:
    """Load Arabic BERT embedder for semantic topic extraction."""
    global ARABERT_EMBEDDER
    if ARABERT_EMBEDDER is not None:
        return
    if AutoTokenizer is None or AutoModel is None or torch is None:
        ARABERT_EMBEDDER = None
        return
    try:
        model_id = "aubmindlab/bert-base-arabertv02"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        device = 0 if torch.cuda.is_available() else -1
        model = model.to(device)
        ARABERT_EMBEDDER = (tokenizer, model)
    except Exception:
        ARABERT_EMBEDDER = None

def _get_text_embedding(tokenizer: AutoTokenizer, model: AutoModel, text: str) -> Optional[torch.Tensor]:
    """Compute mean-pooled embedding for text."""
    try:
        logger.info("Tokenizing text for embedding...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[:20]}...")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info("Computing embeddings...")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        logger.info(f"Embedding shape: {emb.shape}")
        return emb
    except Exception as e:
        logger.error(f"Failed to compute embedding: {e}")
        return None

# ---------------------------
# Fallback YAKE
# ---------------------------

def _topics_yake(text: str, lang: str = "ar", top_k: int = 5) -> List[Tuple[str, float]]:
    """Fallback: Run YAKE with n-grams 1-2 only, normalize and merge near-duplicates."""
    if yake is None:
        return []
    t = normalize_arabic(text) if lang == "ar" else text.lower()
    candidates: List[Tuple[str, float]] = []
    for n in (1, 2):  # Limit to 1-2 grams for shorter topics
        try:
            kw = yake.KeywordExtractor(lan=lang, n=n, top=top_k*2, features=None).extract_keywords(t)
            candidates.extend(kw)
        except Exception:
            continue

    # Merge duplicates
    merged: Dict[str, float] = {}
    for phrase, raw_score in candidates:
        p = re.sub(r"\s+", " ", phrase).strip()
        if len(p.split()) > 2:  # Skip longer than bigrams
            continue
        conf = 1.0 - float(raw_score)
        merged[p] = max(merged.get(p, 0), conf)

    # Boost aspects
    boosted = []
    for p, c in merged.items():
        b = c + 0.05 if any(a in p for a in _AR_ASPECT_BOOST) else c
        b = min(1.0, b)
        boosted.append((p, b))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted[:top_k]

# ---------------------------
# Topic Extraction (POS-guided for Arabic)
# ---------------------------

POS_TAGGER = ArabicPOSTagger()

def extract_topics(text: str, lang: Optional[str]) -> List[Dict[str, Any]]:
    """POS-guided topic extraction for Arabic (noun/adj phrases), re-ranked by BERT similarity. Fallback to YAKE. Top 3 only."""
    logger.info(f"Starting topic extraction for text: {text[:100]}...")
    logger.info(f"Language: {lang}")
    
    lang_eff = lang if lang in {"ar", "en"} else "ar"
    top_k = 3

    if lang_eff != "ar":
        # Non-Arabic: YAKE fallback
        logger.info("Using YAKE for non-Arabic text")
        pairs = _topics_yake(text, lang=lang_eff, top_k=top_k)
        logger.info(f"YAKE extracted {len(pairs)} initial candidates")
        return [{"topic": p, "score": round(float(s), 4)} for p, s in pairs]

    # Arabic: POS extraction
    logger.info("Using POS tagger for Arabic text")
    t_norm = normalize_arabic(text)
    logger.info(f"Normalized text for POS: {t_norm[:100]}...")
    
    candidates = POS_TAGGER.predict(t_norm)
    logger.info(f"POS tagger found {len(candidates)} candidate phrases: {candidates}")
    
    if not candidates:
        # Fallback to YAKE
        pairs = _topics_yake(text, lang=lang_eff, top_k=top_k)
        return [{"topic": p, "score": round(float(s), 4)} for p, s in pairs]

    # Re-rank with BERT if available
    load_embedder()
    if ARABERT_EMBEDDER is None or cosine is None:
        # Use uniform scores or simple boost
        boosted = []
        for p in candidates:
            s = 0.8  # Base score
            if any(a in p for a in _AR_ASPECT_BOOST):
                s = min(1.0, s + 0.2)
            boosted.append((p, s))
    else:
        tokenizer, model = ARABERT_EMBEDDER
        text_emb = _get_text_embedding(tokenizer, model, t_norm)
        if text_emb is None:
            # Fallback uniform
            boosted = [(p, 0.8 + 0.2 if any(a in p for a in _AR_ASPECT_BOOST) else 0.8) for p in candidates]
            boosted = [(p, min(1.0, s)) for p, s in boosted]
        else:
            scores = []
            for phrase in candidates:
                phrase_emb = _get_text_embedding(tokenizer, model, phrase)
                if phrase_emb is not None and text_emb.shape == phrase_emb.shape:
                    sim = 1 - cosine(text_emb, phrase_emb)
                else:
                    sim = 0.5
                scores.append((phrase, sim))
            
            # Boost and sort
            boosted = []
            for p, s in scores:
                boost = 0.2 if any(a in p for a in _AR_ASPECT_BOOST) else 0.0
                boosted.append((p, min(1.0, s + boost)))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return [{"topic": p, "score": round(s, 4)} for p, s in boosted[:top_k]]

# ---------------------------
# Available Models
# ---------------------------

AVAILABLE_MODELS: Dict[str, SentimentModel] = {
    "marbertv2-book-review-sa": SentimentModel(
        model_id="AbdallahNasir/book-review-sentiment-classification",
        display_name="MARBERTv2 Book Review (Positive/Neutral/Negative)",
        description=(
            "Fine-tuned MARBERTv2 model on the LABR dataset (book reviews). "
            "Useful proxy for product reviews with nuanced Arabic phrasing."
        ),
    ),
    "arabert-arsas-sa": SentimentModel(
        model_id="Abdo36/Arabert-Sentiment-Analysis-ArSAS",
        display_name="AraBERTv2 ArSAS (Positive/Neutral/Negative/Mixed)",
        description=(
            "AraBERTv2 fine-tuned on ArSAS (Farasa tokenization). Mixed is mapped to neutral; "
            "good at handling hedged/ambivalent statements."
        ),
    ),
    "xlm-roberta-twitter-sa": SentimentModel(
        model_id="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        display_name="XLM-RoBERTa Twitter (Multilingual)",
        description=(
            "Multilingual XLM-RoBERTa trained on ~198M tweets in 8 languages including Arabic. "
            "Loaded with slow tokenizer to avoid SentencePiece conversion issues."
        ),
    ),
}

# Dialect model (optional)
DIALECT_MODEL = DialectIdentifier(
    model_id="IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
)

# ---------------------------
# Routes
# ---------------------------

@app.route("/models", methods=["GET"])
def list_models() -> Any:
    models_info = [
        {"name": key, "description": model.display_name + " – " + model.description}
        for key, model in AVAILABLE_MODELS.items()
    ]
    return jsonify(models_info)

@app.route("/", methods=["GET"])
def index():
    # Simple UI if you kept templates/index.html
    # If you do not use templates, you can remove this route safely.
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    req_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logger.info(f"[{req_id}] ========== New Analysis Request ==========")
    
    # Support both JSON API clients and simple HTML form submissions
    came_from_form = False
    payload = None

    # Log request details
    logger.info(f"[{req_id}] Request Headers:")
    for header, value in request.headers.items():
        logger.info(f"[{req_id}]   {header}: {value}")

    # Get dialect settings from request
    auto_detect_dialect = request.json.get('autoDetectDialect', True)
    country_code = request.json.get('country')
    
    # Map country code to dialect if provided and auto-detect is disabled
    country_dialect_map = {
        'EG': 'EGY',
        'SA': 'GLF', 'AE': 'GLF', 'KW': 'GLF', 'BH': 'GLF', 'QA': 'GLF', 'OM': 'GLF',
        'JO': 'LEV', 'LB': 'LEV', 'SY': 'LEV', 'PS': 'LEV',
        'MA': 'MAGHREB', 'DZ': 'MAGHREB', 'TN': 'MAGHREB', 'LY': 'MAGHREB'
    }

    # Prefer JSON when present
    if request.is_json:
        try:
            payload = request.get_json(force=True, silent=False)
            logger.info(f"[{req_id}] JSON payload received:")
            logger.info(f"[{req_id}] {json.dumps(payload, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"[{req_id}] Failed to parse JSON: {e}")
            payload = None

    # Fall back to form-encoded data (e.g., HTML form POST)
    if payload is None or not isinstance(payload, dict):
        if request.form:
            came_from_form = True
            payload = {
                "text": request.form.get("text", ""),
                "model": request.form.get("model", ""),
            }
            logger.info(f"[{req_id}] Form data received:")
            logger.info(f"[{req_id}] Text: {payload['text'][:200]}...")
            logger.info(f"[{req_id}] Model: {payload['model']}")

    if not isinstance(payload, dict):
        logger.error(f"[{req_id}] Invalid payload format")
        return jsonify({"error": "Invalid payload.", "code": "bad_payload"}), 400

    text = payload.get("text", "")
    model_name = payload.get("model", "")

    if not isinstance(text, str) or not text.strip():
        logger.error(f"[{req_id}] Empty or invalid input text received")
        return jsonify({"error": "Empty input text.", "code": "empty_text"}), 400

    if model_name not in AVAILABLE_MODELS:
        logger.error(f"[{req_id}] Unknown model requested: {model_name}")
        return jsonify({"error": "Unknown model.", "code": "unknown_model"}), 400

    # Text preprocessing pipeline
    logger.info(f"[{req_id}] Starting text preprocessing pipeline")
    logger.info(f"[{req_id}] Original text ({len(text)} chars): {text[:200]}...")
    
    original_text = text
    cleaned = clean_text(original_text)
    logger.info(f"[{req_id}] Cleaned text ({len(cleaned)} chars): {cleaned[:200]}...")
    
    lang = detect_language(cleaned) or "ar"  # default to Arabic if unsure
    logger.info(f"[{req_id}] Detected language: {lang} (confidence: {'high' if lang else 'fallback to ar'})")

    # Dialect only if Arabic detected
    dialect = None
    if lang == "ar":
        try:
            dialect = DIALECT_MODEL.predict(cleaned)
        except Exception:
            dialect = None

    # Sentiment
    model = AVAILABLE_MODELS[model_name]
    try:
        sentiment = model.predict(cleaned, lang_hint=lang)
    except Exception as e:
        # Fallback: try a different model automatically to avoid hard failure
        fallback_model = "arabert-arsas-sa" if model_name != "arabert-arsas-sa" else "marbertv2-book-review-sa"
        try:
            sentiment = AVAILABLE_MODELS[fallback_model].predict(cleaned, lang_hint=lang)
            model_name = fallback_model  # report the actually used model
        except Exception:
            return jsonify({"error": f"Model execution failed: {str(e)}", "code": "model_error"}), 500

    # Dialect detection logic
    dialect = None
    if lang == "ar":
        if not auto_detect_dialect and country_code in country_dialect_map:
            # Use country-specific dialect if auto-detection is disabled
            dialect = country_dialect_map[country_code]
            logger.info(f"[{req_id}] Using dialect {dialect} based on country {country_code}")
        else:
            # Auto-detect dialect using DIALECT_MODEL
            try:
                dialect = DIALECT_MODEL.predict(cleaned)
                logger.info(f"[{req_id}] Auto-detected dialect: {dialect}")
            except Exception as e:
                logger.error(f"[{req_id}] Failed to detect dialect: {e}")
                dialect = None

    # Topics
    try:
        topics = extract_topics(cleaned, lang)
    except Exception:
        topics = []

    # Build response in the exact order specified
    response = {
        "original_text": original_text,
        "cleaned_text": normalize_arabic(cleaned) if lang == "ar" else cleaned,
        "language": lang,
        "dialect": dialect,
        "sentiment": {
            "label": sentiment["label"],
            "score": float(sentiment["score"]),
        },
        "topics": topics,
        "model_used": model_name,
    }
    
    # Log final results
    logger.info(f"[{req_id}] ===== Analysis Results =====")
    logger.info(f"[{req_id}] Language: {lang}")
    logger.info(f"[{req_id}] Dialect: {dialect}")
    logger.info(f"[{req_id}] Sentiment: {sentiment['label']} (score: {sentiment['score']:.4f})")
    logger.info(f"[{req_id}] Topics found: {len(topics)}")
    for idx, topic in enumerate(topics, 1):
        logger.info(f"[{req_id}]   Topic {idx}: {topic['topic']} (score: {topic['score']:.4f})")
    logger.info(f"[{req_id}] Model used: {model_name}")
    
    # If the request came from an HTML form, render a server-side result page
    if came_from_form:
        try:
            logger.info(f"[{req_id}] Rendering HTML template with results")
            return render_template("result.html", result=response)
        except Exception as e:
            logger.error(f"[{req_id}] Failed to render template: {e}")
            # If templating fails, fall back to JSON response
            pass

    logger.info(f"[{req_id}] Returning JSON response")
    logger.info(f"[{req_id}] ========== Request Complete ==========\n")
    return jsonify(response), 200


def preload_models():
    """
    Load essential models at startup. 
    In a low-memory environment like a default Codespace,
    we only load ONE sentiment model to prevent crashing.
    """
    try:
        import torch
        import transformers
        logger.info(f"torch version: {torch.__version__}")
        logger.info(f"transformers version: {transformers.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available. Using CPU")
            
    except ImportError as e:
        logger.error(f"Failed to import core dependencies: {e}")
        logger.error("Please ensure torch and transformers are installed correctly")
        return

    if not AutoModelForSequenceClassification or not AutoTokenizer or not pipeline:
        logger.error("Transformers components not available. Please ensure the package is installed correctly.")
        return

    # Load one primary model for sentiment analysis
    models_to_preload = ["arabert-arsas-sa"]  # Using the smaller AraBERT model
    
    logger.info("Pre-loading selected sentiment models...")
    for model_name in models_to_preload:
        if model_name in AVAILABLE_MODELS:
            model = AVAILABLE_MODELS[model_name]
            try:
                model.load()
                logger.info(f"Successfully loaded '{model_name}'")
            except Exception as e:
                logger.error(f"FAILED to load '{model_name}': {e}")
                logger.error("Please verify transformers installation and model availability")
    
    # print("\nPre-loading dialect detection model...")
    # try:
    #     DIALECT_MODEL.load()
    #     print("  - Successfully loaded dialect model.")
    # except Exception as e:
    #     print(f"  - FAILED to load dialect model: {e}")

    print("\nPre-loading Arabic BERT embedder for improved topics...")
    try:
        load_embedder()
        logger.info("Successfully loaded embedder")
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")

    logger.info("Pre-loading Arabic POS tagger for topic candidates...")
    try:
        POS_TAGGER.load()
        logger.info("Successfully loaded POS tagger")
    except Exception as e:
        logger.error(f"Failed to load POS tagger: {e}")

if __name__ == "__main__":
    with app.app_context():
        preload_models()
    
    logger.info("Starting sentiment analysis microservice on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=False)