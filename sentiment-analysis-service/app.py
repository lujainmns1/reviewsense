"""
Flask-based sentiment analysis application supporting Arabic reviews with dialect detection.
reviewsense/sentiment-analysis-service/app.py
Endpoints:
- GET /models     → list available models (name + description)
- POST /analyze   → analyze a single review using a selected model

Response schemas strictly follow the spec provided by the user:
- Success: { original_text, cleaned_text, language, dialect, sentiment{label,score}, topics[{topic,score}], model_used }
- Error:   { "error": "<description>", "code": "<short_code>" }
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, render_template

# Optional deps; server can start without them, but inference will require them.
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
    import torch
    import langdetect
    import yake
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore
    torch = None  # type: ignore
    langdetect = None  # type: ignore
    yake = None  # type: ignore

app = Flask(__name__)

# ---------------------------
# Arabic text utilities
# ---------------------------

_AR_ABBREV_POS = {"ممتاز", "رائع", "رائعة", "رائعًا", "جيد", "جيدة", "جداً", "جدًا", "مميز", "أنصح", "أوصي", "أوصي به"}
_AR_ABBREV_NEG = {"سيئ", "سيئة", "رديء", "رديئة", "سيئًا", "سئ", "سئية", "خيبة أمل", "لا أنصح", "مش جيد", "أسوأ", "كارثي"}
_AR_HEDGE_NEUTRAL = {"متوسط", "عادي", "لا بأس", "معقول", "قد", "ربما", "إلى حد ما"}

_AR_ASPECT_BOOST = {
    "السعر", "مرتفع", "رخيص", "الجودة", "الخدمة", "خدمة العملاء",
    "التوصيل", "الشحن", "العبوة", "التغليف", "الأداء"
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

    def load(self) -> None:
        if self._classifier is not None:
            return
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers not available; cannot load sentiment model.")

        device = 0 if torch and torch.cuda.is_available() else -1

        # Avoid SentencePiece fast-tokenizer conversion for XLM-R/Cardiff
        needs_slow = ("xlm-roberta" in self.model_id) or ("cardiffnlp" in self.model_id)
        if needs_slow:
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
        self.load()
        out = self._classifier(text)[0]
        label = self._normalize_label(out.get("label", "neutral"))
        score = float(out.get("score", 0.0))

        # --------- Accuracy guards (light calibration) ----------
        # Hedge detection → pull extreme scores toward neutral
        t_norm = normalize_arabic(text) if (lang_hint == "ar") else text.lower()
        contains_pos = any(w in t_norm for w in _AR_ABBREV_POS) if lang_hint == "ar" else False
        contains_neg = any(w in t_norm for w in _AR_ABBREV_NEG) if lang_hint == "ar" else False
        contains_hedge = any(w in t_norm for w in _AR_HEDGE_NEUTRAL) if lang_hint == "ar" else False

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
# Topic Extraction (YAKE)
# ---------------------------

def _topics_yake(text: str, lang: str = "ar", top_k: int = 5) -> List[Tuple[str, float]]:
    """Run YAKE with n-grams 1..3, normalize and merge near-duplicates."""
    if yake is None:
        return []
    # Use normalized Arabic for robustness
    t = normalize_arabic(text) if lang == "ar" else text.lower()
    candidates: List[Tuple[str, float]] = []
    for n in (1, 2, 3):
        kw = yake.KeywordExtractor(lan=lang, n=n, top=top_k, features=None).extract_keywords(t)
        candidates.extend(kw)

    # YAKE scores: lower is better → convert to confidence = 1 - score
    # Merge duplicates (exact & space-normalized)
    merged: Dict[str, float] = {}
    for phrase, raw_score in candidates:
        p = re.sub(r"\s+", " ", phrase).strip()
        conf = 1.0 - float(raw_score)
        # Keep max confidence for duplicates
        if p in merged:
            merged[p] = max(merged[p], conf)
        else:
            merged[p] = conf

    # Aspect boost for product-review dimensions
    boosted: List[Tuple[str, float]] = []
    for p, c in merged.items():
        b = c
        if any(a in p for a in _AR_ASPECT_BOOST):
            b = min(1.0, c + 0.05)  # small positive bump
        boosted.append((p, b))

    # Sort by confidence desc and take top_k
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted[:top_k]

def extract_topics(text: str, lang: Optional[str]) -> List[Dict[str, Any]]:
    lang_eff = lang if lang in {"ar", "en"} else "ar"  # default to ar for Arabic text
    pairs = _topics_yake(text, lang=lang_eff, top_k=5)
    return [{"topic": p, "score": round(float(s), 4)} for p, s in pairs]

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
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON payload.", "code": "bad_json"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload.", "code": "bad_payload"}), 400

    text = payload.get("text", "")
    model_name = payload.get("model", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Empty input text.", "code": "empty_text"}), 400

    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": "Unknown model.", "code": "unknown_model"}), 400

    original_text = text
    cleaned = clean_text(original_text)
    lang = detect_language(cleaned) or "ar"  # default to Arabic if unsure

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
    return jsonify(response), 200


def preload_models():
    """
    Load essential models at startup. 
    In a low-memory environment like a default Codespace,
    we only load ONE sentiment model to prevent crashing.
    """
    # CHOOSE ONE MODEL to preload. The others will be loaded on-demand,
    # which might still crash, but it won't crash on startup.
    # The best practice is to only use one and comment out the others.
    
    models_to_preload = ["arabert-arsas-sa","marbertv2-book-review-sa"] # Or "arabert-arsas-sa"
    
    print("Pre-loading selected sentiment models...")
    for model_name in models_to_preload:
        if model_name in AVAILABLE_MODELS:
            model = AVAILABLE_MODELS[model_name]
            try:
                model.load()
                print(f"  - Successfully loaded '{model_name}'")
            except Exception as e:
                print(f"  - FAILED to load '{model_name}': {e}")
    
    print("\nPre-loading dialect detection model...")
    try:
        DIALECT_MODEL.load()
        print("  - Successfully loaded dialect model.")
    except Exception as e:
        print(f"  - FAILED to load dialect model: {e}")

if __name__ == "__main__":
    with app.app_context():
        preload_models()
    
    print("\nStarting sentiment analysis microservice...")
    app.run(host="0.0.0.0", port=5001, debug=False)