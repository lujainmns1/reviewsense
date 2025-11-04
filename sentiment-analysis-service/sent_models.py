from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Core required dependencies (torch and transformers)
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModel,
        pipeline,
    )
    logger.info("Core dependencies (torch, transformers) loaded successfully")
    
    # Detailed GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # Set default device to GPU
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.current_device()}")
    else:
        logger.info("CUDA not available, using CPU")
except ImportError as e:
    logger.error(f"Critical dependency not available: {e}")
    logger.error("Please ensure all requirements are installed: pip install -r requirements.txt")
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    pipeline = None  # type: ignore
    torch = None  # type: ignore

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

        # Force CUDA device if available
        if torch and torch.cuda.is_available():
            device = 0
            torch.cuda.set_device(device)
            logger.info(f"Using GPU device {device}: {torch.cuda.get_device_name(device)}")
        else:
            device = -1
            logger.info("Using CPU device")

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
        # Import here to avoid circular dependency
        from utils import normalize_arabic, _AR_ABBREV_POS, _AR_ABBREV_NEG, _AR_HEDGE_NEUTRAL
        
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

        # If clearly mixed language ("جميل لكن… مرتفع") enforce neutral band
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
        if self._pipeline is not None:
            return  # Already loaded
        if pipeline is None:
            logger.warning("transformers not available; POS tagging unavailable.")
            self._pipeline = None
            return
        try:
            logger.info("Loading Arabic POS tagger model...")
            device = 0 if torch and torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "token-classification",
                #  if it terminates
                # model="CAMeL-Lab/bert-base-arabic-camelbert-mix-pos",
                model="CAMeL-Lab/bert-base-arabic-camelbert-ca",
                aggregation_strategy="simple",
                device=device,
            )
            logger.info("Arabic POS tagger loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load POS model: {e}", exc_info=True)
            self._pipeline = None

    def predict(self, text: str) -> List[str]:
        """Extract grouped NOUN/ADJ phrases."""
        try:
            self.load()  # Ensure model is loaded
        except Exception as e:
            logger.warning(f"Failed to load POS tagger: {e}")
            return []
        
        if self._pipeline is None:
            return []
        try:
            logger.info(f"Running POS tagger on text: {text[:100]}...")
            results = self._pipeline(text)
            logger.info(f"POS tagger raw results: {results[:5] if results else 'empty'}")  # Log first 5 results
            
            # Filter and group consecutive NOUN/ADJ
            candidates = []
            current_phrase = []
            for res in results:
                # Handle different possible key names
                pos = res.get('entity', '') or res.get('label', '') or res.get('tag', '')
                token = res.get('word', '') or res.get('token', '') or res.get('text', '')
                token = token.strip()
                
                logger.debug(f"POS result: token='{token}', pos='{pos}'")
                
                if token and pos in {'NOUN', 'ADJ', 'noun', 'adj', 'NOUN_PROP', 'ADJ_COMPL'}:
                    current_phrase.append(token)
                else:
                    if current_phrase:
                        phrase = ' '.join(current_phrase)
                        if len(phrase) > 1:  # Only add non-empty phrases
                            candidates.append(phrase)
                        current_phrase = []
            if current_phrase:
                phrase = ' '.join(current_phrase)
                if len(phrase) > 1:
                    candidates.append(phrase)
            
            logger.info(f"POS tagger extracted {len(candidates)} phrases: {candidates}")
            return [c for c in candidates if len(c) > 0]
        except Exception as e:
            logger.error(f"POS tagger prediction failed: {e}", exc_info=True)
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

