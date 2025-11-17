from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Optional dependencies - imported separately so they don't break core functionality
try:
    import langdetect
    logger.info("langdetect loaded successfully")
except ImportError:
    logger.warning("langdetect not available - language detection will be disabled")
    langdetect = None  # type: ignore

try:
    import yake
    logger.info("yake loaded successfully")
except ImportError:
    logger.warning("yake not available - YAKE keyword extraction will be disabled")
    yake = None  # type: ignore

try:
    from scipy.spatial.distance import cosine
    logger.info("scipy loaded successfully")
except ImportError:
    logger.warning("scipy not available - cosine similarity will be disabled")
    cosine = None  # type: ignore

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
# Fallback YAKE
# ---------------------------

def _topics_yake(text: str, lang: str = "ar", top_k: int = 5) -> List[Tuple[str, float]]:
    """Fallback: Run YAKE with n-grams 1-2 only, normalize and merge near-duplicates."""
    if yake is None:
        logger.warning("YAKE not available, using simple keyword extraction")
        return []
    
    try:
        t = normalize_arabic(text) if lang == "ar" else text.lower()
        logger.info(f"Running YAKE on normalized text: {t[:100]}...")
        candidates: List[Tuple[str, float]] = []
        for n in (1, 2):  # Limit to 1-2 grams for shorter topics
            try:
                kw = yake.KeywordExtractor(lan=lang, n=n, top=top_k*2, features=None).extract_keywords(t)
                logger.info(f"YAKE n={n} extracted {len(kw)} keywords")
                candidates.extend(kw)
            except Exception as e:
                logger.warning(f"YAKE extraction failed for n={n}: {e}")
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
        result = boosted[:top_k]
        logger.info(f"YAKE final results: {result}")
        return result
    except Exception as e:
        logger.error(f"YAKE extraction failed: {e}", exc_info=True)
        return []

_MODELS_MODULE = None


def _load_models_module():
    """Attempt to load the local sentiment service models module regardless of cwd."""
    global _MODELS_MODULE
    if _MODELS_MODULE is not None:
        return _MODELS_MODULE

    service_dir = Path(__file__).resolve().parent
    attempts = []

    def _try_standard_import():
        try:
            import models  # type: ignore
            return models
        except ModuleNotFoundError as exc:
            attempts.append(str(exc))
            return None

    module = _try_standard_import()
    if module:
        _MODELS_MODULE = module
        return module

    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))
        module = _try_standard_import()
        if module:
            _MODELS_MODULE = module
            return module

    models_path = service_dir / "models.py"
    if models_path.exists():
        spec = importlib.util.spec_from_file_location("sentiment_service_models", models_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                sys.modules["models"] = module
                _MODELS_MODULE = module
                return module
            except Exception as exc:
                attempts.append(f"importlib load failed: {exc}")

    logger.error("Unable to import topic extraction dependencies: %s", "; ".join(attempts))
    return None


# ---------------------------
# Topic Extraction (POS-guided for Arabic)
# ---------------------------

def _extract_simple_topics(text: str, lang: Optional[str]) -> List[Dict[str, Any]]:
    """Simple keyword-based topic extraction as final fallback."""
    logger.info("Using simple keyword extraction as fallback")
    t_norm = normalize_arabic(text) if lang == "ar" else text.lower()
    
    # Extract words that match aspect boost list
    found_topics = []
    words = t_norm.split()
    
    for aspect in _AR_ASPECT_BOOST:
        if aspect in t_norm:
            # Find the phrase containing the aspect
            aspect_lower = aspect.lower()
            for i, word in enumerate(words):
                if aspect_lower in word.lower() or word.lower() in aspect_lower:
                    # Try to get a 2-word phrase around it
                    start = max(0, i-1)
                    end = min(len(words), i+2)
                    phrase = ' '.join(words[start:end])
                    if phrase and len(phrase) > 2:
                        found_topics.append((phrase, 0.85))
                        break
    
    # Also look for common product/service related words directly in text
    common_topics = {
        "المنتج": "المنتج", "السعر": "السعر", "الجودة": "الجودة",
        "الخدمة": "الخدمة", "التوصيل": "التوصيل", "الشحن": "الشحن",
        "التصميم": "التصميم", "الأداء": "الأداء", "النظافة": "النظافة",
        "خدمة": "خدمة العملاء", "عملاء": "خدمة العملاء"
    }
    
    for key, topic in common_topics.items():
        if key in t_norm and topic not in [t[0] for t in found_topics]:
            found_topics.append((topic, 0.75))
    
    # Remove duplicates and sort
    unique_topics = {}
    for topic, score in found_topics:
        if topic not in unique_topics or unique_topics[topic] < score:
            unique_topics[topic] = score
    
    result = [{"topic": t, "score": round(float(s), 4)} for t, s in sorted(unique_topics.items(), key=lambda x: x[1], reverse=True)[:3]]
    logger.info(f"Simple extraction found {len(result)} topics: {result}")
    return result

def extract_topics(text: str, lang: Optional[str], pos_tagger=None) -> List[Dict[str, Any]]:
    """POS-guided topic extraction for Arabic (noun/adj phrases), re-ranked by BERT similarity. Fallback to YAKE. Top 3 only."""
    models_module = _load_models_module()
    if models_module is None:
        return _extract_simple_topics(text, lang)

    from models import ARABERT_EMBEDDER, load_embedder, _get_text_embedding, ArabicPOSTagger  # type: ignore
    
    # Use provided POS tagger or create a new one
    if pos_tagger is None:
        pos_tagger = ArabicPOSTagger()
    
    logger.info(f"Starting topic extraction for text: {text[:100]}...")
    logger.info(f"Language: {lang}")
    
    lang_eff = lang if lang in {"ar", "en"} else "ar"
    top_k = 3

    if lang_eff != "ar":
        # Non-Arabic: YAKE fallback
        logger.info("Using YAKE for non-Arabic text")
        pairs = _topics_yake(text, lang=lang_eff, top_k=top_k)
        logger.info(f"YAKE extracted {len(pairs)} initial candidates")
        if pairs:
            return [{"topic": p, "score": round(float(s), 4)} for p, s in pairs]
        # Fallback to simple extraction for non-Arabic too
        return _extract_simple_topics(text, lang_eff)

    # Arabic: POS extraction
    logger.info("Using POS tagger for Arabic text")
    t_norm = normalize_arabic(text)
    logger.info(f"Normalized text for POS: {t_norm[:100]}...")
    
    try:
        candidates = pos_tagger.predict(t_norm)
        logger.info(f"POS tagger found {len(candidates)} candidate phrases: {candidates}")
    except Exception as e:
        logger.warning(f"POS tagger failed: {e}, falling back to YAKE")
        candidates = []
    
    if not candidates:
        # Fallback to YAKE
        logger.info("No POS candidates found, falling back to YAKE")
        pairs = _topics_yake(text, lang=lang_eff, top_k=top_k)
        logger.info(f"YAKE extracted {len(pairs)} fallback candidates")
        if pairs:
            return [{"topic": p, "score": round(float(s), 4)} for p, s in pairs]
        # Final fallback: simple keyword extraction
        logger.info("YAKE also returned no results, using simple keyword extraction")
        return _extract_simple_topics(text, lang_eff)

    # Re-rank with BERT if available
    try:
        load_embedder()
        if ARABERT_EMBEDDER is None or cosine is None:
            # Use uniform scores or simple boost
            logger.info("BERT embedder not available, using uniform scores")
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
                logger.info("Failed to get text embedding, using uniform scores")
                boosted = [(p, 0.8 + 0.2 if any(a in p for a in _AR_ASPECT_BOOST) else 0.8) for p in candidates]
                boosted = [(p, min(1.0, s)) for p, s in boosted]
            else:
                scores = []
                for phrase in candidates:
                    try:
                        phrase_emb = _get_text_embedding(tokenizer, model, phrase)
                        if phrase_emb is not None and text_emb is not None:
                            # Both are numpy arrays (from .cpu().numpy())
                            if hasattr(text_emb, 'shape') and hasattr(phrase_emb, 'shape'):
                                if text_emb.shape == phrase_emb.shape:
                                    sim = 1 - cosine(text_emb, phrase_emb)
                                else:
                                    logger.warning(f"Shape mismatch: text_emb {text_emb.shape} vs phrase_emb {phrase_emb.shape}")
                                    sim = 0.5
                            else:
                                sim = 0.5
                        else:
                            sim = 0.5
                    except Exception as e:
                        logger.warning(f"Failed to compute embedding for phrase '{phrase}': {e}")
                        sim = 0.5
                    scores.append((phrase, sim))
                
                # Boost and sort
                boosted = []
                for p, s in scores:
                    boost = 0.2 if any(a in p for a in _AR_ASPECT_BOOST) else 0.0
                    boosted.append((p, min(1.0, s + boost)))

        boosted.sort(key=lambda x: x[1], reverse=True)
        result = [{"topic": p, "score": round(s, 4)} for p, s in boosted[:top_k]]
        logger.info(f"Final topics extracted: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in topic re-ranking: {e}", exc_info=True)
        # Return simple scored candidates as fallback
        boosted = [(p, 0.8 + (0.2 if any(a in p for a in _AR_ASPECT_BOOST) else 0.0)) for p in candidates[:top_k]]
        return [{"topic": p, "score": round(min(1.0, s), 4)} for p, s in boosted]

# Export constants for use in models.py
__all__ = [
    'normalize_arabic',
    'clean_text',
    'detect_language',
    'extract_topics',
    '_AR_ABBREV_POS',
    '_AR_ABBREV_NEG',
    '_AR_HEDGE_NEUTRAL',
]

