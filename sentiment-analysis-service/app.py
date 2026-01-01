from __future__ import annotations

import json
import logging
from typing import Any, List
from datetime import datetime

from flask import Flask, jsonify, request, render_template

# Import models and utilities
from sent_models import (
    AVAILABLE_MODELS,
    DIALECT_MODEL,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    load_embedder,
    ArabicPOSTagger,
)
from utils import (
    clean_text,
    detect_language,
    normalize_arabic,
    extract_topics,
)

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

app = Flask(__name__)

# Initialize POS tagger instance
POS_TAGGER = ArabicPOSTagger()
ELECTION_MODE_VALUE = "election-mode"
AVAILABLE_MODEL_KEYS: List[str] = list(AVAILABLE_MODELS.keys())

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
    auto_detect_dialect = True
    country_code = None
    if request.is_json:
        auto_detect_dialect = request.json.get('autoDetectDialect', True) if request.json else True
        country_code = request.json.get('country') if request.json else None
    
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
    requested_model = payload.get("model", "")

    if not isinstance(text, str) or not text.strip():
        logger.error(f"[{req_id}] Empty or invalid input text received")
        return jsonify({"error": "Empty input text.", "code": "empty_text"}), 400

    if requested_model != ELECTION_MODE_VALUE and requested_model not in AVAILABLE_MODELS:
        logger.error(f"[{req_id}] Unknown model requested: {requested_model}")
        return jsonify({"error": "Unknown model.", "code": "unknown_model"}), 400

    # Text preprocessing pipeline
    logger.info(f"[{req_id}] Starting text preprocessing pipeline")
    logger.info(f"[{req_id}] Original text ({len(text)} chars): {text[:200]}...")
    
    original_text = text
    cleaned = clean_text(original_text)
    logger.info(f"[{req_id}] Cleaned text ({len(cleaned)} chars): {cleaned[:200]}...")
    
    # Detect language - returns None if detection fails, or language code (may be unsupported)
    detected_lang = detect_language(cleaned)
    
    # Validate that only Arabic or English languages are accepted
    if detected_lang is None:
        # Language detection failed - we cannot determine the language
        logger.error(f"[{req_id}] Language detection failed. Unable to determine language. Only Arabic and English are supported.")
        return jsonify({
            "error": "The input language is not supported. Only Arabic and English languages are allowed.",
            "code": "language_detection_failed"
        }), 400
    
    if detected_lang not in ["ar", "en"]:
        # Unsupported language detected
        logger.error(f"[{req_id}] Unsupported language detected: {detected_lang}. Only Arabic (ar) and English (en) are allowed.")
        return jsonify({
            "error": f"The input language is not supported. Detected language: {detected_lang}. Only Arabic and English languages are allowed.",
            "code": "unsupported_language",
            "detected_language": detected_lang
        }), 400
    
    lang = detected_lang
    logger.info(f"[{req_id}] Detected language: {lang}")

    def run_model(candidate_name: str):
        """Run sentiment model with fallback handling."""
        candidate_model = AVAILABLE_MODELS[candidate_name]
        try:
            result = candidate_model.predict(cleaned, lang_hint=lang)
            return result, candidate_name
        except Exception as err:
            fallback_model = "arabert-arsas-sa" if candidate_name != "arabert-arsas-sa" else "marbertv2-book-review-sa"
            fallback = AVAILABLE_MODELS.get(fallback_model)
            if fallback:
                try:
                    fallback_result = fallback.predict(cleaned, lang_hint=lang)
                    logger.warning(
                        f"[{req_id}] Model '{candidate_name}' failed ({err}); "
                        f"fallback '{fallback_model}' succeeded."
                    )
                    return fallback_result, fallback_model
                except Exception as fallback_err:
                    logger.error(
                        f"[{req_id}] Fallback model '{fallback_model}' also failed: {fallback_err}"
                    )
                    raise
            raise

    candidate_models = AVAILABLE_MODEL_KEYS if requested_model == ELECTION_MODE_VALUE else [requested_model]
    best_sentiment = None
    best_model_name = None
    best_score = -1.0

    for candidate in candidate_models:
        try:
            sentiment_candidate, used_model_name = run_model(candidate)
        except Exception as e:
            logger.error(
                f"[{req_id}] Failed to run model '{candidate}': {e}",
                exc_info=True
            )
            continue

        score = float(sentiment_candidate.get("score", 0))
        if score > best_score:
            best_score = score
            best_sentiment = sentiment_candidate
            best_model_name = used_model_name

    if best_sentiment is None or best_model_name is None:
        return jsonify({"error": "All models failed to process the text.", "code": "model_error"}), 500

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
        topics = extract_topics(cleaned, lang, pos_tagger=POS_TAGGER)
        logger.info(f"[{req_id}] Topic extraction completed: {len(topics)} topics found")
    except Exception as e:
        logger.error(f"[{req_id}] Topic extraction failed: {e}", exc_info=True)
        topics = []

    # Build response in the exact order specified
    response = {
        "original_text": original_text,
        "cleaned_text": normalize_arabic(cleaned) if lang == "ar" else cleaned,
        "language": lang,
        "dialect": dialect,
        "sentiment": {
            "label": best_sentiment["label"],
            "score": float(best_sentiment["score"]),
        },
        "topics": topics,
        "model_used": best_model_name,
        "mode": "election" if len(candidate_models) > 1 else "single",
        "models_considered": candidate_models,
    }
    
    # Log final results
    logger.info(f"[{req_id}] ===== Analysis Results =====")
    logger.info(f"[{req_id}] Language: {lang}")
    logger.info(f"[{req_id}] Dialect: {dialect}")
    logger.info(f"[{req_id}] Sentiment: {best_sentiment['label']} (score: {best_sentiment['score']:.4f})")
    logger.info(f"[{req_id}] Topics found: {len(topics)}")
    for idx, topic in enumerate(topics, 1):
        logger.info(f"[{req_id}]   Topic {idx}: {topic['topic']} (score: {topic['score']:.4f})")
    logger.info(f"[{req_id}] Model used: {best_model_name}")
    logger.info(f"[{req_id}] Mode: {'election' if len(candidate_models) > 1 else 'single'}; Models tried: {candidate_models}")
    
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

    # Check if transformers components are available (from global imports)
    if AutoModelForSequenceClassification is None or AutoTokenizer is None or pipeline is None:
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
