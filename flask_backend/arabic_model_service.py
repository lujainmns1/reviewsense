# arabic_model_service.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import nltk
from nltk.corpus import stopwords

# Download Arabic stopwords with error handling
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK stopwords: {e}")

# Load a proper Arabic sentiment analysis model
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    arabic_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)  # Use CPU
    print("Arabic sentiment model loaded successfully")
except Exception as e:
    print(f"Failed to load Arabic model: {e}")
    arabic_pipeline = None

# ---------- Arabic text preprocessing ----------
def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-Arabic characters but keep basic punctuation
    text = re.sub(r'[^\u0600-\u06FF\s\.,!?]', '', text)
    return text

# ---------- Review analysis function ----------
def analyze_arabic_review(text):
    if arabic_pipeline is None:
        return {"label": "NEUTRAL", "score": 0.5}

    try:
        cleaned_text = clean_arabic_text(text)
        if not cleaned_text:
            return {"label": "NEUTRAL", "score": 0.5}
        
        result = arabic_pipeline(cleaned_text)[0]

        # Map labels to standard format
        label_mapping = {
            "positive": "POSITIVE",
            "negative": "NEGATIVE",
            "neutral": "NEUTRAL"
        }

        return {
            "label": label_mapping.get(result["label"].lower(), "NEUTRAL"),
            "score": result["score"]
        }
    except Exception as e:
        print(f"Error analyzing Arabic text: {e}")
        return {"label": "NEUTRAL", "score": 0.5}
