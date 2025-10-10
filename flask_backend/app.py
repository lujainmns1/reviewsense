from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import google.generativeai as genai
import os
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from arabic_model_service import analyze_arabic_review
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch 
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS - Allow specific GitHub Codespace origin
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin and '.app.github.dev' in origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

CORS(app)

# Helper function to detect Arabic text
def is_arabic(text):
    """Check if text contains Arabic characters"""
    return bool(re.search(r'[\u0600-\u06FF]', text))


# Helper to extract JSON array from AI responses that may include markdown fences
def extract_json_array_from_text(text: str) -> str:
    """Try to clean the AI response and extract a JSON array substring.

    This removes markdown code fences (```json ... ```), any leading/trailing
    whitespace, and returns the substring between the first '[' and the last ']'
    if present. Otherwise returns the stripped text.
    """
    if not text:
        return ''
    t = text.strip()
    # Remove fenced code blocks with optional language tag (e.g. ```json)
    t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*```$', '', t)
    # If the model returned other surrounding text, try to find the JSON array
    start = t.find('[')
    end = t.rfind(']')
    if start != -1 and end != -1 and end > start:
        return t[start:end+1]
    return t

# Initialize Gemini AI
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not set properly")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    raise

@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    """API information endpoint"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    return jsonify({
        "message": "ReviewSense Flask API",
        "version": "1.0.0",
        "description": "AI-powered review analysis supporting Arabic and English using local models and Google Gemini"
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ReviewSense Flask API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_reviews():
    """Analyze product reviews using Google Gemini AI"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data or 'reviews' not in data:
            return jsonify({"error": "No reviews provided"}), 400

        reviews = data['reviews']

        if not isinstance(reviews, list) or len(reviews) == 0:
            return jsonify({"error": "Reviews must be a non-empty list"}), 400

        if len(reviews) > 100:
            return jsonify({"error": "Maximum 100 reviews allowed"}), 400

        # Clean and validate reviews
        cleaned_reviews = []
        for review in reviews:
            if isinstance(review, str) and review.strip():
                cleaned_reviews.append(review.strip())

        if not cleaned_reviews:
            return jsonify({"error": "No valid reviews found"}), 400

        logger.info(f"📝 Processing {len(cleaned_reviews)} reviews")

        # Separate Arabic and non-Arabic reviews
        arabic_reviews = []
        non_arabic_reviews = []

        for review in cleaned_reviews:
            if is_arabic(review):
                arabic_reviews.append(review)
            else:
                non_arabic_reviews.append(review)

        logger.info(f"📝 Arabic reviews: {len(arabic_reviews)}, Non-Arabic reviews: {len(non_arabic_reviews)}")

        # Process all reviews (both Arabic and non-Arabic) with Gemini
        all_results = []
        if cleaned_reviews:  # Process all reviews together
            reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(cleaned_reviews)])

        prompt = f"""
       Analyze the following product reviews. For each review, determine its sentiment (Positive, Negative, or Neutral) 
    and identify the main topics discussed (e.g., quality, shipping, price, customer service, packaging).
        


        Return the results as a JSON array with this exact format:
        [
            {{
                "reviewText": "original review text",
                "sentiment": "Positive/Neutral/Negative",
                "topics": ["topic1", "topic2"]
            }}
        ]

        Reviews to analyze:
        {reviews_text}
        Be concise and ensure the JSON is properly formatted.
        """

        # Call Gemini API for all reviews
        if cleaned_reviews:
            response = model.generate_content(prompt)

            if not response.text:
                return jsonify({"error": "No response from AI service"}), 502

            # Parse JSON response (sanitize possible markdown/code fences)
            import json
            cleaned = extract_json_array_from_text(response.text)
            logger.info(f"Cleaned AI response for parsing: {cleaned[:500]}")
            try:
                results = json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw response: {response.text}")
                return jsonify({"error": "Invalid response format from AI"}), 502

            # Validate results
            if not isinstance(results, list):
                return jsonify({"error": "Response must be an array"}), 502

            # Process all reviews
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    continue

                # Validate required fields
                review_text = result.get('reviewText', '')
                sentiment = result.get('sentiment', 'Neutral')
                topics = result.get('topics', [])

                # Validate sentiment
                if sentiment not in ['Positive', 'Negative', 'Neutral']:
                    sentiment = 'Neutral'

                # Ensure topics is a list
                if not isinstance(topics, list):
                    topics = []

                all_results.append({
                    'reviewText': review_text,
                    'sentiment': sentiment,
                    'topics': topics
                })

        # Use results directly since we processed all reviews together
        validated_results = all_results

        if not validated_results:
            return jsonify({"error": "No valid analysis results"}), 502

        logger.info(f"✅ Successfully analyzed {len(validated_results)} reviews ({len(arabic_reviews)} Arabic, {len(non_arabic_reviews)} others)")
        return jsonify(validated_results)

    except Exception as e:
        logger.error(f"❌ Error in analyze_reviews: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
# analyze_reviews using  tabularisai / multilingual-sentiment-analysis
# @app.route('/multi_lang', methods=['GET'])
def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

# Global model and tokenizer for multilingual sentiment analysis
MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
@app.route('/multi_lang', methods=['GET'])
def multi_lang():
    """Test multilingual sentiment analysis using tabularisai/multilingual-sentiment-analysis"""
    try:

        texts = [
            "I love this product! It works great and exceeds my expectations.",
            "هذا المنتج سيء للغاية ولا يعمل كما هو معلن.",
            "C'est un produit moyen, rien de spécial.",
            "Das Produkt ist fantastisch und sehr nützlich.",
            "El producto no cumple con lo prometido."
        ]
        for text, sentiment in zip(texts, predict_sentiment(texts)):
            print(f"Text: {text}\nSentiment: {sentiment}\n")

        results = [{"text": text, "sentiment": sentiment} for text, sentiment in zip(texts, predict_sentiment(texts))]
        return jsonify(results)

    except Exception as e:
        logger.error(f"❌ Error in multi_lang: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("Starting ReviewSense Flask API...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Press Ctrl+C to stop")

    # Get host and port from environment variables with defaults
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

