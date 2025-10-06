from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from arabic_model_service import analyze_arabic_review

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Helper function to detect Arabic text
def is_arabic(text):
    """Check if text contains Arabic characters"""
    return bool(re.search(r'[\u0600-\u06FF]', text))

# Initialize Gemini AI
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not set properly")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    raise

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
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

@app.route('/analyze', methods=['POST'])
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

        # Process Arabic reviews with local model
        arabic_results = []
        for review in arabic_reviews:
            sentiment_result = analyze_arabic_review(review)
            arabic_results.append({
                'reviewText': review,
                'sentiment': sentiment_result['label'].capitalize(),  # POSITIVE -> Positive
                'topics': []  # Arabic model doesn't extract topics yet
            })

        # Process non-Arabic reviews with Gemini
        gemini_results = []
        if non_arabic_reviews:
            reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(non_arabic_reviews)])

        prompt = f"""
        Analyze the following product reviews. For each review, provide:
        1. Sentiment: "Positive", "Negative", or "Neutral"
        2. Key topics: List of main topics discussed (e.g., quality, price, shipping, customer service)

        Return the results as a JSON array with this exact format:
        [
            {{
                "reviewText": "original review text",
                "sentiment": "Positive/Negative/Neutral",
                "topics": ["topic1", "topic2"]
            }}
        ]

        Reviews to analyze:
        {reviews_text}
        """

        # Call Gemini API for non-Arabic reviews
        if non_arabic_reviews:
            response = model.generate_content(prompt)

            if not response.text:
                return jsonify({"error": "No response from AI service"}), 502

            # Parse JSON response
            import json
            try:
                results = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw response: {response.text}")
                return jsonify({"error": "Invalid response format from AI"}), 502

            # Validate results
            if not isinstance(results, list):
                return jsonify({"error": "Response must be an array"}), 502

            # Ensure we have results for all non-Arabic reviews
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

                gemini_results.append({
                    'reviewText': review_text,
                    'sentiment': sentiment,
                    'topics': topics
                })

        # Combine results: Arabic first, then non-Arabic
        validated_results = arabic_results + gemini_results

        if not validated_results:
            return jsonify({"error": "No valid analysis results"}), 502

        logger.info(f"✅ Successfully analyzed {len(validated_results)} reviews ({len(arabic_results)} Arabic, {len(gemini_results)} others)")
        return jsonify(validated_results)

    except Exception as e:
        logger.error(f"❌ Error in analyze_reviews: {str(e)}")
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