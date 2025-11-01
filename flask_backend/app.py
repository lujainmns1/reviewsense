# reviewsense/flask_backend/app.py
from flask import Flask, request, jsonify, make_response
import requests
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
# try:
#     # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#     # if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
#     #     raise ValueError("GEMINI_API_KEY not set properly")

#     # genai.configure(api_key=GEMINI_API_KEY)
#     # model = genai.GenerativeModel('gemini-2.5-flash')
#     # logger.info("Gemini AI initialized successfully")
# except Exception as e:
#     logger.error(f"Failed to initialize Gemini: {e}")
#     raise

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



ANALYSIS_MICROSERVICE_URL = "http://127.0.0.1:5001/analyze"  # URL of the microservice

@app.route('/analyze_micro_service', methods=["POST"])
def analyze_using_micro_service():
    try:
        data = request.get_json()
        if not data or 'reviews' not in data:
            return jsonify({"error": "No reviews provided"}), 400

        reviews = data.get('reviews', [])
        # You can also let the user choose the model from the frontend
        model_to_use = data.get('model') 

        if not isinstance(reviews, list) or not reviews:
            return jsonify({"error": "Reviews must be a non-empty list"}), 400
        
        # We assume this endpoint is only for Arabic reviews as per your logic
        # You could add a check here if needed.
        
        logger.info(f"📝 Forwarding {len(reviews)} reviews to microservice using model '{model_to_use}'")

        results = []
        for review_text in reviews:
            if not isinstance(review_text, str) or not review_text.strip():
                continue # Skip empty reviews

            payload = {
                "text": review_text,
                "model": model_to_use
            }

            try:
                # Set a timeout for the request to avoid waiting forever
                response = requests.post(ANALYSIS_MICROSERVICE_URL, json=payload, timeout=30)
                
                # Check for successful response
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    # Log the error and add a placeholder result
                    error_info = response.json() if response.content else {"error": "Unknown microservice error"}
                    logger.error(f"❌ Microservice error for review '{review_text[:50]}...': {response.status_code} - {error_info}")
                    results.append({
                        "original_text": review_text,
                        "error": f"Failed to analyze: {error_info.get('error', 'Service unavailable')}",
                        "status_code": response.status_code
                    })
                    
            except requests.exceptions.RequestException as e:
                # Handle connection errors, timeouts, etc.
                logger.error(f"❌ Could not connect to microservice: {e}")
                results.append({
                    "original_text": review_text,
                    "error": "Microservice is unreachable.",
                    "status_code": 503 # Service Unavailable
                })

        logger.info(f"✅ Successfully processed {len(results)} reviews via microservice.")
        # restructure the result: { reviewText: string; sentiment: Sentiment; topics: string[]; }
        structured_results = []
        for res in results:
            if "original_text" in res and "sentiment" in res and "topics" in res:
                structured_results.append({
                    "reviewText": res["original_text"],
                    "sentiment": res["sentiment"]["label"],
                    "sentimentScore": res["sentiment"]["score"],
                    "topics": res["topics"]
                })
        # structured_results.append({"model":model_to_use})
        # structured result with model 
        full_results = {"model": model_to_use, "results": structured_results}
        logger.info(f"Microservice results: {full_results}")
        


        return jsonify(full_results)

    except Exception as e:
        logger.error(f"❌ Unhandled error in /analyze_micro_service: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting ReviewSense Flask API...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Press Ctrl+C to stop")

    # Get host and port from environment variables with defaults
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    # debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

        