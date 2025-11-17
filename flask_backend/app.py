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
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from models import db, User, AnalysisSession, Review, ModelResult, Topic, bcrypt
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/reviewsense')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize database
db.init_app(app)
migrate = Migrate(app, db)

# Create tables
with app.app_context():
    db.create_all()

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "https://*.app.github.dev"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

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
ELECTION_MODE_VALUE = "election-mode"
AVAILABLE_MODEL_KEYS = [
    "arabert-arsas-sa",
    "marbertv2-book-review-sa",
    "xlm-roberta-twitter-sa"
]

@app.route('/auth/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"error": "Email and password are required"}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({"error": "Email already registered"}), 400

        user = User(email=data['email'])
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User registered successfully", "user_id": user.id}), 201
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500

@app.route('/auth/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"error": "Email and password are required"}), 400

        user = User.query.filter_by(email=data['email']).first()
        if user and user.check_password(data['password']):
            return jsonify({
                "message": "Login successful",
                "user_id": user.id,
                "email": user.email
            }), 200
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed"}), 500

@app.route('/analyze_micro_service', methods=["POST"])
def analyze_using_micro_service():
    try:
        if 'text' not in request.form:
            return jsonify({"error": "No reviews provided"}), 400

        review_text = request.form.get('text')
        reviews = [r for r in review_text.split('\n') if r.strip()]
        model_to_use = request.form.get('model')
        user_id = request.form.get('user_id')  # Get user_id from request
        selected_country = request.form.get('country')  # Get country from form (not country_code)
        auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'

        if not isinstance(reviews, list) or not reviews:
            return jsonify({"error": "Reviews must be a non-empty list"}), 400
            
        # Create new analysis session (we'll update detected_dialect after getting results)
        session = AnalysisSession(
            user_id=user_id,
            country_code=selected_country,
            detected_dialect=None  # Will be set after analysis
        )
        db.session.add(session)
        db.session.flush()
        
        # We assume this endpoint is only for Arabic reviews as per your logic
        # You could add a check here if needed.
        
        logger.info(f"📝 Forwarding {len(reviews)} reviews to microservice using model '{model_to_use}'")
        logger.info(f"📝 Selected country: {selected_country}, Auto-detect: {auto_detect}")

        if model_to_use not in AVAILABLE_MODEL_KEYS and model_to_use != ELECTION_MODE_VALUE:
            return jsonify({"error": f"Unknown model '{model_to_use}' requested"}), 400

        election_mode_enabled = model_to_use == ELECTION_MODE_VALUE
        models_for_run = AVAILABLE_MODEL_KEYS if election_mode_enabled else [model_to_use]

        detected_dialects = []  # Collect dialects from all reviews
        structured_results = []
        for review_text in reviews:
            if not isinstance(review_text, str) or not review_text.strip():
                continue # Skip empty reviews

            # Create review record
            review = Review(
                session_id=session.id,
                review_text=review_text,
                language='ar' if is_arabic(review_text) else 'en'
            )
            db.session.add(review)
            db.session.flush()

            best_result_payload = None
            best_topics_payload = []
            best_model_name = None
            best_dialect = None
            best_score = -1.0

            for candidate_model in models_for_run:
                payload = {
                    "text": review_text,
                    "model": candidate_model,
                    "autoDetectDialect": auto_detect,
                    "country": selected_country if selected_country else None
                }

                try:
                    response = requests.post(ANALYSIS_MICROSERVICE_URL, json=payload, timeout=30)
                    if response.status_code != 200:
                        error_info = response.json() if response.content else {"error": "Unknown microservice error"}
                        logger.error(
                            f"❌ Microservice error for review '{review_text[:50]}...' "
                            f"with model '{candidate_model}': {response.status_code} - {error_info}"
                        )
                        continue

                    result_data = response.json()
                    model_used = result_data.get('model_used', candidate_model)

                    model_result = ModelResult(
                        session_id=session.id,
                        review_id=review.id,
                        model_name=model_used,
                        sentiment=result_data.get('sentiment', {}).get('label'),
                        sentiment_score=result_data.get('sentiment', {}).get('score')
                    )
                    db.session.add(model_result)

                    current_score = result_data.get('sentiment', {}).get('score')
                    try:
                        numeric_score = float(current_score)
                    except (TypeError, ValueError):
                        numeric_score = -1.0

                    if numeric_score > best_score:
                        best_score = numeric_score
                        best_result_payload = result_data
                        best_model_name = model_used
                        best_topics_payload = result_data.get('topics', [])
                        best_dialect = result_data.get('dialect')

                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"❌ Could not connect to microservice for model '{candidate_model}': {e}"
                    )
                    continue

            if not best_result_payload:
                logger.error(f"❌ All models failed for review '{review_text[:80]}'")
                raise ValueError("All sentiment models failed for at least one review.")

            chosen_model_name = best_model_name or models_for_run[0]

            structured_results.append({
                "reviewText": best_result_payload.get("original_text", review_text),
                "sentiment": best_result_payload.get("sentiment", {}).get("label"),
                "sentimentScore": best_result_payload.get("sentiment", {}).get("score"),
                "topics": best_topics_payload,
                "modelUsed": chosen_model_name
            })

            if best_dialect:
                detected_dialects.append(best_dialect)

            for topic_data in best_topics_payload:
                if isinstance(topic_data, dict):
                    topic_text = topic_data.get('topic', '')
                    topic_score = topic_data.get('score', 1.0)
                else:
                    topic_text = str(topic_data)
                    topic_score = 1.0

                topic = Topic(
                    review_id=review.id,
                    topic_text=topic_text,
                    score=topic_score
                )
                db.session.add(topic)

        logger.info(f"✅ Successfully processed {len(structured_results)} reviews via microservice.")
        
        # Determine the detected dialect (most common dialect from all reviews, or from first review)
        detected_dialect = None
        if detected_dialects:
            # Use the most common dialect, or the first one if all are unique
            from collections import Counter
            dialect_counts = Counter(detected_dialects)
            detected_dialect = dialect_counts.most_common(1)[0][0] if dialect_counts else None
            logger.info(f"📝 Detected dialects: {detected_dialects}, Selected: {detected_dialect}")
        
        # Update session with detected dialect
        session.detected_dialect = detected_dialect
        db.session.add(session)
        
        # restructure the result: { reviewText: string; sentiment: Sentiment; topics: string[]; }
        # structured result with model 
        full_results = {
            "model": ELECTION_MODE_VALUE if election_mode_enabled else model_to_use, 
            "mode": "election" if election_mode_enabled else "single",
            "modelsConsidered": models_for_run,
            "results": structured_results,
            "selectedCountry": selected_country,
            "detectedDialect": detected_dialect
        }
        logger.info(f"Microservice results: {full_results}")
        


        # Commit all database changes
        db.session.commit()
        
        # Add session_id to the response
        full_results['session_id'] = session.id
        return jsonify(full_results)

    except Exception as e:
        logger.error(f"❌ Unhandled error in /analyze_micro_service: {str(e)}")
        db.session.rollback()  # Rollback on error
        return jsonify({"error": "Internal server error"}), 500

@app.route('/analysis/session/<int:session_id>', methods=['GET'])
def get_session_results(session_id):
    try:
        session = AnalysisSession.query.get_or_404(session_id)
        
        results = []
        models_used = set()
        for review in session.reviews:
            best_model_result = None
            for model_result in review.model_results:
                models_used.add(model_result.model_name)
                if (
                    best_model_result is None
                    or (model_result.sentiment_score or 0) > (best_model_result.sentiment_score or 0)
                ):
                    best_model_result = model_result

            if best_model_result:
                topics = [{"topic": topic.topic_text, "score": topic.score} for topic in review.topics]
                results.append({
                    "reviewText": review.review_text,
                    "sentiment": best_model_result.sentiment,
                    "sentimentScore": best_model_result.sentiment_score,
                    "topics": topics,
                    "modelUsed": best_model_result.model_name
                })

        models_used_list = sorted(models_used)
        mode = "election" if len(models_used_list) > 1 else "single"
        model_label = "election-mode" if mode == "election" else (models_used_list[0] if models_used_list else "")

        return jsonify({
            "results": results,
            "model": model_label,
            "mode": mode,
            "modelsConsidered": models_used_list,
            "selectedCountry": session.country_code,
            "detectedDialect": session.detected_dialect
        })
    except Exception as e:
        logger.error(f"❌ Error fetching session results: {str(e)}")
        return jsonify({"error": "Failed to fetch session results"}), 500

@app.route('/analysis/history/<int:user_id>', methods=['GET'])
def get_analysis_history(user_id):
    try:
        # Get page and limit from query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('limit', 10, type=int)

        # Query analysis sessions for the user with pagination
        sessions = AnalysisSession.query.filter_by(user_id=user_id)\
            .order_by(AnalysisSession.created_at.desc())\
            .paginate(page=page, per_page=per_page)

        history = []
        for session in sessions.items:
            session_data = {
                'session_id': session.id,
                'created_at': session.created_at.isoformat(),
                'country_code': session.country_code,
                'detected_dialect': session.detected_dialect,
                'reviews_count': len(session.reviews),
                'models_used': []
            }
            
            # Get unique models used in this session
            models_used = set()
            for review in session.reviews:
                for result in review.model_results:
                    models_used.add(result.model_name)
            session_data['models_used'] = list(models_used)
            
            history.append(session_data)

        return jsonify({
            'history': history,
            'pagination': {
                'total': sessions.total,
                'pages': sessions.pages,
                'current_page': sessions.page,
                'per_page': sessions.per_page
            }
        })
    except Exception as e:
        logger.error(f"❌ Unhandled error in /getting hisgtory: {str(e)}")

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

        