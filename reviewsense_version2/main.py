from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google.genai import Client, types

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reviewsense_v2.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Validate environment setup
def validate_environment():
    """Validate that all required environment variables are set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY not properly configured in .env file")
        raise ValueError("Please set GEMINI_API_KEY in your .env file")

    logger.info("Environment validation passed")
    return True

# Initialize FastAPI app
app = FastAPI(
    title="ReviewSense Version2 API",
    description="AI-powered review analysis API using Google Gemini",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini AI
try:
    validate_environment()
    API_KEY = os.getenv("GEMINI_API_KEY")
    ai = Client(api_key=API_KEY)
    logger.info("Google Gemini AI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI client: {e}")
    raise

# Pydantic models
class ReviewRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, max_items=100)

    class Config:
        schema_extra = {
            "example": {
                "reviews": [
                    "This product is amazing! Great quality and fast shipping.",
                    "Terrible experience. The item arrived damaged."
                ]
            }
        }

class AnalysisResult(BaseModel):
    reviewText: str = Field(..., description="The original text of the product review")
    sentiment: str = Field(..., description="The sentiment analysis result")
    topics: List[str] = Field(..., description="Key topics identified in the review")

    class Config:
        schema_extra = {
            "example": {
                "reviewText": "This product is amazing! Great quality and fast shipping.",
                "sentiment": "Positive",
                "topics": ["quality", "shipping"]
            }
        }

class Sentiment(str):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: Optional[str] = None

# Schema for Gemini API
review_schema = {
    "type": "object",
    "properties": {
        "reviewText": {
            "type": "string",
            "description": "The original text of the product review."
        },
        "sentiment": {
            "type": "string",
            "enum": [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL],
            "description": "The overall sentiment of the review."
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A specific topic discussed in the review, e.g., quality, shipping, price, customer service."
            },
            "description": "A list of key topics mentioned in the review."
        }
    },
    "required": ["reviewText", "sentiment", "topics"]
}

@app.on_event("startup")
async def startup_event():
    """Validate environment and log startup information."""
    logger.info("🚀 Starting ReviewSense Version2 API")
    validate_environment()
    logger.info("✅ All systems ready")

@app.get("/", summary="API Information")
async def root():
    """Get basic API information."""
    return {
        "message": "ReviewSense Version2 API",
        "version": "2.0.0",
        "description": "AI-powered review analysis using Google Gemini"
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Get service health status."""
    from datetime import datetime
    return HealthResponse(
        status="healthy",
        service="ReviewSense Version2",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/analyze", response_model=List[AnalysisResult], summary="Analyze Reviews")
async def analyze_reviews(request: ReviewRequest):
    """
    Analyze product reviews using Google Gemini AI.

    - **Input**: List of review texts (1-100 reviews)
    - **Output**: Analysis results with sentiment and topics for each review
    - **Processing**: Server-side AI analysis with error handling
    """
    try:
        # Validate and clean input
        if not request.reviews:
            raise HTTPException(status_code=400, detail="No reviews provided")

        # Filter out empty reviews and validate
        reviews = []
        for review in request.reviews:
            cleaned = review.strip()
            if cleaned:
                reviews.append(cleaned)

        if not reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found after cleaning")

        if len(reviews) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 reviews allowed per request")

        logger.info(f"Processing {len(reviews)} reviews for analysis")

        # Prepare data for Gemini
        review_data = [{"reviewText": review} for review in reviews]

        prompt = f"""
        Analyze the following product reviews. For each review, determine its sentiment (Positive, Negative, or Neutral)
        and identify the main topics discussed (e.g., quality, shipping, price, customer service, packaging).

        Return the results as a JSON array where each item has:
        - reviewText: the original review text
        - sentiment: "Positive", "Negative", or "Neutral"
        - topics: array of key topics mentioned

        Reviews to analyze:
        {review_data}
        """

        # Call Gemini API with error handling
        try:
            response = await ai.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': review_schema,
                },
            )
        except Exception as api_error:
            logger.error(f"Gemini API error: {str(api_error)}")
            raise HTTPException(
                status_code=503,
                detail="AI service temporarily unavailable. Please try again later."
            )

        # Parse and validate response
        try:
            import json
            json_string = response.text
            parsed_results = json.loads(json_string)

            # Validate response structure
            if not isinstance(parsed_results, list):
                raise ValueError("API response is not an array")

            if len(parsed_results) != len(reviews):
                logger.warning(f"Response count mismatch: expected {len(reviews)}, got {len(parsed_results)}")

        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse AI response: {json_error}")
            raise HTTPException(
                status_code=502,
                detail="Invalid response from AI service"
            )
        except Exception as parse_error:
            logger.error(f"Response parsing error: {parse_error}")
            raise HTTPException(
                status_code=502,
                detail="Failed to process AI response"
            )

        # Convert to AnalysisResult objects with validation
        results = []
        for i, result in enumerate(parsed_results):
            try:
                # Ensure all required fields are present
                if not all(key in result for key in ["reviewText", "sentiment", "topics"]):
                    logger.warning(f"Missing fields in result {i}: {result}")
                    continue

                # Validate sentiment value
                sentiment = result["sentiment"]
                if sentiment not in [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL]:
                    logger.warning(f"Invalid sentiment value: {sentiment}")
                    sentiment = Sentiment.NEUTRAL  # Default fallback

                results.append(AnalysisResult(
                    reviewText=result["reviewText"],
                    sentiment=sentiment,
                    topics=result["topics"] if isinstance(result["topics"], list) else []
                ))
            except Exception as result_error:
                logger.error(f"Error processing result {i}: {result_error}")
                continue

        if not results:
            raise HTTPException(
                status_code=502,
                detail="No valid analysis results generated"
            )

        logger.info(f"Successfully analyzed {len(results)} reviews")
        return results

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_reviews: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)