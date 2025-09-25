# ReviewSense Version2 Backend

A modern FastAPI backend for ReviewSense that provides AI-powered review analysis using Google Gemini.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs with auto-generated documentation
- **Google Gemini Integration**: Server-side AI analysis for better security and API key protection
- **CORS Support**: Configured for seamless frontend-backend communication
- **Type Safety**: Full Pydantic models for request/response validation with examples
- **Async Support**: Asynchronous API endpoints for better performance and scalability
- **Robust Error Handling**: Comprehensive error handling with detailed logging
- **Health Checks**: Built-in health monitoring and status endpoints
- **Environment Validation**: Automatic validation of configuration on startup
- **Automated Setup**: One-command setup script for easy installation
- **Request Limits**: Built-in validation for request size and content

## API Endpoints

### GET /
- **Description**: Root endpoint with API information
- **Response**: Basic API info and version

### GET /health
- **Description**: Health check endpoint
- **Response**: Service status information

### POST /analyze
- **Description**: Analyze product reviews using Google Gemini AI
- **Request Body**:
  ```json
  {
    "reviews": ["review text 1", "review text 2", ...]
  }
  ```
- **Response**: Array of analysis results with sentiment and topics

## Setup and Installation

### Quick Setup (Recommended)

1. **Run the automated setup script**:
   ```bash
   python setup.py
   ```
   This script will:
   - Check your Python version
   - Install all dependencies
   - Validate your environment configuration
   - Start the server automatically

### Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   - Edit the `.env` file and set your `GEMINI_API_KEY`

3. **Run the server**:
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## Configuration

- **GEMINI_API_KEY**: Your Google Gemini API key (required)
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **CORS Origins**: http://localhost:5173, http://localhost:3000

## Technologies Used

- **Framework**: FastAPI
- **AI**: Google Gemini AI
- **Validation**: Pydantic
- **Server**: Uvicorn
- **Environment**: python-dotenv