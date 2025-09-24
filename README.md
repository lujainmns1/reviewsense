<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# ReviewSense

An intelligent web application that analyzes product reviews using sentiment analysis and topic extraction powered by Google Gemini AI.

## Features

- **AI-Powered Analysis**: Uses Google Gemini AI to analyze review sentiment (Positive, Negative, Neutral)
- **Topic Extraction**: Identifies key topics discussed in reviews (e.g., quality, shipping, price, customer service)
- **Modern React Interface**: Clean and responsive user interface built with React and TypeScript
- **Multiple Input Methods**: Support for both text input and CSV file uploads
- **Real-time Processing**: Fast analysis with detailed results and insights
- **Dual Architecture**: Choose between client-side or server-side AI processing

## Architecture Options

### Current Version (Client-Side Processing)
- **Frontend**: React + TypeScript with direct Gemini API calls
- **Processing**: Client-side analysis
- **Port**: http://localhost:5173
- **Setup**: Simple, no backend required

### Version 2 (Server-Side Processing)
- **Frontend**: Same React interface
- **Backend**: FastAPI with server-side Gemini processing
- **Processing**: Server-side analysis for better security
- **Port**: Frontend: 5173, Backend: 8000
- **Setup**: Requires both frontend and backend

## Project Structure

```
reviewsense/
├── components/              # React components
├── services/                # API integration services (TypeScript)
├── types.ts                 # TypeScript type definitions
├── *.tsx files              # Main application files
├── package.json             # Node.js dependencies
├── vite.config.ts           # Vite configuration
├── reviewsense_version2/    # Version 2 with Python backend
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Backend documentation
└── README files and configuration
```

## Getting Started

**Prerequisites:** Node.js

1. Install dependencies:
    ```bash
    npm install
    ```
2. Configure your API key in [.env.local](.env.local):
    - Set `GEMINI_API_KEY` to your Google Gemini API key
3. Run the development server:
    ```bash
    npm run dev
    ```
4. Open your browser and navigate to `http://localhost:5173`

## Running ReviewSense Version2 (Server-Side Processing)

**Prerequisites:** Node.js and Python 3.8+

### Backend Setup
1. Navigate to the Version2 directory:
    ```bash
    cd reviewsense_version2
    ```
2. **Quick setup (recommended)**:
    ```bash
    python setup.py
    ```
    This automated script will handle everything for you!

3. **Or manual setup**:
   - Install dependencies: `pip install -r requirements.txt`
   - Configure API key in `.env` file
   - Run: `python main.py`

4. Backend will be available at `http://localhost:8000`
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Frontend Setup (for Version2)
1. Return to the main directory:
    ```bash
    cd ..
    ```
2. Install Node.js dependencies:
    ```bash
    npm install
    ```
3. Configure your API key in [.env.local](.env.local):
    - Set `GEMINI_API_KEY` to your Google Gemini API key
4. Run the development server:
    ```bash
    npm run dev
    ```
5. Frontend will be available at `http://localhost:5173`

**Note:** Version2 uses the same React frontend but calls the Python backend API instead of making direct Gemini calls.

### Switching Between Versions

To switch between the client-side and server-side processing:

1. **For Current Version (Client-side)**:
   - Set `REACT_APP_USE_BACKEND=false` in `.env.local` (or remove the variable)
   - Only the React frontend is needed

2. **For Version2 (Server-side)**:
   - Set `REACT_APP_USE_BACKEND=true` in `.env.local`
   - Set `REACT_APP_API_URL=http://localhost:8000` in `.env.local`
   - Both frontend and backend must be running

## Usage

### Text Input
1. Enter product reviews (one per line) in the text area
2. Click "Analyze Reviews"
3. View detailed sentiment analysis and topic extraction results

### CSV Upload
1. Upload a CSV file containing reviews
2. The system automatically detects review columns
3. Process multiple reviews at once with batch analysis

## API Integration

The application uses the Google Gemini AI API for natural language processing. Make sure your API key has the necessary permissions for text generation and analysis.

## Technologies Used

### Current Version
- **Frontend**: React, TypeScript, Vite
- **AI**: Google Gemini AI
- **Build Tool**: Vite
- **Language**: TypeScript

### Version 2
- **Frontend**: React, TypeScript, Vite
- **Backend**: FastAPI, Python
- **AI**: Google Gemini AI
- **Validation**: Pydantic
- **Server**: Uvicorn
