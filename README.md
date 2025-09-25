# ReviewSense

A comprehensive AI-powered review analysis application that helps businesses and individuals analyze customer feedback using advanced natural language processing. The application provides sentiment analysis, topic extraction, and detailed insights from product reviews.

## 🚀 Features

- **Multi-Modal Analysis**: Support for both direct AI integration and backend API processing
- **Sentiment Analysis**: Automatic classification of reviews as Positive, Negative, or Neutral
- **Topic Extraction**: Intelligent identification of key topics and themes in reviews
- **Modern UI**: Clean, responsive React-based interface with Tailwind CSS styling
- **Multiple Backend Options**: Choose between FastAPI and Flask backends
- **Real-time Processing**: Fast analysis with loading indicators and error handling
- **Flexible Architecture**: Environment-based service selection for different deployment scenarios

## 🏗️ Architecture

ReviewSense consists of two main components:

### Frontend (React + TypeScript)
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development and optimized production builds
- **UI Components**: Custom React components with modern styling
- **State Management**: React hooks for component state
- **Service Layer**: Modular service architecture supporting multiple AI providers

### Backend (FastAPI)
- **Framework**: FastAPI with async support
- **AI Integration**: Google Gemini API for server-side processing
- **Features**: Auto-generated API documentation, CORS support, health checks
- **Deployment**: Production-ready with environment validation

## 📋 Prerequisites

- **Node.js** (v16 or higher) for the frontend
- **Python** (v3.8 or higher) for the backend
- **Google Gemini API Key** for AI analysis functionality

## 🛠️ Installation & Setup

### Frontend Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Access the application**:
   - Frontend: http://localhost:5173
   - The application will be available in your browser

### Backend Setup Options

### Backend Setup (FastAPI)

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Run the automated setup**:
   ```bash
   python setup.py
   ```
   This will install dependencies, validate configuration, and start the server.

3. **Or setup manually**:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

4. **Access the API**:
   - API Base: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## ⚙️ Configuration

### Environment Variables

Create a `.env.local` file in the root directory for frontend configuration:

```env
REACT_APP_USE_BACKEND=true  # Set to true to use backend API, false for direct Gemini calls
```

For backend services, configure the following in their respective `.env` files:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### Service Selection

The frontend automatically selects the analysis service based on the `REACT_APP_USE_BACKEND` environment variable:

- **Backend Mode** (`REACT_APP_USE_BACKEND=true`): Uses the FastAPI/Flask backend for analysis
- **Direct Mode** (`REACT_APP_USE_BACKEND=false`): Makes direct calls to Google Gemini API

## 🎯 Usage

1. **Start the Application**:
   - Launch the frontend: `npm run dev`
   - Start your preferred backend service

2. **Access the Interface**:
   - Open http://localhost:5173 in your browser

3. **Analyze Reviews**:
   - Click "Get Started" on the home page
   - Upload or paste product reviews
   - Click "Analyze Reviews" to process
   - View detailed sentiment analysis and topic extraction results

4. **Review Results**:
   - Each review is analyzed for sentiment (Positive/Negative/Neutral)
   - Key topics and themes are automatically extracted
   - Results are displayed in an easy-to-read format

## 🔧 Development

### Project Structure

```
reviewsense/
├── components/           # React components
├── services/            # API service layers
├── backend/             # FastAPI backend
├── types.ts             # TypeScript type definitions
└── package.json         # Frontend dependencies
```

### Key Components

- **HomePage**: Landing page with application introduction
- **UploadPage**: Review input interface with validation
- **ResultsPage**: Analysis results display with formatting
- **Loader**: Loading states and progress indicators
- **Services**: Modular API integration (Gemini + Backend options)

### Adding New Features

1. **Frontend Changes**:
   - Add new components in the `components/` directory
   - Update types in `types.ts` for new data structures
   - Modify services in `services/` for new API endpoints

2. **Backend Changes**:
   - Add new endpoints in `backend/main.py`
   - Update requirements files for new dependencies

## 🚀 Deployment

### Frontend Deployment

1. **Build for production**:
   ```bash
   npm run build
   ```

2. **Deploy the dist folder** to your web server or CDN

### Backend Deployment

```bash
# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🧪 Testing

### Frontend Testing
```bash
# Add tests in future with a testing framework like Jest or Vitest
npm test
```

### Backend Testing
```bash
# Use pytest for API endpoint testing
pytest
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 API Reference

### Analysis Endpoint (FastAPI)
```http
POST /analyze
Content-Type: application/json

{
  "reviews": ["Great product!", "Could be better"]
}
```

**Response**:
```json
[
  {
    "reviewText": "Great product!",
    "sentiment": "Positive",
    "topics": ["quality", "value"]
  }
]
```

### Health Check Endpoint
```http
GET /health
```

## 🔐 Security

- API keys are stored in environment variables
- CORS is configured for frontend-backend communication
- Input validation and sanitization on all endpoints
- Error messages don't expose sensitive system information

## 📊 Performance

- **Frontend**: Optimized React components with efficient re-rendering
- **Backend**: Async processing for better concurrency
- **AI Processing**: Batch processing for multiple reviews
- **Caching**: Consider implementing Redis for result caching in production

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify backend services are running
   - Check CORS configuration
   - Validate API endpoints are accessible

2. **Gemini API Issues**:
   - Ensure valid API key is configured
   - Check API quota and rate limits
   - Verify network connectivity

3. **Build Errors**:
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Update dependencies: `npm update`
   - Check Node.js version compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for powerful natural language processing
- FastAPI and Flask communities for excellent web frameworks
- React and TypeScript communities for modern frontend development
- Vite for fast build tooling

## 📞 Support

For support, please create an issue in the GitHub repository or contact the development team.

---

**ReviewSense** - Transform customer feedback into actionable insights with AI-powered analysis.