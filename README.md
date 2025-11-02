# ReviewSense

A comprehensive AI-powered review analysis application that helps businesses and individuals analyze customer feedback using advanced natural language processing. The application provides sentiment analysis, topic extraction, and detailed insights from product reviews.

## 🚀 Features

- **Multi-Modal Analysis**: Support for both direct AI integration and backend API processing
- **Sentiment Analysis**: Automatic classification of reviews as Positive, Negative, or Neutral
- **Topic Extraction**: Intelligent identification of key topics and themes in reviews
- **Arabic Language Support**: Specialized analysis for Arabic reviews with dialect detection
- **User Authentication**: Secure user registration and login system
- **Session Management**: Track and retrieve analysis history
- **Modern UI**: Clean, responsive React-based interface with dashboard layout
- **Multiple Backend Services**: Flask backend with microservice architecture
- **Real-time Processing**: Fast analysis with loading indicators and error handling
- **Flexible Architecture**: Environment-based service selection for different deployment scenarios

## 🏗️ Architecture

ReviewSense consists of multiple components:

### Frontend (React + TypeScript)
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development and optimized production builds
- **Routing**: React Router for navigation
- **UI Components**: Custom React components with dashboard layout
- **State Management**: React hooks for component state
- **Service Layer**: Modular service architecture supporting direct Gemini API and backend integration

### Backend (Flask)
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: PostgreSQL with Flask-Migrate for schema management
- **AI Integration**: Google Gemini API and Transformers for server-side processing
- **Features**: User authentication, session management, Arabic text analysis, CORS support
- **Port**: Runs on port 5000 by default

### Additional Services
- **sentiment-analysis-service**: Standalone Flask service for advanced sentiment analysis
- **sentiment-analyzer**: ML model training and inference toolkit for custom models

## 📋 Prerequisites

- **Node.js** (v16 or higher) for the frontend
- **Python** (v3.8 or higher) for the backend
- **PostgreSQL** database server
- **Google Gemini API Key** for AI analysis functionality
- **PyTorch** and **Transformers** for ML-based analysis (installed via requirements.txt)

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

### Backend Setup (Flask)

1. **Navigate to the Flask backend directory**:
   ```bash
   cd flask_backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   - Ensure PostgreSQL is running
   - Create a database named `reviewsense` (or configure your own)
   - Set the `DATABASE_URL` environment variable in your `.env` file

5. **Configure environment variables**:
   Create a `.env` file in the `flask_backend` directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/reviewsense
   SECRET_KEY=your-secret-key-here
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5000
   ```

6. **Initialize database tables**:
   The application will automatically create tables on first run, or you can use Flask-Migrate:
   ```bash
   flask db upgrade
   ```

7. **Run the Flask server**:
   ```bash
   python app.py
   ```

8. **Access the API**:
   - API Base: http://localhost:5000
   - Health Check: http://localhost:5000/health
   - Root endpoint: http://localhost:5000/

## ⚙️ Configuration

### Environment Variables

#### Frontend Configuration

Create a `.env` file in the root directory for frontend configuration:

```env
VITE_USE_BACKEND=true  # Set to true to use backend API, false for direct Gemini calls
GEMINI_API_KEY=your_google_gemini_api_key_here  # Required for direct mode
```

#### Backend Configuration

Create a `.env` file in the `flask_backend` directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/reviewsense
SECRET_KEY=your-secret-key-here
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

### Service Selection

The frontend automatically selects the analysis service based on the `VITE_USE_BACKEND` environment variable:

- **Backend Mode** (`VITE_USE_BACKEND=true`): Uses the Flask backend API for analysis (recommended for production)
- **Direct Mode** (`VITE_USE_BACKEND=false`): Makes direct calls to Google Gemini API from the browser

## 🎯 Usage

1. **Start the Application**:
   - Launch the frontend: `npm run dev` (in the root directory)
   - Start the Flask backend: `python flask_backend/app.py`

2. **Access the Interface**:
   - Open http://localhost:5173 in your browser
   - You'll be redirected to the login page if not authenticated

3. **User Authentication**:
   - Create an account using the Signup page
   - Or login with existing credentials
   - Authentication is required to access analysis features

4. **Analyze Reviews**:
   - Navigate to the Upload page from the dashboard
   - Paste or upload product reviews (supports both English and Arabic)
   - Select analysis options (model, country/dialect for Arabic)
   - Click "Analyze Reviews" to process
   - View detailed sentiment analysis and topic extraction results

5. **Review Results**:
   - Each review is analyzed for sentiment (Positive/Negative/Neutral)
   - Key topics and themes are automatically extracted
   - Arabic reviews include dialect detection
   - Results are displayed in an easy-to-read format
   - Analysis sessions are saved and can be retrieved from history

## 🔧 Development

### Project Structure

```
reviewsense-main/
├── components/                    # React components
│   ├── auth/                     # Authentication components (Login, Signup)
│   ├── icons/                    # Icon components
│   ├── DashboardLayout.tsx       # Main dashboard layout wrapper
│   ├── DashboardPage.tsx         # Dashboard home page
│   ├── HomePage.tsx              # Landing page
│   ├── UploadPage.tsx            # Review input interface
│   ├── ResultsPage.tsx           # Analysis results display
│   └── Loader.tsx                # Loading indicators
├── services/                     # API service layers
│   ├── authService.ts            # User authentication API
│   ├── backendService.ts         # Flask backend integration
│   ├── geminiService.ts          # Direct Gemini API integration
│   └── index.ts                  # Service selector
├── flask_backend/                # Flask backend application
│   ├── app.py                    # Main Flask application
│   ├── models.py                 # SQLAlchemy database models
│   ├── arabic_model_service.py   # Arabic text analysis service
│   ├── requirements.txt          # Backend dependencies
│   └── migrations/               # Database migration files
├── sentiment-analysis-service/   # Standalone sentiment analysis service
│   ├── app.py                    # Flask microservice
│   ├── ar_pipeline.py            # Arabic processing pipeline
│   └── requirements.txt          # Service dependencies
├── sentiment-analyzer/           # ML model training and inference
│   ├── main.py                   # Main entry point
│   ├── src/                      # Source code modules
│   └── config/                   # Configuration files
├── types.ts                      # TypeScript type definitions
├── App.tsx                       # Main React application component
├── index.tsx                     # React entry point
├── vite.config.ts                # Vite configuration
├── package.json                  # Frontend dependencies
└── requirements.txt              # Root Python dependencies
```

### Key Components

- **HomePage**: Dashboard landing page with application introduction
- **DashboardPage**: User dashboard with analysis history
- **UploadPage**: Review input interface with validation and language selection
- **ResultsPage**: Analysis results display with formatting
- **Loader**: Loading states and progress indicators
- **LoginPage/SignupPage**: User authentication components
- **Services**: Modular API integration (Gemini + Backend options)

### Adding New Features

1. **Frontend Changes**:
   - Add new components in the `components/` directory
   - Update types in `types.ts` for new data structures
   - Modify services in `services/` for new API endpoints
   - Update routing in `App.tsx` if needed

2. **Backend Changes**:
   - Add new endpoints in `flask_backend/app.py`
   - Update database models in `flask_backend/models.py`
   - Create and run migrations: `flask db migrate` and `flask db upgrade`
   - Update `flask_backend/requirements.txt` for new dependencies

## 🚀 Deployment

### Frontend Deployment

1. **Build for production**:
   ```bash
   npm run build
   ```

2. **Deploy the dist folder** to your web server or CDN

### Backend Deployment

For Flask backend production deployment:

```bash
# Using Gunicorn (recommended for production)
pip install gunicorn
cd flask_backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or using Waitress (Windows-compatible)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

For development, the built-in Flask server is sufficient:
```bash
python flask_backend/app.py
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

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Analysis Endpoints

#### Analyze Reviews
```http
POST /analyze_micro_service
Content-Type: multipart/form-data

Form Data:
- text: "Review text here..."
- model: "gemini" or "transformer"
- country: "SA" (optional, for Arabic)
- auto_detect: "true" or "false" (optional)
```

**Response**:
```json
{
  "results": [
    {
      "reviewText": "Great product!",
      "sentiment": "positive",
      "topics": ["quality", "value"]
    }
  ],
  "model": "gemini-2.5-flash",
  "session_id": 123,
  "detectedDialect": "SA" (if Arabic detected)
}
```

#### Get Analysis Session
```http
GET /analysis/session/<session_id>
```

#### Get Analysis History
```http
GET /analysis/history/<user_id>?page=1
```

### Utility Endpoints

#### Health Check
```http
GET /health
```

#### API Info
```http
GET /
```

## 🔐 Security

- API keys are stored in environment variables (never commit to version control)
- User passwords are hashed using Argon2
- CORS is configured for frontend-backend communication
- SQLAlchemy ORM prevents SQL injection attacks
- Input validation and sanitization on all endpoints
- Error messages don't expose sensitive system information
- Session-based authentication with secure token storage

## 📊 Performance

- **Frontend**: Optimized React components with efficient re-rendering
- **Backend**: Flask with SQLAlchemy for efficient database operations
- **AI Processing**: Supports both Google Gemini API and local Transformers models
- **Arabic Analysis**: Specialized models for Arabic text with dialect detection
- **Database**: PostgreSQL for robust data storage and session management
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
- Flask community for excellent web framework
- React and TypeScript communities for modern frontend development
- Vite for fast build tooling
- Transformers library by Hugging Face for ML-based analysis
- CAMEL Tools for Arabic language processing

## 📞 Support

For support, please create an issue in the GitHub repository or contact the development team.

---

**ReviewSense** - Transform customer feedback into actionable insights with AI-powered analysis.