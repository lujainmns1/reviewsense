
import React, { useState, useCallback, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ResultsPage from './components/ResultsPage';
import TopicClassificationPage from './components/TopicClassificationPage';
import LoginPage from './components/auth/LoginPage';
import SignupPage from './components/auth/SignupPage';
import DashboardPage from './components/DashboardPage';
import DashboardLayout from './components/DashboardLayout';
import Loader from './components/Loader';
import { analyzeReviews } from './services';
import { AnalysisResult } from './types';

const PrivateRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = localStorage.getItem('user') !== null;
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

const AppContent: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Reset loading and error when navigating away from upload page
  useEffect(() => {
    if (location.pathname !== '/upload') {
      setIsLoading(false);
      setError(null);
    }
  }, [location.pathname]);

  const handleAnalyze = useCallback(async (formData: FormData) => {
    if (!formData.get('text')) {
      setError("Please provide at least one review to analyze.");
      setIsLoading(false);
      return;
    }
    
    // Set loading state first
    setIsLoading(true);
    setError(null);

    try {
      console.log('Starting analysis...');
      const results = await analyzeReviews(formData);
      console.log('Results from analyzeReviews:', results);
      
      // Navigate to results page with session_id
      if (results.session_id) {
        console.log(`Navigating to results page with session_id: ${results.session_id}`);
        // Navigate immediately - loading will be cleared by useEffect when route changes
        navigate(`/results/${results.session_id}`);
      } else {
        console.error('No session_id in results:', results);
        setError("Analysis completed but no session ID was returned. Please try again.");
        setIsLoading(false);
      }
    } catch (e) {
      console.error('Analysis error:', e);
      const errorMessage = e instanceof Error ? e.message : "An error occurred during analysis. Please check your API configuration and try again.";
      setError(errorMessage);
      setIsLoading(false);
    }
    // Note: We don't clear loading in finally because we want to keep it during navigation
    // It will be cleared on error or when the component unmounts
  }, [navigate]);

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/" element={<HomePage />} />
      <Route
        path="/dashboard"
        element={
          <PrivateRoute>
            <DashboardLayout>
              <DashboardPage />
            </DashboardLayout>
          </PrivateRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <PrivateRoute>
            <DashboardLayout>
              <UploadPage onAnalyze={handleAnalyze} error={error} isLoading={isLoading} />
            </DashboardLayout>
          </PrivateRoute>
        }
      />
      <Route
        path="/results/:sessionId"
        element={
          <PrivateRoute>
            <DashboardLayout>
              <ResultsPage onAnalyzeAnother={() => navigate('/upload')} />
            </DashboardLayout>
          </PrivateRoute>
        }
      />
      <Route
        path="/topics"
        element={
          <PrivateRoute>
            <DashboardLayout>
              <TopicClassificationPage />
            </DashboardLayout>
          </PrivateRoute>
        }
      />
      <Route
        path="/topics/:sessionId"
        element={
          <PrivateRoute>
            <DashboardLayout>
              <TopicClassificationPage />
            </DashboardLayout>
          </PrivateRoute>
        }
      />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;
