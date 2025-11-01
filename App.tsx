
import React, { useState, useCallback, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ResultsPage from './components/ResultsPage';
import LoginPage from './components/auth/LoginPage';
import SignupPage from './components/auth/SignupPage';
import DashboardPage from './components/DashboardPage';
import Loader from './components/Loader';
import { analyzeReviews } from './services';
import { Page, AnalysisResult } from './types';

const PrivateRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = localStorage.getItem('user') !== null;
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<{
    results: AnalysisResult[];
    model: string;
    selectedCountry?: string;
    detectedDialect?: string;
  }>({ results: [], model: '', selectedCountry: undefined, detectedDialect: undefined });

  const handleAnalyze = useCallback(async (formData: FormData) => {
    if (!formData.get('text')) {
      setError("Please provide at least one review to analyze.");
      return;
    }
    
    setIsLoading(true);
    setError(null);

    try {
      const results = await analyzeReviews(formData);
      console.log('Results from analyzeReviews:', results);
      setAnalysisResults({
        results: results.results,
        model: results.model,
        selectedCountry: results.selectedCountry || country,
        detectedDialect: results.detectedDialect
      });
      return results.session_id;
    } catch (e) {
      console.error(e);
      setError("An error occurred during analysis. Please check your API configuration and try again.");
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route
          path="/"
          element={
            <PrivateRoute>
              <HomePage />
            </PrivateRoute>
          }
        />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <DashboardPage />
            </PrivateRoute>
          }
        />
        <Route
          path="/upload"
          element={
            <PrivateRoute>
              <UploadPage onAnalyze={handleAnalyze} error={error} />
            </PrivateRoute>
          }
        />
        <Route
          path="/results/:sessionId"
          element={
            <PrivateRoute>
              <ResultsPage
                results={analysisResults.results}
                model={analysisResults.model}
                selectedCountry={analysisResults.selectedCountry}
                detectedDialect={analysisResults.detectedDialect}
                onAnalyzeAnother={() => null}
              />
            </PrivateRoute>
          }
        />
      </Routes>
    </Router>
  );
};

export default App;
