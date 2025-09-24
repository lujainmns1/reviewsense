
import React, { useState, useCallback } from 'react';
import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ResultsPage from './components/ResultsPage';
import Loader from './components/Loader';
import { analyzeReviews } from './services';
import { Page, AnalysisResult } from './types';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>(Page.Home);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  console.log('App is rendering, current page:', currentPage);

  const handleStart = () => {
    setCurrentPage(Page.Upload);
    setError(null);
    setAnalysisResults([]);
  };

  const handleAnalyze = useCallback(async (reviews: string[]) => {
    if (reviews.length === 0) {
      setError("Please provide at least one review to analyze.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setCurrentPage(Page.Results);

    try {
      const results = await analyzeReviews(reviews);
      setAnalysisResults(results);
    } catch (e) {
      console.error(e);
      setError("An error occurred during analysis. Please check your API configuration and try again.");
      setCurrentPage(Page.Upload); // Go back to upload page on error
    } finally {
      setIsLoading(false);
    }
  }, []);

  const renderPage = () => {
    console.log('Rendering page:', currentPage);
    if (isLoading) {
      return <Loader message="Analyzing reviews... This may take a moment." />;
    }

    switch (currentPage) {
      case Page.Home:
        return <HomePage onStart={handleStart} />;
      case Page.Upload:
        return <UploadPage onAnalyze={handleAnalyze} error={error} />;
      case Page.Results:
        return <ResultsPage results={analysisResults} onAnalyzeAnother={handleStart} />;
      default:
        return <HomePage onStart={handleStart} />;
    }
  };

  return (
    <div className="min-h-screen font-sans flex flex-col items-center justify-center p-4">
      {renderPage()}
    </div>
  );
};

export default App;
