import { AnalysisResponse } from '../types';

// Use relative path so dev server can proxy requests to the backend (avoids CORS in dev)
const API_BASE_URL = '/api';

export const analyzeReviewsWithBackend = async (
  formData: FormData
): Promise<AnalysisResponse> => {
  try {
    console.log('Using backend API for analysis');

    const response = await fetch(`${API_BASE_URL}/analyze_micro_service`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      // Extract error message from backend response (check both 'error' and 'detail' fields)
      const errorMessage = errorData.error || errorData.detail || `HTTP error! status: ${response.status}`;
      throw new Error(errorMessage);
    }

    const results = await response.json();
    return results;
  } catch (error) {
    console.error("Error calling backend API:", error);
    // Preserve the original error message if it's already an Error with a message
    if (error instanceof Error && error.message) {
      throw error;
    }
    // Handle network errors (e.g., backend not running)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error("Failed to connect to the backend. Please check if the backend server is running.");
    }
    throw new Error("Failed to analyze reviews with backend API.");
  }
};

export const getSessionResults = async (sessionId: number): Promise<AnalysisResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/analysis/session/${sessionId}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      // Extract error message from backend response (check both 'error' and 'detail' fields)
      const errorMessage = errorData.error || errorData.detail || `HTTP error! status: ${response.status}`;
      throw new Error(errorMessage);
    }

    const results = await response.json();
    return results;
  } catch (error) {
    console.error("Error fetching session results:", error);
    // Preserve the original error message if it's already an Error with a message
    if (error instanceof Error && error.message) {
      throw error;
    }
    // Handle network errors (e.g., backend not running)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error("Failed to connect to the backend. Please check if the backend server is running.");
    }
    throw new Error("Failed to fetch analysis results.");
  }
};