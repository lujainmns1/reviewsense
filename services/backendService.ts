import { AnalysisResult } from '../types';

// Use relative path so dev server can proxy requests to the backend (avoids CORS in dev)
const API_BASE_URL = '/api';

interface AnalysisResponse {
  results: AnalysisResult[];
  model: string;
  selectedCountry?: string;
  detectedDialect?: string;
  session_id?: number;
}

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
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const results = await response.json();
    return results;
  } catch (error) {
    console.error("Error calling backend API:", error);
    throw new Error("Failed to analyze reviews. Please check that the backend and Docker are running.");
  }
};

export const getSessionResults = async (sessionId: number): Promise<AnalysisResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/analysis/session/${sessionId}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const results = await response.json();
    return results;
  } catch (error) {
    console.error("Error fetching session results:", error);
    throw new Error("Failed to fetch analysis results.");
  }
};