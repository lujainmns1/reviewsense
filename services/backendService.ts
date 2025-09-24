import { AnalysisResult } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const analyzeReviewsWithBackend = async (reviews: string[]): Promise<AnalysisResult[]> => {
  try {
    console.log('Using backend API for analysis');

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reviews }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const results = await response.json();

    // Validate the response structure
    if (!Array.isArray(results)) {
      throw new Error("Backend response is not an array.");
    }

    return results as AnalysisResult[];
  } catch (error) {
    console.error("Error calling backend API:", error);
    throw new Error("Failed to analyze reviews with backend API.");
  }
};