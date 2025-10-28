import { AnalysisResult } from '../types';
import * as geminiService from './geminiService';
import * as backendService from './backendService';

// Determine which service to use based on environment variable
const USE_BACKEND = import.meta.env.VITE_USE_BACKEND === 'true';

interface AnalysisResponse {
  results: AnalysisResult[];
  model: string;
  selectedCountry?: string;
  detectedDialect?: string;
}

export const analyzeReviews = async (reviews: string[], model: string, country?: string, autoDetectDialect?: boolean): Promise<AnalysisResponse> => {
  console.log("use backend:", USE_BACKEND)
  try {
    if (USE_BACKEND) {
      // console.log('Using backend API for analysis');
      const res = await backendService.analyzeReviewsWithBackend(reviews, model, country, autoDetectDialect);
      console.log("backend results:", res)
      return {
        results: res.results,
        model: res.model,
        selectedCountry: country,
        detectedDialect: autoDetectDialect ? res.detectedDialect : undefined
      };
    } else {
      console.log('Using Front-End Gemini API for analysis');
      const results = await geminiService.analyzeReviews(reviews);
      return {
        results,
        model,
        selectedCountry: country
      };
    }
  } catch (error) {
    console.error('Error with analysis:', error);
    throw error;
  }
};