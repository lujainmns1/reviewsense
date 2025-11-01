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
  session_id?: number;
}

export const analyzeReviews = async (
  formData: FormData
): Promise<AnalysisResponse> => {
  console.log("use backend:", USE_BACKEND)
  try {
    if (USE_BACKEND) {
      const res = await backendService.analyzeReviewsWithBackend(formData);
      console.log("backend results:", res)
      return {
        results: res.results,
        model: res.model,
        selectedCountry: formData.get('country')?.toString() || undefined,
        detectedDialect: formData.get('auto_detect') === 'true' ? res.detectedDialect : undefined
      };
    } else {
      console.log('Using Front-End Gemini API for analysis');
      const reviewText = formData.get('text')?.toString() || '';
      const reviews = reviewText.split('\n').filter(r => r.trim());
      const results = await geminiService.analyzeReviews(reviews);
      return {
        results,
        model: formData.get('model')?.toString() || '',
        selectedCountry: formData.get('country')?.toString() || undefined
      };
    }
  } catch (error) {
    console.error('Error with analysis:', error);
    throw error;
  }
};