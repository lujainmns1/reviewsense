import { AnalysisResult } from '../types';
import * as geminiService from './geminiService';
import * as backendService from './backendService';

// Determine which service to use based on environment variable
const USE_BACKEND = import.meta.env.VITE_USE_BACKEND === 'true';

export const analyzeReviews = async (reviews: string[], model: string): Promise<AnalysisResult[]> => {
  console.log("use backend:", USE_BACKEND)
  try {
    if (USE_BACKEND) {
      // console.log('Using backend API for analysis');
      return await backendService.analyzeReviewsWithBackend(reviews, model);
    } else {
      console.log('Using Front-End Gemini API for analysis');
      return await geminiService.analyzeReviews(reviews);
    }
  } catch (error) {
    console.error('Error with analysis:', error);
    throw error;
  }
};