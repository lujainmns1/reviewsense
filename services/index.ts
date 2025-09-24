import { AnalysisResult } from '../types';
import * as geminiService from './geminiService';
import * as backendService from './backendService';

// Determine which service to use based on environment variable
const USE_BACKEND = process.env.REACT_APP_USE_BACKEND === 'true';

export const analyzeReviews = async (reviews: string[]): Promise<AnalysisResult[]> => {
  try {
    if (USE_BACKEND) {
      console.log('Using backend API for analysis');
      return await backendService.analyzeReviewsWithBackend(reviews);
    } else {
      console.log('Using Gemini API for analysis');
      return await geminiService.analyzeReviews(reviews);
    }
  } catch (error) {
    console.error('Error with analysis:', error);
    throw error;
  }
};