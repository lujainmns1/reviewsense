
export enum Page {
  Home,
  Upload,
  Results,
  Login,
  Signup,
  Dashboard
}

export enum Sentiment {
  Positive = 'positive',
  Negative = 'negative',
  Neutral = 'neutral',
}

export interface Topic {
  topic: string;
  score: number;
}

export interface AnalysisResult {
  reviewText: string;
  sentiment: Sentiment;
  sentimentScore?: number;
  topics: (string | Topic)[];
}

export interface User {
  id: number;
  email: string;
}

export interface AnalysisSession {
  id: number;
  userId: number;
  countryCode?: string;
  detectedDialect?: string;
  createdAt: string;
  reviews: AnalysisResult[];
}

export interface AnalysisResponse {
  results: AnalysisResult[];
  model: string;
  selectedCountry?: string;
  detectedDialect?: string;
  session_id: number;
}
