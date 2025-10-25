
export enum Page {
  Home,
  Upload,
  Results,
}

export enum Sentiment {
  Positive = 'positive',
  Negative = 'negative',
  Neutral = 'neutral',
}

export interface AnalysisResult {
  reviewText: string;
  sentiment: Sentiment;
  topics: string[];
}
