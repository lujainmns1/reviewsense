
export enum Page {
  Home,
  Upload,
  Results,
}

export enum Sentiment {
  Positive = 'Positive',
  Negative = 'Negative',
  Neutral = 'Neutral',
}

export interface AnalysisResult {
  reviewText: string;
  sentiment: Sentiment;
  topics: string[];
}
