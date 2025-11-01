
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
