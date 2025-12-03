
import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult, Sentiment } from '../types';



const API_KEY = process.env.GEMINI_API_KEY;
// const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

if (!API_KEY) {
  throw new Error("GEMINI_API_KEY environment variable is not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

/*const response = await fetch('http://localhost:5000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ reviews: reviews }) // <- هنا نتأكد أن المتغير معروف
});

if (!response.ok) {
  throw new Error(`Server error: ${response.statusText}`);
}

const parsedResults = await response.json();*/



const reviewSchema = {
  type: Type.OBJECT,
  properties: {
    reviewText: {
      type: Type.STRING,
      description: "The original text of the product review."
    },
    sentiment: {
      type: Type.STRING,
      enum: [Sentiment.Positive, Sentiment.Negative, Sentiment.Neutral],
      description: "The overall sentiment of the review."
    },
    topics: {
      type: Type.ARRAY,
      items: {
        type: Type.STRING,
        description: "A specific topic discussed in the review, e.g., quality, shipping, price, customer service."
      },
      description: "A list of key topics mentioned in the review."
    }
  },
  required: ["reviewText", "sentiment", "topics"]
};

export const analyzeReviews = async (reviews: string[]): Promise<AnalysisResult[]> => {
  const reviewData = reviews.map(review => ({ reviewText: review }));
  
  const prompt = `
    Analyze the following product reviews. For each review, determine its sentiment (Positive, Negative, or Neutral) 
    and identify the main topics discussed (e.g., quality, shipping, price, customer service, packaging).
    
    Here are the reviews:
    ${JSON.stringify(reviewData, null, 2)}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: reviewSchema,
        },
      },
    });

    const jsonString = response.text;
    const parsedResults = JSON.parse(jsonString);
    
    // Validate the parsed structure
    if (!Array.isArray(parsedResults)) {
        throw new Error("API response is not an array.");
    }
    
    return parsedResults as AnalysisResult[];
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Failed to analyze reviews with Gemini API.");
  }
};

export const explainReviewLine = async (reviewText: string): Promise<string> => {
  const prompt = `
    Explain the following product review in detail. Provide insights about:
    - What the reviewer is expressing (their main concerns or praises)
    - The sentiment and emotional tone
    - Key aspects or topics mentioned
    - Any implicit meanings or context
    
    Review text: "${reviewText}"
    
    Provide a clear, concise explanation that helps understand the reviewer's perspective.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "Unable to generate explanation.";
  } catch (error) {
    console.error("Error explaining review with Gemini API:", error);
    throw new Error("Failed to generate AI explanation. Please check your internet connection and try again");
  }
};


