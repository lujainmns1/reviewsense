
import React, { useState, useEffect } from 'react';
import { AnalysisResponse, AnalysisResult, Sentiment } from '../types';
import StarIcon from './icons/StarIcon';
import Flag from 'react-world-flags';
import { useParams, useNavigate } from 'react-router-dom';
import { getSessionResults } from '../services/backendService';
import { explainReviewLine } from '../services/geminiService';

// Utility function to convert markdown to HTML
const markdownToHtml = (text: string): string => {
  if (!text) return '';
  
  let html = text;
  
  // First, escape any existing HTML to prevent XSS
  const escapeHtml = (str: string) => {
    const map: { [key: string]: string } = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };
    return str.replace(/[&<>"']/g, (m) => map[m]);
  };
  
  // Escape HTML first
  html = escapeHtml(html);
  
  // Convert **bold** to <strong>bold</strong> (handle multiple on same line)
  html = html.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
  
  // Convert *italic* to <em>italic</em> (only single asterisks, not part of **)
  // Process after bold to avoid conflicts
  html = html.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em>$1</em>');
  
  // Convert `code` to <code>code</code>
  html = html.replace(/`([^`\n]+)`/g, '<code class="bg-white/10 px-1 py-0.5 rounded text-xs font-mono text-blue-300">$1</code>');
  
  // Split into lines to handle lists properly
  const lines = html.split('\n');
  const processedLines: string[] = [];
  let inList = false;
  let listType: 'ul' | 'ol' | null = null;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    
    // Check for numbered list
    const numberedMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
    if (numberedMatch) {
      if (!inList || listType !== 'ol') {
        if (inList && listType === 'ul') {
          processedLines.push('</ul>');
        }
        processedLines.push('<ol class="list-decimal list-inside space-y-1 my-2 ml-4">');
        inList = true;
        listType = 'ol';
      }
      processedLines.push(`<li>${numberedMatch[2]}</li>`);
      continue;
    }
    
    // Check for bullet list (- or *)
    const bulletMatch = trimmedLine.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      if (!inList || listType !== 'ul') {
        if (inList && listType === 'ol') {
          processedLines.push('</ol>');
        }
        processedLines.push('<ul class="list-disc list-inside space-y-1 my-2 ml-4">');
        inList = true;
        listType = 'ul';
      }
      processedLines.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }
    
    // Not a list item - close any open list
    if (inList) {
      processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
      inList = false;
      listType = null;
    }
    
    // Regular line - convert line breaks
    if (trimmedLine) {
      processedLines.push(trimmedLine);
    } else {
      processedLines.push('<br>');
    }
  }
  
  // Close any remaining list
  if (inList) {
    processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
  }
  
  return processedLines.join('\n');
};

interface ResultsPageProps {
  onAnalyzeAnother: () => void;
}


const sentimentColors: { [key in Sentiment]: { bg: string; text: string; border: string } } = {
  [Sentiment.Positive]: { bg: 'bg-emerald-500/15', text: 'text-emerald-200', border: 'border-emerald-400/60' },
  [Sentiment.Negative]: { bg: 'bg-rose-500/15', text: 'text-rose-200', border: 'border-rose-400/60' },
  [Sentiment.Neutral]: { bg: 'bg-amber-500/15', text: 'text-amber-200', border: 'border-amber-400/60' },
};

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'arabert-arsas-sa': 'AraBERTv2 · ArSAS',
  'marbertv2-book-review-sa': 'MARBERTv2 · Book Review',
  'xlm-roberta-twitter-sa': 'XLM‑RoBERTa · Twitter',
  'election-mode': 'Election Mode (best score)',
};

const formatModelName = (modelKey?: string) => {
  if (!modelKey) return 'Unknown model';
  return MODEL_DISPLAY_NAMES[modelKey] || modelKey;
};

const SummaryCard: React.FC<{ title: string; count: number; colorClass: string }> = ({ title, count, colorClass }) => (
  <div className={`p-5 rounded-2xl text-center border ${colorClass}`}>
    <p className="text-3xl font-black">{count}</p>
    <p className="text-sm font-medium text-slate-200 mt-1">{title}</p>
  </div>
);

const ResultsPage: React.FC<ResultsPageProps> = ({ onAnalyzeAnother }) => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [model, setModel] = useState<string>('');
  const [selectedCountry, setSelectedCountry] = useState<string | undefined>();
  const [detectedDialect, setDetectedDialect] = useState<string | undefined>();
  const [mode, setMode] = useState<'single' | 'election'>('single');
  const [modelsConsidered, setModelsConsidered] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sentimentCounts, setSentimentCounts] = useState<{ [key in Sentiment]?: number }>({});
  const [selectedReviewIndex, setSelectedReviewIndex] = useState<number | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (!sessionId) return;
      
      try {
        const data = await getSessionResults(parseInt(sessionId, 10)) as AnalysisResponse & {
          mode?: 'single' | 'election';
          modelsConsidered?: string[];
        };
        setResults(data.results);
        setModel(data.model);
        setSelectedCountry(data.selectedCountry);
        setDetectedDialect(data.detectedDialect);
        setMode(data.mode || 'single');
        setModelsConsidered(data.modelsConsidered || (data.model ? [data.model] : []));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch results');
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [sessionId]);

  useEffect(() => {
    if (results.length > 0) {
      const counts = results.reduce((acc, result) => {
        acc[result.sentiment] = (acc[result.sentiment] || 0) + 1;
        return acc;
      }, {} as { [key in Sentiment]?: number });
      setSentimentCounts(counts);
    }
  }, [results]);

  const handleExplainReview = async (reviewText: string, index: number) => {
    // If clicking the same review, close the explanation
    if (selectedReviewIndex === index && explanation) {
      setSelectedReviewIndex(null);
      setExplanation(null);
      return;
    }

    setSelectedReviewIndex(index);
    setExplanation(null);
    setExplanationError(null);
    setExplaining(true);

    try {
      const explanationText = await explainReviewLine(reviewText);
      setExplanation(explanationText);
    } catch (err) {
      setExplanationError(err instanceof Error ? err.message : 'Failed to explain review');
    } finally {
      setExplaining(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center p-8 bg-slate-900/60 border border-white/10 rounded-3xl shadow-2xl shadow-blue-500/10 text-white">
        <h2 className="text-2xl font-bold">Loading results...</h2>
      </div>
    );
  }

  if (error || !results.length) {
    return (
      <div className="text-center p-8 bg-slate-900/60 border border-white/10 rounded-3xl shadow-2xl shadow-blue-500/10 text-white">
        <h2 className="text-2xl font-bold">{error || 'No results to display.'}</h2>
        <button 
          onClick={() => navigate('/upload')} 
          className="mt-4 bg-blue-600 text-white font-bold py-3 px-8 rounded-full hover:bg-blue-500 transition"
        >
          Try Another Analysis
        </button>
      </div>
    );
  }

  const positivePercentage = (sentimentCounts[Sentiment.Positive] || 0) / results.length;
  const starRating = Math.max(1, Math.ceil(positivePercentage * 5));
  const isElectionMode = mode === 'election';
  const consideredModels = (modelsConsidered && modelsConsidered.length > 0
    ? modelsConsidered
    : (model ? [model] : []));

  return (
    <div className="w-full max-w-6xl mx-auto bg-slate-900/60 border border-white/10 rounded-3xl shadow-2xl shadow-blue-500/10 p-8 text-white">
      <div className="text-center mb-6 space-y-2">
        <h2 className="text-3xl font-black">Analysis Results</h2>
        <div className="flex flex-wrap items-center justify-center gap-3 text-sm text-slate-300">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/20 px-3 py-1 bg-white/5 text-white font-semibold">
            Mode: {isElectionMode ? 'Election (best score)' : 'Single model'}
          </span>
          <span className="inline-flex items-center gap-2 rounded-full border border-blue-400/40 px-3 py-1 text-blue-100">
            Primary model: {formatModelName(model)}
          </span>
        </div>
        {consideredModels.length > 0 && (
          <div className="flex flex-wrap justify-center gap-2 text-xs text-slate-300">
            {consideredModels.map((m) => (
              <span key={m} className="px-2 py-1 rounded-lg border border-white/15 bg-white/5">
                {formatModelName(m)}
              </span>
            ))}
          </div>
        )}
        {isElectionMode && (
          <p className="text-xs text-slate-400">
            Each review is scored by all available models, and the sentiment with the highest confidence is kept.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="md:col-span-1 p-4 bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center">
          {selectedCountry && (
            <div className="flex flex-col items-center gap-2 mb-3">
              <Flag code={selectedCountry} className="w-8 h-6 object-cover rounded shadow-lg" />
              {detectedDialect && (
                <div className="text-sm text-slate-300 mt-1">
                  Detected Dialect: {detectedDialect}
                </div>
              )}
            </div>
          )}
          <h3 className="text-lg font-semibold text-slate-200 mb-2">Overall Rating</h3>
          <div className="flex">
            {[...Array(5)].map((_, i) => (
              <StarIcon key={i} className={`h-8 w-8 ${i < starRating ? 'text-blue-400' : 'text-slate-700'}`} />
            ))}
          </div>
        </div>
        <SummaryCard 
          title="Positive Reviews" 
          count={sentimentCounts[Sentiment.Positive] || 0} 
          colorClass="border-emerald-400/40 bg-emerald-500/10 text-emerald-200" 
        />
        <SummaryCard 
          title="Negative Reviews" 
          count={sentimentCounts[Sentiment.Negative] || 0} 
          colorClass="border-rose-400/40 bg-rose-500/10 text-rose-200" 
        />
        <SummaryCard 
          title="Neutral Reviews" 
          count={sentimentCounts[Sentiment.Neutral] || 0} 
          colorClass="border-amber-400/40 bg-amber-500/10 text-amber-200" 
        />
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-white/10 text-left">
          <thead className="bg-white/5 text-slate-300">
            <tr>
              <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                Review Text
              </th>
              <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                Sentiment
              </th>
              <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                Score
              </th>
              <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                Topics
              </th>
              {isElectionMode && (
                <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                  Model Used
                </th>
              )}
              <th className="px-6 py-3 text-xs font-semibold uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-transparent divide-y divide-white/10">
            {results.map((result, index) => (
              <React.Fragment key={index}>
                <tr className={selectedReviewIndex === index ? 'bg-blue-500/10' : ''}>
                  <td className="px-6 py-4 whitespace-normal text-sm text-slate-200">
                    {result.reviewText}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col gap-1">
                      <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full border ${sentimentColors[result.sentiment].bg} ${sentimentColors[result.sentiment].text} ${sentimentColors[result.sentiment].border}`}>
                        {result.sentiment}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-200">
                    {result.sentimentScore !== undefined ? result.sentimentScore.toFixed(2) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-200">
                    <div className="flex flex-wrap gap-2">
                      {result.topics.length > 0 ? result.topics.map((topic, i) => (
                        <span key={i} className="px-2 py-1 bg-white/10 text-white rounded-md text-xs border border-white/20">
                          {typeof topic === 'string' ? topic : topic.topic}
                        </span>
                      )) : <span>-</span>}
                    </div>
                  </td>
                  {isElectionMode && (
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-200">
                      {formatModelName(result.modelUsed)}
                    </td>
                  )}
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <button
                      onClick={() => handleExplainReview(result.reviewText, index)}
                      disabled={explaining && selectedReviewIndex === index}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                        selectedReviewIndex === index
                          ? 'bg-blue-600 text-white hover:bg-blue-500'
                          : 'bg-white/10 text-slate-200 hover:bg-white/20 border border-white/20'
                      } ${explaining && selectedReviewIndex === index ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      {explaining && selectedReviewIndex === index ? 'Explaining...' : selectedReviewIndex === index ? 'Hide Explanation' : 'Explain with AI'}
                    </button>
                  </td>
                </tr>
                {selectedReviewIndex === index && (
                  <tr>
                    <td colSpan={isElectionMode ? 6 : 5} className="px-6 py-4 bg-slate-800/50 border-t border-white/10">
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm font-semibold text-blue-300">AI Explanation:</span>
                        </div>
                        {explaining ? (
                          <div className="flex items-center gap-2 text-slate-300">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                            <span className="text-sm">Generating explanation...</span>
                          </div>
                        ) : explanationError ? (
                          <div className="text-sm text-rose-300 bg-rose-500/10 border border-rose-400/30 rounded-lg p-3">
                            {explanationError}
                          </div>
                        ) : explanation ? (
                          <div 
                            className="text-sm text-slate-200 bg-white/5 border border-white/10 rounded-lg p-4 leading-relaxed prose prose-invert prose-sm max-w-none"
                            dangerouslySetInnerHTML={{ __html: markdownToHtml(explanation) }}
                          />
                        ) : null}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-8 text-center">
        <button
          onClick={() => navigate('/upload')}
          className="bg-blue-600 text-white font-bold py-3 px-8 rounded-full hover:bg-blue-500 transition-all duration-300 transform hover:scale-105 shadow-lg shadow-blue-500/40"
        >
          Do Another Analysis
        </button>
      </div>
    </div>
  );
};

export default ResultsPage;
