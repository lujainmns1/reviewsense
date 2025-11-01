
import React, { useState, useEffect } from 'react';
import { AnalysisResult, Sentiment } from '../types';
import StarIcon from './icons/StarIcon';
import Flag from 'react-world-flags';
import { useParams, useNavigate } from 'react-router-dom';
import { getSessionResults } from '../services/backendService';

interface ResultsPageProps {
  onAnalyzeAnother: () => void;
}


const sentimentColors: { [key in Sentiment]: { bg: string; text: string; border: string } } = {
  [Sentiment.Positive]: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-400' },
  [Sentiment.Negative]: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-400' },
  [Sentiment.Neutral]: { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-400' },
};

const SummaryCard: React.FC<{ title: string; count: number; colorClass: string }> = ({ title, count, colorClass }) => (
  <div className={`p-4 rounded-lg text-center ${colorClass}`}>
    <p className="text-3xl font-bold">{count}</p>
    <p className="text-sm font-medium">{title}</p>
  </div>
);

const ResultsPage: React.FC<ResultsPageProps> = ({ onAnalyzeAnother }) => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [model, setModel] = useState<string>('');
  const [selectedCountry, setSelectedCountry] = useState<string | undefined>();
  const [detectedDialect, setDetectedDialect] = useState<string | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sentimentCounts, setSentimentCounts] = useState<{ [key in Sentiment]?: number }>({});

  useEffect(() => {
    const fetchResults = async () => {
      if (!sessionId) return;
      
      try {
        const data = await getSessionResults(parseInt(sessionId, 10));
        setResults(data.results);
        setModel(data.model);
        setSelectedCountry(data.selectedCountry);
        setDetectedDialect(data.detectedDialect);
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

  if (loading) {
    return (
      <div className="text-center p-8 bg-white rounded-2xl shadow-xl">
        <h2 className="text-2xl font-bold">Loading results...</h2>
      </div>
    );
  }

  if (error || !results.length) {
    return (
      <div className="text-center p-8 bg-white rounded-2xl shadow-xl">
        <h2 className="text-2xl font-bold">{error || 'No results to display.'}</h2>
        <button 
          onClick={() => navigate('/upload')} 
          className="mt-4 bg-primary text-white font-bold py-2 px-6 rounded-full hover:bg-blue-800"
        >
          Try Another Analysis
        </button>
      </div>
    );
  }

  const positivePercentage = (sentimentCounts[Sentiment.Positive] || 0) / results.length;
  const starRating = Math.max(1, Math.ceil(positivePercentage * 5));

  return (
    <div className="w-full max-w-6xl p-8 bg-white rounded-2xl shadow-xl">
      <h2 className="text-3xl font-bold text-center text-slate-800 mb-6">
        Analysis Results Using Model: <b><u>{model}</u></b>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="md:col-span-1 p-4 bg-slate-50 rounded-lg flex flex-col items-center justify-center">
          {selectedCountry && (
            <div className="flex flex-col items-center gap-2 mb-3">
              <Flag code={selectedCountry} className="w-8 h-6 object-cover rounded shadow-sm" />
              {detectedDialect && (
                <div className="text-sm text-slate-600 mt-1">
                  Detected Dialect: {detectedDialect}
                </div>
              )}
            </div>
          )}
          <h3 className="text-lg font-semibold text-slate-700 mb-2">Overall Rating</h3>
          <div className="flex">
            {[...Array(5)].map((_, i) => (
              <StarIcon key={i} className={`h-8 w-8 ${i < starRating ? 'text-accent' : 'text-slate-300'}`} />
            ))}
          </div>
        </div>
        <SummaryCard 
          title="Positive Reviews" 
          count={sentimentCounts[Sentiment.Positive] || 0} 
          colorClass="bg-green-100 text-green-800" 
        />
        <SummaryCard 
          title="Negative Reviews" 
          count={sentimentCounts[Sentiment.Negative] || 0} 
          colorClass="bg-red-100 text-red-800" 
        />
        <SummaryCard 
          title="Neutral Reviews" 
          count={sentimentCounts[Sentiment.Neutral] || 0} 
          colorClass="bg-yellow-100 text-yellow-800" 
        />
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Review Text
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Sentiment
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Topics
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {results.map((result, index) => (
              <tr key={index}>
                <td className="px-6 py-4 whitespace-normal text-sm text-slate-600">
                  {result.reviewText}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex flex-col gap-1">
                    <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${sentimentColors[result.sentiment].bg} ${sentimentColors[result.sentiment].text}`}>
                      {result.sentiment}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                  {result.sentimentScore !== undefined ? result.sentimentScore.toFixed(2) : '-'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                  <div className="flex flex-wrap gap-2">
                    {result.topics.length > 0 ? result.topics.map((topic, i) => (
                      <span key={i} className="px-2 py-1 bg-slate-200 text-slate-700 rounded-md text-xs">
                        {typeof topic === 'string' ? topic : topic.topic}
                      </span>
                    )) : <span>-</span>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-8 text-center">
        <button
          onClick={() => navigate('/upload')}
          className="bg-primary text-white font-bold py-3 px-8 rounded-full hover:bg-blue-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
        >
          Do Another Analysis
        </button>
      </div>
    </div>
  );

  return (
    <div className="w-full max-w-6xl p-8 bg-white rounded-2xl shadow-xl">
      <h2 className="text-3xl font-bold text-center text-slate-800 mb-6">
        Analysis Results Using Model: <b><u>{model}</u></b>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="md:col-span-1 p-4 bg-slate-50 rounded-lg flex flex-col items-center justify-center">
          {selectedCountry && (
            <div className="flex flex-col items-center gap-2 mb-3">
              <Flag code={selectedCountry} className="w-8 h-6 object-cover rounded shadow-sm" />
              {detectedDialect && (
                <div className="text-sm text-slate-600 mt-1">
                  Detected Dialect: {detectedDialect}
                </div>
              )}
            </div>
          )}
          <h3 className="text-lg font-semibold text-slate-700 mb-2">Overall Rating</h3>
          <div className="flex">
            {[...Array(5)].map((_, i) => (
              <StarIcon key={i} className={`h-8 w-8 ${i < starRating ? 'text-accent' : 'text-slate-300'}`} />
            ))}
          </div>
        </div>
        <SummaryCard 
          title="Positive Reviews" 
          count={sentimentCounts[Sentiment.Positive] || 0} 
          colorClass="bg-green-100 text-green-800" 
        />
        <SummaryCard 
          title="Negative Reviews" 
          count={sentimentCounts[Sentiment.Negative] || 0} 
          colorClass="bg-red-100 text-red-800" 
        />
        <SummaryCard 
          title="Neutral Reviews" 
          count={sentimentCounts[Sentiment.Neutral] || 0} 
          colorClass="bg-yellow-100 text-yellow-800" 
        />
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Review Text
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Sentiment
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Topics
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {results.map((result, index) => (
              <tr key={index}>
                <td className="px-6 py-4 whitespace-normal text-sm text-slate-600">
                  {result.reviewText}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex flex-col gap-1">
                    <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${sentimentColors[result.sentiment].bg} ${sentimentColors[result.sentiment].text}`}>
                      {result.sentiment}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                  {result.sentimentScore !== undefined ? result.sentimentScore.toFixed(2) : '-'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                  <div className="flex flex-wrap gap-2">
                    {result.topics.length > 0 ? result.topics.map((topic, i) => (
                      <span key={i} className="px-2 py-1 bg-slate-200 text-slate-700 rounded-md text-xs">
                        {typeof topic === 'string' ? topic : topic.topic}
                      </span>
                    )) : <span>-</span>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-8 text-center">
        <button
          onClick={() => navigate('/upload')}
          className="bg-primary text-white font-bold py-3 px-8 rounded-full hover:bg-blue-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
        >
          Do Another Analysis
        </button>
      </div>
    </div>
  );
};

export default ResultsPage;
