
import React,{useState , useEffect} from 'react';
import { AnalysisResult, Sentiment } from '../types';
import StarIcon from './icons/StarIcon';
import Flag from 'react-world-flags';

interface ResultsPageProps {
  results: AnalysisResult[];
  model: string;
  selectedCountry?: string;
  detectedDialect?: string;
  onAnalyzeAnother: () => void;
}


const sentimentColors: { [key in Sentiment]: { bg: string; text: string; border: string } } = {
  [Sentiment.Positive]: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-400' },
  [Sentiment.Negative]: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-400' },
  [Sentiment.Neutral]: { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-400' },
};

const ResultsPage: React.FC<ResultsPageProps> = ({ results, onAnalyzeAnother, model, selectedCountry, detectedDialect }) => {
  const [sentimentCounts,setSentimentCounts]= React.useState<{ [key in Sentiment]?: number }>({});
  console.log('Rendering ResultsPage with results:', results);
  if (results.length === 0) {
    return (
      <div className="text-center p-8 bg-white rounded-2xl shadow-xl">
        <h2 className="text-2xl font-bold">No results to display.</h2>
        <button onClick={onAnalyzeAnother} className="mt-4 bg-primary text-white font-bold py-2 px-6 rounded-full hover:bg-blue-800">
          Analyze Again
        </button>
      </div>
    );
  }

  useEffect(() => {
    const counts = results.reduce((acc, result) => {
      acc[result.sentiment] = (acc[result.sentiment] || 0) + 1;
      return acc;
    }, {} as { [key in Sentiment]?: number });
    setSentimentCounts(counts);
  }
  , [results]);

  const positivePercentage = (sentimentCounts[Sentiment.Positive] || 0) / results.length;
  const starRating = Math.max(1, Math.ceil(positivePercentage * 5));
  console.log('Sentiment counts:', sentimentCounts);
  const SummaryCard: React.FC<{ title: string; count: number; colorClass: string }> = ({ title, count, colorClass }) => (
   console.log('Rendering SummaryCard:', title, count),
   <div className={`p-4 rounded-lg text-center ${colorClass}`}>
      <p className="text-3xl font-bold">{count}</p>
      <p className="text-sm font-medium">{title}</p>
    </div>
  );

  return (
    <div className="w-full max-w-6xl p-8 bg-white rounded-2xl shadow-xl">
      <h2 className="text-3xl font-bold text-center text-slate-800 mb-6">Analysis Results Using Model : <b><u>{model}</u></b></h2> 

      {/* Summary Section */}
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
                                Detected Dialect: {detectedDialect}

            <h3 className="text-lg font-semibold text-slate-700 mb-2">Overall Rating</h3>
            <div className="flex">
                {[...Array(5)].map((_, i) => (
                    <StarIcon key={i} className={`h-8 w-8 ${i < starRating ? 'text-accent' : 'text-slate-300'}`} />
                ))}
            </div>
        </div>
        <SummaryCard title="Positive Reviews" count={sentimentCounts[Sentiment.Positive] || 0} colorClass="bg-green-100 text-green-800" />
        <SummaryCard title="Negative Reviews" count={sentimentCounts[Sentiment.Negative] || 0} colorClass="bg-red-100 text-red-800" />
        <SummaryCard title="Neutral Reviews" count={sentimentCounts[Sentiment.Neutral] || 0} colorClass="bg-yellow-100 text-yellow-800" />
      </div>

      {/* Results Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Review Text</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Sentiment</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Detected Topics</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {results.map((result, index) => (
              <tr key={index}>
                <td className="px-6 py-4 whitespace-normal text-sm text-slate-600">{result.reviewText}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${sentimentColors[result.sentiment].bg} ${sentimentColors[result.sentiment].text}`}>
                    {result.sentiment}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    <div className="flex flex-wrap gap-2">
                        {result.topics.length > 0 ? result.topics.map((topic, i) => (
                            <span key={i} className="px-2 py-1 bg-slate-200 text-slate-700 rounded-md text-xs">{typeof topic === 'string' ? topic : topic.topic}</span>
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
          onClick={onAnalyzeAnother}
          className="bg-primary text-white font-bold py-3 px-8 rounded-full hover:bg-blue-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
        >
          Do Another Analysis
        </button>
      </div>
    </div>
  );
};

export default ResultsPage;
