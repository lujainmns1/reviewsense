import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getSessionResults } from '../services/backendService';
import { analyzeReviews } from '../services/geminiService';
import { AnalysisResult } from '../types';

interface TopicCount {
  topic: string;
  count: number;
  reviews: AnalysisResult[];
}

const TopicClassificationPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [topicCounts, setTopicCounts] = useState<TopicCount[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reviews, setReviews] = useState<AnalysisResult[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [uploadedText, setUploadedText] = useState<string>('');

  useEffect(() => {
    if (sessionId) {
      loadSessionReviews();
    }
  }, [sessionId]);

  const loadSessionReviews = async (reclassify: boolean = false) => {
    if (!sessionId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await getSessionResults(parseInt(sessionId, 10));
      setReviews(data.results);
      
      // Check if reviews already have topics and we don't need to reclassify
      const hasTopics = data.results.some(r => r.topics && r.topics.length > 0);
      
      if (hasTopics && !reclassify) {
        // Use existing topics
        countTopicsByReviews(data.results);
      } else {
        // Reclassify using geminiService
        await classifyTopics(data.results);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session reviews');
    } finally {
      setLoading(false);
    }
  };
  
  const countTopicsByReviews = (reviewsToCount: AnalysisResult[]) => {
    const topicMap = new Map<string, AnalysisResult[]>();
    
    reviewsToCount.forEach(review => {
      const topics = review.topics || [];
      topics.forEach(topic => {
        const topicName = typeof topic === 'string' ? topic : topic.topic;
        if (!topicMap.has(topicName)) {
          topicMap.set(topicName, []);
        }
        topicMap.get(topicName)!.push(review);
      });
    });
    
    // Convert to array and sort by count
    const counts: TopicCount[] = Array.from(topicMap.entries())
      .map(([topic, reviews]) => ({
        topic,
        count: reviews.length,
        reviews
      }))
      .sort((a, b) => b.count - a.count);
    
    setTopicCounts(counts);
  };

  const classifyTopics = async (reviewsToClassify: AnalysisResult[]) => {
    setLoading(true);
    setError(null);

    try {
      // Extract review texts
      const reviewTexts = reviewsToClassify.map(r => r.reviewText);
      
      // Use geminiService to analyze and get topics
      const analyzedResults = await analyzeReviews(reviewTexts);
      
      // Update reviews with analyzed results (in case topics weren't extracted before)
      const updatedReviews = reviewsToClassify.map((review, index) => ({
        ...review,
        topics: analyzedResults[index]?.topics || review.topics
      }));
      
      setReviews(updatedReviews);
      
      // Count reviews per topic
      const topicMap = new Map<string, AnalysisResult[]>();
      
      updatedReviews.forEach(review => {
        const topics = review.topics || [];
        topics.forEach(topic => {
          const topicName = typeof topic === 'string' ? topic : topic.topic;
          if (!topicMap.has(topicName)) {
            topicMap.set(topicName, []);
          }
          topicMap.get(topicName)!.push(review);
        });
      });
      
      // Convert to array and sort by count
      const counts: TopicCount[] = Array.from(topicMap.entries())
        .map(([topic, reviews]) => ({
          topic,
          count: reviews.length,
          reviews
        }))
        .sort((a, b) => b.count - a.count);
      
      setTopicCounts(counts);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify topics');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadAndClassify = async () => {
    if (!uploadedText.trim()) {
      setError('Please enter review text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Split by lines or new paragraphs
      const reviewTexts = uploadedText
        .split(/\n\s*\n/)
        .map(text => text.trim())
        .filter(text => text.length > 0);
      
      if (reviewTexts.length === 0) {
        setError('No valid reviews found in the text');
        return;
      }

      // Use geminiService to analyze
      const analyzedResults = await analyzeReviews(reviewTexts);
      setReviews(analyzedResults);
      
      // Count reviews per topic
      const topicMap = new Map<string, AnalysisResult[]>();
      
      analyzedResults.forEach(review => {
        const topics = review.topics || [];
        topics.forEach(topic => {
          const topicName = typeof topic === 'string' ? topic : topic.topic;
          if (!topicMap.has(topicName)) {
            topicMap.set(topicName, []);
          }
          topicMap.get(topicName)!.push(review);
        });
      });
      
      // Convert to array and sort by count
      const counts: TopicCount[] = Array.from(topicMap.entries())
        .map(([topic, reviews]) => ({
          topic,
          count: reviews.length,
          reviews
        }))
        .sort((a, b) => b.count - a.count);
      
      setTopicCounts(counts);
      setUploadedText(''); // Clear input after successful classification
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify topics');
    } finally {
      setLoading(false);
    }
  };

  const getTopicColor = (index: number): string => {
    const colors = [
      'bg-blue-500/15 border-blue-400/60 text-blue-200',
      'bg-emerald-500/15 border-emerald-400/60 text-emerald-200',
      'bg-purple-500/15 border-purple-400/60 text-purple-200',
      'bg-amber-500/15 border-amber-400/60 text-amber-200',
      'bg-rose-500/15 border-rose-400/60 text-rose-200',
      'bg-cyan-500/15 border-cyan-400/60 text-cyan-200',
      'bg-pink-500/15 border-pink-400/60 text-pink-200',
      'bg-indigo-500/15 border-indigo-400/60 text-indigo-200',
    ];
    return colors[index % colors.length];
  };

  const selectedTopicData = topicCounts.find(tc => tc.topic === selectedTopic);

  return (
    <div className="w-full max-w-6xl mx-auto bg-slate-900/60 border border-white/10 rounded-3xl shadow-2xl shadow-blue-500/10 p-8 text-white">
      <div className="text-center mb-6 space-y-2">
        <h2 className="text-3xl font-black">Topic Classification</h2>
        <p className="text-slate-300">Analyze and count reviews by topics</p>
      </div>

      {!sessionId && (
        <div className="mb-6 p-6 bg-slate-800/50 border border-white/10 rounded-2xl">
          <h3 className="text-lg font-semibold mb-3">Upload Reviews for Classification</h3>
          <textarea
            value={uploadedText}
            onChange={(e) => setUploadedText(e.target.value)}
            placeholder="Enter reviews here, one per line or separated by blank lines..."
            className="w-full h-32 p-3 bg-slate-900/50 border border-white/20 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          />
          <button
            onClick={handleUploadAndClassify}
            disabled={loading || !uploadedText.trim()}
            className="mt-4 bg-blue-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-blue-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Classifying...' : 'Classify Topics'}
          </button>
        </div>
      )}

      {sessionId && (
        <div className="mb-6 flex gap-3">
          <button
            onClick={() => loadSessionReviews(false)}
            disabled={loading}
            className="bg-blue-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-blue-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : 'Load Topics'}
          </button>
          <button
            onClick={() => loadSessionReviews(true)}
            disabled={loading}
            className="bg-purple-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-purple-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Re-classifying...' : 'Re-classify with Gemini'}
          </button>
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-rose-500/10 border border-rose-400/30 rounded-lg text-rose-200">
          {error}
        </div>
      )}

      {loading && !topicCounts.length && (
        <div className="text-center p-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto"></div>
          <p className="mt-4 text-slate-300">Classifying topics...</p>
        </div>
      )}

      {topicCounts.length > 0 && (
        <>
          <div className="mb-6">
            <h3 className="text-xl font-semibold mb-4">Topic Distribution</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {topicCounts.map((topicCount, index) => (
                <div
                  key={topicCount.topic}
                  onClick={() => setSelectedTopic(selectedTopic === topicCount.topic ? null : topicCount.topic)}
                  className={`p-5 rounded-2xl border cursor-pointer transition-all hover:scale-105 ${getTopicColor(index)} ${
                    selectedTopic === topicCount.topic ? 'ring-2 ring-blue-400' : ''
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-lg font-bold truncate">{topicCount.topic}</h4>
                    <span className="text-2xl font-black ml-2">{topicCount.count}</span>
                  </div>
                  <p className="text-sm opacity-80">
                    {topicCount.count} {topicCount.count === 1 ? 'review' : 'reviews'}
                  </p>
                  <div className="mt-3 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-current opacity-50 transition-all"
                      style={{ width: `${(topicCount.count / reviews.length) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedTopicData && (
            <div className="mt-6 p-6 bg-slate-800/50 border border-white/10 rounded-2xl">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">
                  Reviews for "{selectedTopicData.topic}" ({selectedTopicData.count})
                </h3>
                <button
                  onClick={() => setSelectedTopic(null)}
                  className="text-slate-400 hover:text-white"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {selectedTopicData.reviews.map((review, index) => (
                  <div
                    key={index}
                    className="p-4 bg-slate-900/50 border border-white/10 rounded-lg"
                  >
                    <p className="text-slate-200 mb-2">{review.reviewText}</p>
                    <div className="flex flex-wrap gap-2">
                      <span className={`px-2 py-1 text-xs rounded ${
                        review.sentiment === 'positive' ? 'bg-emerald-500/20 text-emerald-200' :
                        review.sentiment === 'negative' ? 'bg-rose-500/20 text-rose-200' :
                        'bg-amber-500/20 text-amber-200'
                      }`}>
                        {review.sentiment}
                      </span>
                      {review.topics.filter(t => {
                        const topicName = typeof t === 'string' ? t : t.topic;
                        return topicName !== selectedTopicData.topic;
                      }).map((topic, i) => {
                        const topicName = typeof topic === 'string' ? topic : topic.topic;
                        return (
                          <span key={i} className="px-2 py-1 bg-white/10 text-white rounded text-xs border border-white/20">
                            {topicName}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6 p-4 bg-slate-800/30 border border-white/10 rounded-lg">
            <p className="text-sm text-slate-300">
              <span className="font-semibold">Total Reviews:</span> {reviews.length} |{' '}
              <span className="font-semibold">Total Topics:</span> {topicCounts.length}
            </p>
          </div>
        </>
      )}

      <div className="mt-8 text-center">
        <button
          onClick={() => navigate('/dashboard')}
          className="bg-slate-700 text-white font-bold py-3 px-8 rounded-full hover:bg-slate-600 transition-all duration-300"
        >
          Back to Dashboard
        </button>
      </div>
    </div>
  );
};

export default TopicClassificationPage;

