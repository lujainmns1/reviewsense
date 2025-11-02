import { getAnalysisHistory } from '@/services/authService';
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface AnalysisSession {
  session_id: number;
  created_at: string;
  country_code: string;
  detected_dialect: string;
  reviews_count: number;
  models_used: string[];
}

interface PaginationInfo {
  total: number;
  pages: number;
  current_page: number;
  per_page: number;
}

const DashboardPage: React.FC = () => {
  const [sessions, setSessions] = useState<AnalysisSession[]>([]);
  const [pagination, setPagination] = useState<PaginationInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem('user') || '{}');

  useEffect(() => {
    // Redirect if not logged in
    if (!user.id) {
      navigate('/login');
      return;
    }

    fetchAnalysisHistory();
  }, [user.id]);

  const fetchAnalysisHistory = async (page: number = 1) => {
    try {
      setLoading(true);
      const data = await getAnalysisHistory(user.id, page);

      setSessions(data.history);
      setPagination(data.pagination);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNewAnalysis = () => {
    navigate('/upload');
  };

  const handleViewResults = (sessionId: number) => {
    navigate(`/results/${sessionId}`);
  };

  return (
    <div className="w-full">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">View and manage your analysis sessions</p>
      </div>

      <div className="max-w-7xl">
        {error && (
          <div className="bg-red-50 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            {error}
          </div>
        )}

        <div className="mb-6">
          <button
            onClick={handleNewAnalysis}
            className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            New Analysis
          </button>
        </div>

        {loading ? (
          <div className="text-center">Loading...</div>
        ) : sessions.length === 0 ? (
          <div className="text-center text-gray-500">
            No analysis sessions found. Start by creating a new analysis.
          </div>
        ) : (
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {sessions.map((session) => (
                <li key={session.session_id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-indigo-600 truncate">
                            Session #{session.session_id}
                          </p>
                          <div className="ml-2 flex-shrink-0 flex">
                            <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                              {session.reviews_count} reviews
                            </p>
                          </div>
                        </div>
                        <div className="mt-2 sm:flex sm:justify-between">
                          <div className="sm:flex">
                            <p className="flex items-center text-sm text-gray-500">
                              Country: {session.country_code || 'N/A'}
                            </p>
                            <p className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0 sm:ml-6">
                              Dialect: {session.detected_dialect || 'N/A'}
                            </p>
                          </div>
                          <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                            <p>
                              {new Date(session.created_at).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <div className="mt-2">
                          <p className="text-sm text-gray-500">
                            Models used: {session.models_used.join(', ')}
                          </p>
                        </div>
                      </div>
                      <div className="ml-5 flex-shrink-0">
                        <button
                          onClick={() => handleViewResults(session.session_id)}
                          className="text-indigo-600 hover:text-indigo-900"
                        >
                          View Results
                        </button>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>

            {pagination && pagination.pages > 1 && (
              <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
                <div className="flex-1 flex justify-between sm:hidden">
                  <button
                    onClick={() => fetchAnalysisHistory(pagination.current_page - 1)}
                    disabled={pagination.current_page === 1}
                    className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => fetchAnalysisHistory(pagination.current_page + 1)}
                    disabled={pagination.current_page === pagination.pages}
                    className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;