import { getAnalysisHistory, updateSessionName } from '@/services/authService';
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface AnalysisSession {
  session_id: number;
  name: string | null;
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
  const [editingSessionId, setEditingSessionId] = useState<number | null>(null);
  const [editingName, setEditingName] = useState<string>('');
  const [renaming, setRenaming] = useState(false);
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

  const handleStartRename = (session: AnalysisSession) => {
    setEditingSessionId(session.session_id);
    setEditingName(session.name || '');
  };

  const handleCancelRename = () => {
    setEditingSessionId(null);
    setEditingName('');
  };

  const handleSaveRename = async (sessionId: number) => {
    if (renaming) return;
    
    setRenaming(true);
    try {
      await updateSessionName(sessionId, editingName.trim() || '', user.id);
      // Update local state
      setSessions(prevSessions =>
        prevSessions.map(s =>
          s.session_id === sessionId
            ? { ...s, name: editingName.trim() || null }
            : s
        )
      );
      setEditingSessionId(null);
      setEditingName('');
    } catch (err: any) {
      setError(err.message || 'Failed to rename session');
    } finally {
      setRenaming(false);
    }
  };

  return (
    <div className="w-full text-slate-100">
      <div className="mb-6">
        <h1 className="text-3xl font-black text-white tracking-tight">Dashboard</h1>
        <p className="text-slate-400 mt-1">View and manage your analysis sessions</p>
      </div>

      <div className="max-w-7xl">
        {error && (
          <div className="bg-red-500/10 border border-red-500/40 text-red-100 px-4 py-3 rounded-2xl relative mb-4">
            {error}
          </div>
        )}

        <div className="mb-6 flex gap-3">
          <button
            onClick={handleNewAnalysis}
            className="bg-blue-500/80 text-white px-5 py-2.5 rounded-xl hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 font-semibold"
          >
            New Analysis
          </button>
          <button
            onClick={() => navigate('/topics')}
            className="bg-purple-500/80 text-white px-5 py-2.5 rounded-xl hover:bg-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 font-semibold"
          >
            Topic Classification
          </button>
        </div>

        {loading ? (
          <div className="text-center text-slate-400">Loading...</div>
        ) : sessions.length === 0 ? (
          <div className="text-center text-slate-400">
            No analysis sessions found. Start by creating a new analysis.
          </div>
        ) : (
          <div className="bg-slate-900/50 border border-white/5 shadow-xl shadow-black/40 overflow-hidden sm:rounded-3xl">
            <ul className="divide-y divide-white/5">
              {sessions.map((session) => (
                <li key={session.session_id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          {editingSessionId === session.session_id ? (
                            <div className="flex items-center gap-2 flex-1">
                              <input
                                type="text"
                                value={editingName}
                                onChange={(e) => setEditingName(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') {
                                    handleSaveRename(session.session_id);
                                  } else if (e.key === 'Escape') {
                                    handleCancelRename();
                                  }
                                }}
                                className="flex-1 px-3 py-1 bg-slate-800 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                placeholder="Session name..."
                                autoFocus
                                disabled={renaming}
                              />
                              <button
                                onClick={() => handleSaveRename(session.session_id)}
                                disabled={renaming}
                                className="px-3 py-1 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                {renaming ? 'Saving...' : 'Save'}
                              </button>
                              <button
                                onClick={handleCancelRename}
                                disabled={renaming}
                                className="px-3 py-1 bg-slate-700 text-white text-sm rounded-lg hover:bg-slate-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 flex-1">
                              <p className="text-sm font-semibold text-blue-300 truncate">
                                {session.name || `Session #${session.session_id}`}
                              </p>
                              <button
                                onClick={() => handleStartRename(session)}
                                className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded hover:bg-white/10 transition"
                                title="Rename session"
                              >
                                ✏️
                              </button>
                            </div>
                          )}
                          <div className="ml-2 flex-shrink-0 flex">
                            <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-400/30">
                              {session.reviews_count} reviews
                            </p>
                          </div>
                        </div>
                        <div className="mt-2 sm:flex sm:justify-between">
                          <div className="sm:flex">
                            <p className="flex items-center text-sm text-slate-400">
                              Country: {session.country_code || 'N/A'}
                            </p>
                            <p className="mt-2 flex items-center text-sm text-slate-400 sm:mt-0 sm:ml-6">
                              Dialect: {session.detected_dialect || 'N/A'}
                            </p>
                          </div>
                          <div className="mt-2 flex items-center text-sm text-slate-400 sm:mt-0">
                            <p>
                              {new Date(session.created_at).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <div className="mt-2">
                          <p className="text-sm text-slate-400">
                            Models used: {session.models_used.join(', ')}
                          </p>
                        </div>
                      </div>
                      <div className="ml-5 flex-shrink-0">
                        <button
                          onClick={() => handleViewResults(session.session_id)}
                          className="text-blue-300 hover:text-white font-semibold"
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
              <div className="bg-slate-900/60 px-4 py-4 flex items-center justify-between border-t border-white/10 rounded-b-3xl sm:px-6">
                <div className="flex flex-1 items-center justify-between gap-4 flex-wrap text-sm">
                  <button
                    onClick={() => fetchAnalysisHistory(pagination.current_page - 1)}
                    disabled={pagination.current_page === 1}
                    className={`inline-flex items-center px-4 py-2 border rounded-xl font-medium transition ${
                      pagination.current_page === 1
                        ? 'cursor-not-allowed border-white/5 text-slate-500 bg-white/5'
                        : 'border-white/10 text-white bg-transparent hover:bg-white/10'
                    }`}
                  >
                    Previous
                  </button>
                  <p className="text-slate-400">
                    Page <span className="font-semibold text-white">{pagination.current_page}</span> of{' '}
                    <span className="font-semibold text-white">{pagination.pages}</span>
                  </p>
                  <button
                    onClick={() => fetchAnalysisHistory(pagination.current_page + 1)}
                    disabled={pagination.current_page === pagination.pages}
                    className={`inline-flex items-center px-4 py-2 border rounded-xl font-medium transition ${
                      pagination.current_page === pagination.pages
                        ? 'cursor-not-allowed border-white/5 text-slate-500 bg-white/5'
                        : 'border-white/10 text-white bg-transparent hover:bg-white/10'
                    }`}
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