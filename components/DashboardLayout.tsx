import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { LayoutDashboard, Upload, BarChart3, LogOut, User } from 'lucide-react';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem('user') || '{}');

  const handleLogout = () => {
    localStorage.removeItem('user');
    navigate('/login');
  };

  const navigation = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Upload', path: '/upload', icon: Upload },
  ];

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(`${path}/`);
  };

  // Check if we're on results page
  const isResultsPage = location.pathname.startsWith('/results');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col lg:flex-row">
      {/* Sidebar */}
      <aside className="w-full lg:w-64 bg-slate-900/60 border-b lg:border-b-0 lg:border-r border-white/10 backdrop-blur flex flex-col">
        {/* Logo/Header */}
        <div className="p-6 border-b border-white/10">
          <h1 className="text-2xl font-black text-white tracking-tight">ReviewSense</h1>
          <p className="text-xs text-slate-400 mt-1 uppercase tracking-[0.4em]">Sentiment Intelligence</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              return (
                <li key={item.path}>
                  <Link
                    to={item.path}
                    className={`flex items-center gap-3 px-4 py-3 rounded-2xl border transition-all ${
                      active
                        ? 'border-blue-500 bg-blue-500/20 text-white shadow-lg shadow-blue-500/30'
                        : 'border-white/5 text-slate-300 hover:border-white/20 hover:bg-white/5'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <span className="font-medium">{item.name}</span>
                  </Link>
                </li>
              );
            })}
            {/* Results link - only show when on results page */}
            {isResultsPage && (
              <li>
                <div className="flex items-center gap-3 px-4 py-3 rounded-2xl border border-emerald-400/60 bg-emerald-500/10 text-white shadow-lg shadow-emerald-500/30">
                  <BarChart3 className="h-5 w-5" />
                  <span className="font-medium">Results</span>
                </div>
              </li>
            )}
          </ul>
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-white/10">
          <div className="flex items-center gap-3 px-4 py-2 text-slate-200">
            <div className="h-10 w-10 rounded-2xl bg-white/10 flex items-center justify-center">
              <User className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold truncate text-white">{user.email || 'User'}</p>
              <p className="text-xs text-slate-400 truncate">Logged in</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-4 py-3 mt-3 rounded-2xl text-slate-200 border border-white/10 hover:border-red-500/60 hover:bg-red-500/10 hover:text-white transition-colors"
          >
            <LogOut className="h-5 w-5" />
            <span className="font-medium">Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-slate-950 text-slate-100">
        <div className="p-6 md:p-8">
          {children}
        </div>
      </main>
    </div>
  );
};

export default DashboardLayout;
