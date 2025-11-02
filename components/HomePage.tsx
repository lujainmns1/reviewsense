
import React from 'react';

interface HomePageProps {
  onStart: () => void;
}

import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem('user') || '{}');
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="text-center bg-white rounded-2xl shadow-xl p-8 md:p-12">
        <h1 className="text-5xl font-extrabold text-primary mb-4">
          ReviewSense
        </h1>
        <p className="text-slate-600 mb-8 text-lg">
          Unlock insights from your customer feedback. ReviewSense uses AI to analyze product reviews for sentiment and key topics, helping you understand what your customers are really saying.
        </p>
        {user.id ? (
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/dashboard')}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
            >
              Go to Dashboard
            </button>
            <button
              onClick={() => navigate('/upload')}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
            >
              New Analysis
            </button>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/login')}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
            >
              Sign In
            </button>
            <button
              onClick={() => navigate('/signup')}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
            >
              Sign Up
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;
