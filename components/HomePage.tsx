
import React from 'react';

interface HomePageProps {
  onStart: () => void;
}

const HomePage: React.FC<HomePageProps> = ({ onStart }) => {
  return (
    <div className="text-center p-8 max-w-2xl mx-auto bg-white rounded-2xl shadow-xl">
      <h1 className="text-5xl font-extrabold text-primary mb-4">
        ReviewSense
      </h1>
      <p className="text-slate-600 mb-8 text-lg">
        Unlock insights from your customer feedback. ReviewSense uses AI to analyze product reviews for sentiment and key topics, helping you understand what your customers are really saying.
      </p>
      <button
        onClick={onStart}
        className="bg-primary text-white font-bold py-3 px-8 rounded-full hover:bg-blue-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
      >
        Get Started
      </button>
    </div>
  );
};

export default HomePage;
