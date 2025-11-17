
import React from 'react';

interface LoaderProps {
  message?: string;
}

const Loader: React.FC<LoaderProps> = ({ message = "Loading..." }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 bg-slate-900/60 border border-white/10 rounded-2xl shadow-xl shadow-blue-500/20">
      <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mb-4"></div>
      <p className="text-lg text-slate-200 font-semibold">{message}</p>
    </div>
  );
};

export default Loader;
