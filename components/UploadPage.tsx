
import React, { useState, useCallback } from 'react';

interface UploadPageProps {
  onAnalyze: (reviews: string[]) => void;
  error: string | null;
}

const UploadPage: React.FC<UploadPageProps> = ({ onAnalyze, error }) => {
  const [reviewsText, setReviewsText] = useState('');
  const [fileName, setFileName] = useState('');

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        // Simple CSV parsing: assumes one review per line, ignoring headers/columns for simplicity
        const lines = text.split('\n').map(line => line.trim()).filter(line => line);
        setReviewsText(lines.join('\n'));
      };
      reader.readAsText(file);
    }
  }, []);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const reviews = reviewsText.split('\n').map(line => line.trim()).filter(line => line);
    onAnalyze(reviews);
  };

  return (
    <div className="w-full max-w-3xl p-8 bg-white rounded-2xl shadow-xl">
      <h2 className="text-3xl font-bold text-center text-slate-800 mb-2">Provide Your Reviews</h2>
      <p className="text-center text-slate-500 mb-8">Upload a CSV file or paste your reviews into the text box below.</p>

      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-6" role="alert">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="flex flex-col md:flex-row gap-8">
          {/* File Upload */}
          <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-lg p-6 text-center bg-slate-50 hover:border-primary transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-slate-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <label htmlFor="file-upload" className="cursor-pointer bg-white text-primary font-semibold py-2 px-4 border border-primary rounded-lg hover:bg-blue-50">
              Upload a CSV file
            </label>
            <input id="file-upload" name="file-upload" type="file" className="sr-only" accept=".csv" onChange={handleFileChange} />
            {fileName && <p className="text-sm text-slate-500 mt-3">{fileName}</p>}
          </div>

          <div className="flex items-center text-slate-400">OR</div>

          {/* Text Area */}
          <div className="flex-1">
            <textarea
              value={reviewsText}
              onChange={(e) => setReviewsText(e.target.value)}
              placeholder="Enter one review per line..."
              className="w-full h-48 p-4 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary focus:outline-none transition-shadow"
            />
          </div>
        </div>

        <div className="mt-8 text-center">
          <button
            type="submit"
            className="w-full md:w-auto bg-primary text-white font-bold py-3 px-12 rounded-full hover:bg-blue-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
          >
            Analyze Reviews
          </button>
        </div>
      </form>
    </div>
  );
};

export default UploadPage;
