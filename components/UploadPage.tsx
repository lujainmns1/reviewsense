
import React, { useState, useCallback } from 'react';
import Flag from 'react-world-flags';

interface ArabicCountry {
  code: string;
  name: string;
  dialect: string;
}

const ARABIC_COUNTRIES: ArabicCountry[] = [
  { code: 'EG', name: 'Egypt', dialect: 'EGY' },
  { code: 'SA', name: 'Saudi Arabia', dialect: 'GLF' },
  { code: 'AE', name: 'UAE', dialect: 'GLF' },
  { code: 'KW', name: 'Kuwait', dialect: 'GLF' },
  { code: 'BH', name: 'Bahrain', dialect: 'GLF' },
  { code: 'QA', name: 'Qatar', dialect: 'GLF' },
  { code: 'OM', name: 'Oman', dialect: 'GLF' },
  { code: 'JO', name: 'Jordan', dialect: 'LEV' },
  { code: 'LB', name: 'Lebanon', dialect: 'LEV' },
  { code: 'SY', name: 'Syria', dialect: 'LEV' },
  { code: 'PS', name: 'Palestine', dialect: 'LEV' },
  { code: 'MA', name: 'Morocco', dialect: 'MAGHREB' },
  { code: 'DZ', name: 'Algeria', dialect: 'MAGHREB' },
  { code: 'TN', name: 'Tunisia', dialect: 'MAGHREB' },
  { code: 'LY', name: 'Libya', dialect: 'MAGHREB' }
];

interface UploadPageProps {
  onAnalyze: (reviews: string[], model: string, country?: string, autoDetectDialect?: boolean) => void;
  error: string | null;
}

const UploadPage: React.FC<UploadPageProps> = ({ onAnalyze, error }) => {
  const [reviewsText, setReviewsText] = useState('');
  const [model, setModel] = useState('arabert-arsas-sa');
  const [fileName, setFileName] = useState('');
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [autoDetectDialect, setAutoDetectDialect] = useState(false);

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
    onAnalyze(reviews, model, selectedCountry || undefined, autoDetectDialect);
  };

  return (
    <div className="w-full max-w-3xl p-8 bg-white rounded-2xl shadow-xl">
      <h2 className="text-3xl font-bold text-center text-slate-800 mb-2">Provide Your Reviews</h2>
      <p className="text-center text-slate-500 mb-8">Upload a CSV file or paste your reviews into the text box below.</p>
      {/* chose model */}
      <div className="space-y-4">
        <div className="mb-4">
          <label htmlFor="model" className="block text-sm font-medium text-slate-700 mb-1">Choose Analysis Model</label>
          <select id="model" name="model" className="block w-full border border-slate-300 rounded-lg p-2 focus:ring-2 focus:ring-primary focus:outline-none" value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="arabert-arsas-sa">Default Model(AraBERTv2 ArSAS (Positive/Neutral/Negative/Mixed))</option>
            <option value="marbertv2-book-review-sa">MARBERTv2 Book Review (Positive/Neutral/Negative)</option>
            <option value="xlm-roberta-twitter-sa">XLM-RoBERTa Twitter (Multilingual)</option>
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">Dialect Detection</label>
          <div className="flex items-center mb-4">
            {/* <input
              id="autoDetect"
              type="checkbox"
              checked={autoDetectDialect}
              onChange={(e) => setAutoDetectDialect(e.target.checked)}
              className="h-4 w-4 text-primary border-slate-300 rounded focus:ring-primary"
            />
            <label htmlFor="autoDetect" className="ml-2 block text-sm text-slate-700">
              Auto-detect dialect
            </label> */}
          </div>

          {!autoDetectDialect && (
            <div className="grid grid-cols-3 gap-2 mt-2">
              {ARABIC_COUNTRIES.map((country) => (
                <button
                  key={country.code}
                  type="button"
                  onClick={() => setSelectedCountry(country.code)}
                  className={'flex items-center p-2 rounded-lg border ' +
                    (selectedCountry === country.code
                      ? 'border-primary bg-blue-50'
                      : 'border-slate-200 hover:border-primary') +
                    ' transition-colors'}
                >
                  <Flag
                    code={country.code}
                    className="w-6 h-4 object-cover rounded"
                  />
                  <span className="ml-2 text-sm text-slate-700">{country.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
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
            disabled={selectedCountry === null || selectedCountry === ''}
            type="submit"
            className={`w-full md:w-auto text-white font-bold py-3 px-12 rounded-full transition-all duration-300 transform shadow-lg ${selectedCountry == null ? 'bg-gray-200 cursor-not-allowed' : 'bg-primary hover:bg-blue-800 hover:scale-105'}`}
          >
            Analyze Reviews
          </button>
        </div>
      </form>
    </div>
  );
};

export default UploadPage;
