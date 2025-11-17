"use client";

import React, { useState, useCallback, useMemo, useEffect } from "react";
import Flag from "react-world-flags";
import { Info, Sparkles, Upload, FileText, HelpCircle, CheckCircle2, Languages, X } from "lucide-react";
import Loader from "./Loader";

interface ArabicCountry {
  code: string;
  name: string;
  dialect: string; // EGY | GLF | LEV | MAGHREB
}

const ARABIC_COUNTRIES: ArabicCountry[] = [
  { code: "EG", name: "Egypt", dialect: "EGY" },
  { code: "SA", name: "Saudi Arabia", dialect: "GLF" },
  { code: "AE", name: "UAE", dialect: "GLF" },
  { code: "KW", name: "Kuwait", dialect: "GLF" },
  { code: "BH", name: "Bahrain", dialect: "GLF" },
  { code: "QA", name: "Qatar", dialect: "GLF" },
  { code: "OM", name: "Oman", dialect: "GLF" },
  { code: "JO", name: "Jordan", dialect: "LEV" },
  { code: "LB", name: "Lebanon", dialect: "LEV" },
  { code: "SY", name: "Syria", dialect: "LEV" },
  { code: "PS", name: "Palestine", dialect: "LEV" },
  { code: "MA", name: "Morocco", dialect: "MAGHREB" },
  { code: "DZ", name: "Algeria", dialect: "MAGHREB" },
  { code: "TN", name: "Tunisia", dialect: "MAGHREB" },
  { code: "LY", name: "Libya", dialect: "MAGHREB" },
];

// === Model metadata with strengths/descriptions ===
const MODEL_META = {
  "arabert-arsas-sa": {
    title: "AraBERTv2 · ArSAS",
    subtitle: "4-class (Positive / Neutral / Negative / Mixed)",
    badge: "Default",
    strengths: [
      "Balanced performance on product/service reviews",
      "Handles neutral and mixed sentiments explicitly",
      "Robust MSA coverage with dialect tolerance",
    ],
    idealFor: ["E‑commerce summaries", "Customer support QA", "Balanced datasets"],
    notes: "If you need consistent 4‑class labels for dashboards, start here.",
  },
  "marbertv2-book-review-sa": {
    title: "MARBERTv2 · Book Review",
    subtitle: "3-class (Positive / Neutral / Negative)",
    badge: "Gulf‑aware",
    strengths: [
      "Strong performance on informal Arabic (incl. Gulf slang)",
      "Good at polarized opinions and emphatic expressions",
      "Stable on short social‑style sentences",
    ],
    idealFor: ["App store reviews", "Tweets", "Short comments"],
    notes: "Use for clearer positive/negative separation when ‘Mixed’ isn’t needed.",
  },
  "xlm-roberta-twitter-sa": {
    title: "XLM‑RoBERTa · Twitter (Multilingual)",
    subtitle: "3-class multilingual",
    badge: "Multilingual",
    strengths: [
      "Cross‑language batches in one run",
      "Resilient with code‑switching (Arabic ⇄ English)",
      "Useful for regional/global datasets",
    ],
    idealFor: ["Mixed‑language corpora", "Cross‑country analyses"],
    notes: "Prefer when reviews contain English, emojis, or non‑Arabic tokens.",
  },
} as const;

// ——— Lightweight Tailwind Modal (no shadcn) ———
function Modal({ open, onClose, title, subtitle, children }: {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    if (open) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-[1px]"
        onClick={onClose}
      />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div role="dialog" aria-modal="true" className="w-full max-w-lg rounded-2xl bg-slate-900 text-white shadow-2xl border border-white/10">
          <div className="flex items-start justify-between p-4">
            <div>
              <h3 className="text-base font-semibold text-white flex items-center gap-2"><Info className="h-5 w-5 text-blue-400" /> {title}</h3>
              {subtitle && <p className="mt-0.5 text-sm text-slate-300">{subtitle}</p>}
            </div>
            <button onClick={onClose} aria-label="Close" className="p-1 rounded-lg hover:bg-white/10"><X className="h-5 w-5" /></button>
          </div>
          <div className="px-4 pb-4">{children}</div>
          <div className="p-4 pt-0 flex justify-end">
            <button onClick={onClose} className="inline-flex items-center gap-2 rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-semibold hover:bg-blue-500">Got it</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export interface UploadPageProps {
  onAnalyze: (formData: FormData) => Promise<void>;
  error: string | null;
  isLoading?: boolean;
}

const UploadPage: React.FC<UploadPageProps> = ({ onAnalyze, error, isLoading = false }) => {
  const [reviewsText, setReviewsText] = useState("");
  const [model, setModel] = useState<keyof typeof MODEL_META>("arabert-arsas-sa");
  const [fileName, setFileName] = useState("");
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [autoDetectDialect, setAutoDetectDialect] = useState(false);
  const [openModelInfo, setOpenModelInfo] = useState<keyof typeof MODEL_META | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const user = JSON.parse(localStorage.getItem('user') || '{}');

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = (e.target?.result as string) || "";
        // Simple CSV/line parsing – assume one review per line
        const lines = text
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line);
        setReviewsText(lines.join("\n"));
      };
      reader.readAsText(file);
    }
  }, []);

    const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null); // Clear previous errors
    
    if (!user.id) {
      // Redirect to login if no user
      window.location.href = '/auth/login';
      return;
    }
    if (isLoading) {
      console.log('Already loading, preventing duplicate submission');
      return; // Prevent double submission
    }
    if (!selectedCountry || selectedCountry === '') {
      setLocalError('Please select a country for dialect detection');
      return;
    }
    if (!reviewsText.trim()) {
      setLocalError('Please provide at least one review to analyze');
      return;
    }
    
    console.log('Submitting form for analysis...');
    const formData = new FormData();
    formData.append('text', reviewsText);
    formData.append('model', model);
    formData.append('country', selectedCountry || '');
    formData.append('auto_detect', autoDetectDialect.toString());
    formData.append('user_id', user.id.toString());
    
    try {
      await onAnalyze(formData);
      console.log('Analysis request completed');
    } catch (error) {
      console.error('Error in handleSubmit:', error);
      // Error is handled by parent component, but we can set local error too
      setLocalError('Failed to analyze reviews. Please try again.');
    }
  };

  const selectedModel = useMemo(() => MODEL_META[model], [model]);

  return (
    <div className="w-full max-w-4xl mx-auto bg-slate-900/60 rounded-3xl shadow-2xl shadow-blue-500/10 border border-white/10 p-6 md:p-8 text-white">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight flex items-center gap-2">
            <Sparkles className="h-6 w-6" aria-hidden /> Provide Your Reviews
          </h2>
          <p className="text-slate-300 mt-1">Upload a CSV file or paste reviews. Then choose a model and (optionally) a dialect.</p>
        </div>
      </div>

      {/* Model selection */}
      <div className={`mt-6 grid grid-cols-1 gap-4 ${isLoading ? 'opacity-60 pointer-events-none' : ''}`}>
        <label htmlFor="model" className="text-sm font-semibold text-slate-200">Choose Analysis Model</label>
        <div className="flex items-center gap-3">
          <select
            id="model"
            name="model"
            className="block w-full border border-white/10 bg-slate-950/70 rounded-xl p-2.5 focus:ring-2 focus:ring-blue-600 focus:outline-none disabled:bg-slate-800 disabled:cursor-not-allowed text-white"
            value={model}
            onChange={(e) => setModel(e.target.value as keyof typeof MODEL_META)}
            disabled={isLoading}
          >
            <option value="arabert-arsas-sa">AraBERTv2 · ArSAS (Pos/Neu/Neg/Mixed)</option>
            <option value="marbertv2-book-review-sa">MARBERTv2 · Book Review (Pos/Neu/Neg)</option>
            <option value="xlm-roberta-twitter-sa">XLM‑RoBERTa · Twitter (Multilingual)</option>
          </select>

          {/* Info icon with accessible title and hover hint */}
          <button
            type="button"
            aria-label="Show model description"
            title="What is this model good at?"
            onClick={() => setOpenModelInfo(model)}
            className="group inline-flex items-center justify-center h-10 w-10 rounded-xl border border-white/10 hover:border-white/40 hover:bg-white/10 transition relative"
            disabled={isLoading}
          >
            <Info className="h-5 w-5 text-white" />
            <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-white/10 border border-white/20 px-2 py-1 text-[10px] text-white opacity-0 group-hover:opacity-100">Click for strengths</span>
          </button>
        </div>

        {/* Selected model quick summary */}
        <div className="rounded-2xl border border-white/10 p-4 bg-white/5">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-white font-semibold">{selectedModel.title}</span>
                <span className="inline-flex items-center rounded-full border border-blue-400/60 text-blue-200 px-2 py-0.5 text-xs">{selectedModel.badge}</span>
              </div>
              <p className="text-sm text-slate-300 mt-0.5">{selectedModel.subtitle}</p>
            </div>
            <button
              type="button"
              onClick={() => setOpenModelInfo(model)}
              className="text-sm inline-flex items-center gap-1 text-blue-200 hover:text-white"
            >
              <HelpCircle className="h-4 w-4" /> Learn more
            </button>
          </div>
        </div>
      </div>

      {/* Dialect */}
      <div className={`mt-6 ${isLoading ? 'opacity-60 pointer-events-none' : ''}`}>
        <div className="flex items-center justify-between">
          <label className="block text-sm font-semibold text-slate-200">Dialect Detection</label>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Languages className="h-4 w-4" />
            <span>EGY = Egyptian, GLF = Gulf, LEV = Levant, MAGHREB = Maghrebi</span>
          </div>
        </div>
{/* 
        <div className="flex items-center mt-2">
          <input
            id="autoDetect"
            type="checkbox"
            checked={autoDetectDialect}
            onChange={(e) => setAutoDetectDialect(e.target.checked)}
            className="h-4 w-4 text-blue-600 border-slate-300 rounded focus:ring-blue-600"
          />
          <label htmlFor="autoDetect" className="ml-2 block text-sm text-slate-700">Auto-detect dialect</label>
        </div> */}

        {!autoDetectDialect && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 mt-3">
            {ARABIC_COUNTRIES.map((country) => {
              const active = selectedCountry === country.code;
              return (
                <button
                  key={country.code}
                  type="button"
                  onClick={() => setSelectedCountry(country.code)}
                  className={`group flex items-center justify-between gap-2 p-2 rounded-xl border transition-all ${
                    active
                      ? "border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/20"
                      : "border-white/10 hover:border-blue-500/60 hover:bg-white/5"
                  } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                  aria-pressed={active}
                  disabled={isLoading}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Flag code={country.code} className="w-6 h-4 object-cover rounded" />
                    <span className="truncate text-sm text-slate-100">{country.name}</span>
                  </span>
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] ${active ? "bg-blue-500 text-white" : "border border-white/30 text-slate-200"}`}>
                    {country.dialect}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {(error || localError) && (
        <div className="mt-6 bg-red-500/10 border border-red-500/40 text-red-100 px-4 py-3 rounded-2xl" role="alert">{error || localError}</div>
      )}

      {/* Upload + Text */}
      <form onSubmit={handleSubmit} className="mt-6">
        <div className={`flex flex-col md:flex-row gap-6 ${isLoading ? 'opacity-60 pointer-events-none' : ''}`}>
          {/* File Upload */}
          <div className="flex-1">
            <label htmlFor="file-upload" className="block text-sm font-semibold text-slate-200 mb-2">Upload CSV</label>
            <div className={`flex flex-col items-center justify-center border-2 border-dashed border-white/15 rounded-2xl p-6 text-center bg-white/5 transition-colors ${isLoading ? '' : 'hover:border-blue-500/60'}`}>
              <Upload className="h-10 w-10 text-slate-300 mb-3" aria-hidden />
              <label htmlFor="file-upload" className={`${isLoading ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} bg-blue-500/10 text-blue-200 font-semibold py-2 px-4 border border-blue-400/60 rounded-xl ${isLoading ? '' : 'hover:bg-blue-500/20'}`}>Choose file</label>
              <input id="file-upload" name="file-upload" type="file" className="sr-only" accept=".csv,text/csv" onChange={handleFileChange} disabled={isLoading} />
              {fileName ? (
                <p className="text-sm text-slate-200 mt-3 inline-flex items-center gap-1"><CheckCircle2 className="h-4 w-4 text-emerald-300" /> {fileName}</p>
              ) : (
                <p className="text-xs text-slate-400 mt-3">CSV with one review per line. Headers are ignored.</p>
              )}
            </div>
          </div>

          <div className="hidden md:flex items-center text-slate-400 select-none" aria-hidden>OR</div>

          {/* Text Area */}
          <div className="flex-1">
            <label className="block text-sm font-semibold text-slate-200 mb-2">Paste Reviews</label>
            <div className="relative">
              <FileText className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
              <textarea
                value={reviewsText}
                onChange={(e) => setReviewsText(e.target.value)}
                placeholder="Enter one review per line…"
                className="w-full h-48 p-4 pl-10 border border-white/10 rounded-2xl bg-slate-950/70 text-white placeholder:text-slate-500 focus:ring-2 focus:ring-blue-600 focus:outline-none transition shadow-inner shadow-black/40 disabled:bg-slate-800 disabled:cursor-not-allowed"
                disabled={isLoading}
              />
            </div>
            <p className="text-xs text-slate-400 mt-2">Tip: You can mix Arabic and English text. The multilingual model handles code-switching well.</p>
          </div>
        </div>

        <div className="mt-8 flex items-center justify-center">
          {isLoading ? (
            <div className="w-full flex flex-col items-center gap-4 py-4">
              <Loader message="Analyzing reviews..." />
              <p className="text-sm text-slate-300 animate-pulse">This may take a moment...</p>
            </div>
          ) : (
            <button  
              disabled={selectedCountry === null || selectedCountry === '' || isLoading} 
              type="submit" 
              className={`w-full md:w-auto text-white font-bold py-3 px-12 rounded-full transition-all duration-300 transform shadow-lg ${
                selectedCountry === null || selectedCountry === '' || isLoading
                  ? 'bg-white/10 cursor-not-allowed text-slate-400'
                  : 'bg-blue-600 hover:bg-blue-500 hover:scale-105 shadow-blue-500/40'
              }`}
            >
              Analyze Reviews
            </button>
          )}
        </div>
      </form>

      {/* Model details modal */}
      <Modal
        open={!!openModelInfo}
        onClose={() => setOpenModelInfo(null)}
        title={openModelInfo ? MODEL_META[openModelInfo].title : ""}
        subtitle={openModelInfo ? MODEL_META[openModelInfo].subtitle : undefined}
      >
        {openModelInfo && (
          <div className="space-y-3 text-slate-100">
            <div>
              <p className="text-sm font-semibold text-white">Strengths</p>
              <ul className="mt-1 list-disc list-inside space-y-1 text-sm text-slate-200">
                {MODEL_META[openModelInfo].strengths.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-sm font-semibold text-white">Best when…</p>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {MODEL_META[openModelInfo].idealFor.map((tag) => (
                  <span key={tag} className="inline-flex items-center rounded-full border border-white/20 px-2 py-0.5 text-xs text-slate-100">{tag}</span>
                ))}
              </div>
            </div>
            <p className="text-sm text-slate-300"><span className="font-semibold text-white">Note: </span>{MODEL_META[openModelInfo].notes}</p>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default UploadPage;
