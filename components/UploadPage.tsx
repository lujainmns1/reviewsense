"use client";

import React, { useState, useCallback, useMemo, useEffect } from "react";
import Flag from "react-world-flags";
import { Info, Sparkles, Upload, FileText, HelpCircle, CheckCircle2, Languages, X } from "lucide-react";

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
        <div role="dialog" aria-modal="true" className="w-full max-w-lg rounded-2xl bg-white shadow-2xl border border-slate-100">
          <div className="flex items-start justify-between p-4">
            <div>
              <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2"><Info className="h-5 w-5 text-blue-700" /> {title}</h3>
              {subtitle && <p className="mt-0.5 text-sm text-slate-600">{subtitle}</p>}
            </div>
            <button onClick={onClose} aria-label="Close" className="p-1 rounded-lg hover:bg-slate-100"><X className="h-5 w-5" /></button>
          </div>
          <div className="px-4 pb-4">{children}</div>
          <div className="p-4 pt-0 flex justify-end">
            <button onClick={onClose} className="inline-flex items-center gap-2 rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-medium hover:bg-blue-700">Got it</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export interface UploadPageProps {
  onAnalyze: (formData: FormData) => void;
  error: string | null;
}

const UploadPage: React.FC<UploadPageProps> = ({ onAnalyze, error }) => {
  const [reviewsText, setReviewsText] = useState("");
  const [model, setModel] = useState<keyof typeof MODEL_META>("arabert-arsas-sa");
  const [fileName, setFileName] = useState("");
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [autoDetectDialect, setAutoDetectDialect] = useState(false);
  const [openModelInfo, setOpenModelInfo] = useState<keyof typeof MODEL_META | null>(null);
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
    if (!user.id) {
      // Redirect to login if no user
      window.location.href = '/auth/login';
      return;
    }
    const formData = new FormData();
    formData.append('text', reviewsText);
    formData.append('model', model);
    formData.append('country', selectedCountry || '');
    formData.append('auto_detect', autoDetectDialect.toString());
    formData.append('user_id', user.id.toString());
    onAnalyze(formData);
  };

  const selectedModel = useMemo(() => MODEL_META[model], [model]);

  return (
    <div className="w-full max-w-4xl p-6 md:p-8 bg-white rounded-3xl shadow-xl border border-slate-100">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight flex items-center gap-2">
            <Sparkles className="h-6 w-6" aria-hidden /> Provide Your Reviews
          </h2>
          <p className="text-slate-500 mt-1">Upload a CSV file or paste reviews. Then choose a model and (optionally) a dialect.</p>
        </div>
      </div>

      {/* Model selection */}
      <div className="mt-6 grid grid-cols-1 gap-4">
        <label htmlFor="model" className="text-sm font-medium text-slate-700">Choose Analysis Model</label>
        <div className="flex items-center gap-3">
          <select
            id="model"
            name="model"
            className="block w-full border border-slate-300 rounded-xl p-2.5 focus:ring-2 focus:ring-blue-600 focus:outline-none"
            value={model}
            onChange={(e) => setModel(e.target.value as keyof typeof MODEL_META)}
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
            className="group inline-flex items-center justify-center h-10 w-10 rounded-xl border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition relative"
          >
            <Info className="h-5 w-5 text-slate-700" />
            <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 text-[10px] text-white opacity-0 group-hover:opacity-100">Click for strengths</span>
          </button>
        </div>

        {/* Selected model quick summary */}
        <div className="rounded-2xl border border-slate-200 p-4 bg-slate-50/60">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-slate-900 font-semibold">{selectedModel.title}</span>
                <span className="inline-flex items-center rounded-full border border-blue-200 text-blue-700 px-2 py-0.5 text-xs">{selectedModel.badge}</span>
              </div>
              <p className="text-sm text-slate-600 mt-0.5">{selectedModel.subtitle}</p>
            </div>
            <button
              type="button"
              onClick={() => setOpenModelInfo(model)}
              className="text-sm inline-flex items-center gap-1 text-blue-700 hover:underline"
            >
              <HelpCircle className="h-4 w-4" /> Learn more
            </button>
          </div>
        </div>
      </div>

      {/* Dialect */}
      <div className="mt-6">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-slate-700">Dialect Detection</label>
          <div className="flex items-center gap-2 text-xs text-slate-500">
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
                      ? "border-blue-600 bg-blue-50 shadow-sm"
                      : "border-slate-200 hover:border-blue-400 hover:bg-slate-50"
                  }`}
                  aria-pressed={active}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Flag code={country.code} className="w-6 h-4 object-cover rounded" />
                    <span className="truncate text-sm text-slate-700">{country.name}</span>
                  </span>
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] ${active ? "bg-blue-600 text-white" : "border border-slate-300 text-slate-700"}`}>
                    {country.dialect}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-xl" role="alert">{error}</div>
      )}

      {/* Upload + Text */}
      <form onSubmit={handleSubmit} className="mt-6">
        <div className="flex flex-col md:flex-row gap-6">
          {/* File Upload */}
          <div className="flex-1">
            <label htmlFor="file-upload" className="block text-sm font-medium text-slate-700 mb-2">Upload CSV</label>
            <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-2xl p-6 text-center bg-slate-50 hover:border-blue-500 transition-colors">
              <Upload className="h-10 w-10 text-slate-400 mb-3" aria-hidden />
              <label htmlFor="file-upload" className="cursor-pointer bg-white text-blue-700 font-semibold py-2 px-4 border border-blue-600 rounded-xl hover:bg-blue-50">Choose file</label>
              <input id="file-upload" name="file-upload" type="file" className="sr-only" accept=".csv,text/csv" onChange={handleFileChange} />
              {fileName ? (
                <p className="text-sm text-slate-600 mt-3 inline-flex items-center gap-1"><CheckCircle2 className="h-4 w-4" /> {fileName}</p>
              ) : (
                <p className="text-xs text-slate-500 mt-3">CSV with one review per line. Headers are ignored.</p>
              )}
            </div>
          </div>

          <div className="hidden md:flex items-center text-slate-400 select-none" aria-hidden>OR</div>

          {/* Text Area */}
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-2">Paste Reviews</label>
            <div className="relative">
              <FileText className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
              <textarea
                value={reviewsText}
                onChange={(e) => setReviewsText(e.target.value)}
                placeholder="Enter one review per line…"
                className="w-full h-48 p-4 pl-10 border border-slate-300 rounded-2xl focus:ring-2 focus:ring-blue-600 focus:outline-none transition-shadow"
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">Tip: You can mix Arabic and English text. The multilingual model handles code‑switching well.</p>
          </div>
        </div>

        <div className="mt-8 flex items-center justify-center">
          <button  disabled={selectedCountry === null || selectedCountry === ''} type="submit" 
            className={`w-full md:w-auto text-white font-bold py-3 px-12 rounded-full transition-all duration-300 transform shadow-lg ${selectedCountry == null ? 'bg-gray-200 cursor-not-allowed' : 'bg-primary hover:bg-blue-800 hover:scale-105'}`}
            >
              Analyze Reviews
          </button>
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
          <div className="space-y-3">
            <div>
              <p className="text-sm font-semibold text-slate-800">Strengths</p>
              <ul className="mt-1 list-disc list-inside space-y-1 text-sm text-slate-700">
                {MODEL_META[openModelInfo].strengths.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-800">Best when…</p>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {MODEL_META[openModelInfo].idealFor.map((tag) => (
                  <span key={tag} className="inline-flex items-center rounded-full border border-slate-300 px-2 py-0.5 text-xs text-slate-700">{tag}</span>
                ))}
              </div>
            </div>
            <p className="text-sm text-slate-600"><span className="font-medium">Note: </span>{MODEL_META[openModelInfo].notes}</p>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default UploadPage;
