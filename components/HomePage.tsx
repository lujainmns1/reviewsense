
import React from 'react';
import { useNavigate } from 'react-router-dom';

type StoredUser = {
  id?: string | number;
  email?: string;
};

const keyFeatures = [
  {
    title: 'Multi-language support',
    description: 'Analyze Modern Standard Arabic, Gulf dialect, Egyptian, Maghreb, and English reviews in one workspace.',
  },
  {
    title: 'Fast AI sentiment analysis',
    description: 'Transformer-based microservices deliver instant polarity, intensity, and emotion signals.',
  },
  {
    title: 'Topic extraction',
    description: 'Automatically highlights the core themes mentioned inside each review—even mixed-language text.',
  },
  {
    title: 'Saved analysis history',
    description: 'Every session is archived so teams can revisit their insights from the dashboard any time.',
  },
];

const howItWorksSteps = [
  'Create an account or sign in',
  'Enter or upload reviews',
  'Choose the AI model',
  'Receive sentiment, score, topics, and detected dialect',
  'View your full history in the dashboard',
];

const whyReviewSense = [
  {
    title: 'Advanced model stack',
    detail: 'AraBERTv2, MARBERTv2, and XLM-RoBERTa optimized for Arabic + English feedback.',
  },
  {
    title: 'Accuracy across dialects',
    detail: 'Understands formal Arabic, dialects, and blended Arabic-English comments without manual tuning.',
  },
  {
    title: 'Clean, human-friendly UI',
    detail: 'Surface sentiment, scores, and topics with zero clutter, ready for CX, ops, and product teams.',
  },
  {
    title: 'Robust backend',
    detail: 'Flask + PostgreSQL + Docker stack engineered for reliability and enterprise deployments.',
  },
];

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const user = (JSON.parse(localStorage.getItem('user') || 'null') as StoredUser | null) || null;

  const handleScrollTo = (sectionId: string) => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const primaryCta = () => {
    if (user?.id) {
      navigate('/dashboard');
      return;
    }
    navigate('/signup');
  };

  const secondaryCta = () => {
    if (user?.id) {
      navigate('/upload');
      return;
    }
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="sticky top-0 z-20 bg-slate-950/80 backdrop-blur border-b border-white/10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="text-2xl font-black tracking-tight">ReviewSense</div>
          <nav className="hidden md:flex gap-8 text-sm text-slate-200">
            <button onClick={() => handleScrollTo('features')} className="hover:text-white transition">Key Features</button>
            <button onClick={() => handleScrollTo('how-it-works')} className="hover:text-white transition">How it works</button>
            <button onClick={() => handleScrollTo('why-us')} className="hover:text-white transition">Why ReviewSense</button>
          </nav>
          <div className="flex gap-3">
            {user?.id ? (
              <>
                <button
                  onClick={() => navigate('/dashboard')}
                  className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-sm font-semibold"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => navigate('/upload')}
                  className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-sm font-semibold"
                >
                  New Analysis
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => navigate('/signup')}
                  className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-sm font-semibold"
                >
                  Create Account
                </button>
                <button
                  onClick={() => navigate('/login')}
                  className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-sm font-semibold"
                >
                  Sign In
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4">
        <section id="hero" className="py-20 flex flex-col lg:flex-row gap-12 items-center">
          <div className="flex-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 text-xs uppercase tracking-wide text-slate-200 mb-6">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              New: Arabic dialect detection
            </div>
            <h1 className="text-4xl md:text-6xl font-black leading-tight mb-6">
              Understand every review with <span className="text-blue-400">AI clarity</span>.
            </h1>
            <p className="text-lg md:text-xl text-slate-200 mb-10">
              ReviewSense transforms raw Arabic and English reviews into accurate sentiment insights, topic extraction,
              and dialect detection — all in real time.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 mb-8">
              <button
                onClick={primaryCta}
                className="flex-1 px-6 py-4 rounded-xl bg-blue-500 hover:bg-blue-600 text-base font-semibold shadow-lg shadow-blue-500/30"
              >
                {user?.id ? 'Go to Dashboard' : 'Create Account'}
              </button>
              <button
                onClick={secondaryCta}
                className="flex-1 px-6 py-4 rounded-xl border border-white/30 hover:border-white/60 text-base font-semibold"
              >
                {user?.id ? 'Upload data' : 'Sign In'}
              </button>
            </div>
            <div className="grid grid-cols-2 gap-6 text-left text-slate-300">
              <div>
                <p className="text-3xl font-bold text-white">Multi-language</p>
                <p className="text-sm mt-1">Seamless Arabic + English pipelines with dialect detection.</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-white">Human-ready</p>
                <p className="text-sm mt-1">Narratives and summaries your leadership team can trust.</p>
              </div>
            </div>
          </div>
          <div className="flex-1 w-full">
            <div className="relative bg-gradient-to-br from-blue-500/30 via-purple-500/20 to-emerald-500/20 rounded-3xl border border-white/10 p-8 shadow-2xl shadow-blue-600/10">
              <div className="absolute inset-0 blur-3xl bg-blue-500/20 -z-10" />
              <div className="flex items-center justify-between mb-6">
                <p className="text-sm font-semibold text-slate-200 uppercase tracking-wide">Live insight</p>
                <span className="px-3 py-1 rounded-full bg-white/10 text-xs">Updated 2m ago</span>
              </div>
              <div className="space-y-6">
                <div className="bg-slate-900/60 rounded-2xl p-5 border border-white/10">
                  <p className="text-sm text-slate-300 mb-3">Sentiment blend</p>
                  <div className="flex items-end gap-4">
                    <div className="flex-1">
                      <div className="h-32 rounded-2xl bg-gradient-to-t from-emerald-400 to-emerald-200 flex items-end">
                        <div className="w-full h-24 bg-gradient-to-t from-emerald-500 to-emerald-300 rounded-2xl" />
                      </div>
                    </div>
                    <div className="w-1/3 text-right">
                      <p className="text-4xl font-black">72%</p>
                      <p className="text-xs uppercase tracking-wide text-emerald-300 mt-1">Positive</p>
                    </div>
                  </div>
                </div>
                <div className="bg-slate-900/60 rounded-2xl p-5 border border-white/10">
                  <p className="text-sm text-slate-300 mb-4">Emerging topics</p>
                  <ul className="space-y-3">
                    {['ATM reliability', 'Loan approvals', 'Mobile UX'].map((topic) => (
                      <li key={topic} className="flex items-center justify-between text-sm">
                        <span>{topic}</span>
                        <span className="text-emerald-300">+18%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="py-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <p className="text-xs uppercase tracking-[0.5em] text-blue-300">Key features</p>
            <h2 className="text-3xl md:text-4xl font-black mt-3">Designed for review-heavy teams</h2>
            <p className="text-slate-300 mt-4">
              Everything you need to capture, understand, and act on thousands of Arabic and English reviews without
              spreadsheets or manual tagging.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {keyFeatures.map((feature) => (
              <div key={feature.title} className="rounded-3xl border border-white/10 bg-white/5 p-8 shadow-lg shadow-black/20">
                <h3 className="text-2xl font-bold mb-3 text-white">{feature.title}</h3>
                <p className="text-slate-200 text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="how-it-works" className="py-16">
          <div className="rounded-[32px] border border-white/10 bg-white/5 p-10">
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-8">
              <div className="md:w-1/3">
                <p className="text-xs uppercase tracking-[0.5em] text-blue-300">How it works</p>
                <h2 className="text-3xl font-black mt-3">Five steps to sentiment clarity</h2>
                <p className="text-slate-300 mt-4 text-sm">
                  ReviewSense guides teams end-to-end—from signing in to mining every historical insight.
                </p>
              </div>
              <ol className="flex-1 space-y-6">
                {howItWorksSteps.map((step, index) => (
                  <li key={step} className="flex gap-4 items-start">
                    <span className="h-10 w-10 rounded-full bg-blue-500/20 border border-blue-400/60 flex items-center justify-center text-lg font-bold">
                      {index + 1}
                    </span>
                    <p className="text-slate-100 text-base">{step}</p>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        <section id="why-us" className="py-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <p className="text-xs uppercase tracking-[0.5em] text-blue-300">Why ReviewSense?</p>
            <h2 className="text-3xl md:text-4xl font-black mt-3">Engineered for accuracy and trust</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {whyReviewSense.map((item) => (
              <div key={item.title} className="rounded-3xl border border-white/10 bg-gradient-to-b from-white/10 to-white/5 p-8 shadow-lg shadow-black/20">
                <h3 className="text-2xl font-bold mb-3 text-white">{item.title}</h3>
                <p className="text-slate-200 text-sm leading-relaxed">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="cta" className="py-16">
          <div className="rounded-[32px] border border-blue-400/40 bg-gradient-to-r from-blue-600/40 to-purple-600/30 p-10 text-center shadow-2xl shadow-blue-500/30">
            <h2 className="text-3xl md:text-4xl font-black mb-4">Ready to hear your customers clearly?</h2>
            <p className="text-slate-100 text-lg mb-8">
              Secure, SOC2-ready infrastructure. Human-in-the-loop verification. Deployment in under a week.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 max-w-2xl mx-auto">
              <button
                onClick={primaryCta}
                className="flex-1 px-6 py-4 rounded-2xl bg-white text-slate-900 font-semibold"
              >
                {user?.id ? 'Open Dashboard' : 'Book a walkthrough'}
              </button>
              <button
                onClick={secondaryCta}
                className="flex-1 px-6 py-4 rounded-2xl border border-white/40 text-white font-semibold"
              >
                {user?.id ? 'Upload latest data' : 'Sign in instead'}
              </button>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/10 py-10 mt-10">
        <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-slate-400">
          <p>© {new Date().getFullYear()} ReviewSense. All rights reserved.</p>
          <div className="flex gap-6">
            <button onClick={() => handleScrollTo('features')} className="hover:text-white transition">Key Features</button>
            <button onClick={() => handleScrollTo('how-it-works')} className="hover:text-white transition">How it works</button>
            <button onClick={() => handleScrollTo('why-us')} className="hover:text-white transition">Why ReviewSense</button>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
