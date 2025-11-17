
import React from 'react';
import { useNavigate } from 'react-router-dom';

type StoredUser = {
  id?: string | number;
  email?: string;
};

const features = [
  {
    title: 'AI Sentiment Precision',
    description: 'Detect nuanced sentiment in Arabic and English at scale with hybrid LLM + custom models.',
    highlight: '94%+ accuracy across dialects',
  },
  {
    title: 'Instant Topic Surfacing',
    description: 'Group thousands of reviews into actionable product, service, and CX themes.',
    highlight: 'Auto-tagged insights in seconds',
  },
  {
    title: 'Team-ready Reporting',
    description: 'Share branded summaries, download CSVs, or push results into BI tools.',
    highlight: 'One-click export',
  },
];

const stats = [
  { label: 'Reviews processed', value: '4.7M+' },
  { label: 'Customer teams', value: '120+' },
  { label: 'Average time saved', value: '18 hrs/week' },
];

const testimonials = [
  {
    quote: 'ReviewSense helps our CX team react to customer feedback before it becomes a headline. The clarity is unmatched.',
    name: 'Layla Rahman',
    role: 'Head of Service, GulfAir',
  },
  {
    quote: 'Uploading messy spreadsheets and getting board-ready insights in minutes feels like cheating—in the best way.',
    name: 'Omar Khalid',
    role: 'Product Lead, FinEdge',
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
            <button onClick={() => handleScrollTo('features')} className="hover:text-white transition">Features</button>
            <button onClick={() => handleScrollTo('proof')} className="hover:text-white transition">Why ReviewSense</button>
            <button onClick={() => handleScrollTo('testimonials')} className="hover:text-white transition">Stories</button>
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
                  onClick={() => navigate('/login')}
                  className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-sm font-semibold"
                >
                  Sign In
                </button>
                <button
                  onClick={() => navigate('/signup')}
                  className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-sm font-semibold"
                >
                  Start for Free
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
              ReviewSense transforms raw customer feedback into prioritized sentiment, topics, and
              next-best actions—so product, CX, and growth teams can ship what people actually want.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 mb-8">
              <button
                onClick={primaryCta}
                className="flex-1 px-6 py-4 rounded-xl bg-blue-500 hover:bg-blue-600 text-base font-semibold shadow-lg shadow-blue-500/30"
              >
                {user?.id ? 'Go to Dashboard' : 'Create a free account'}
              </button>
              <button
                onClick={secondaryCta}
                className="flex-1 px-6 py-4 rounded-xl border border-white/30 hover:border-white/60 text-base font-semibold"
              >
                {user?.id ? 'Upload data' : 'Sign in'}
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

        <section id="proof" className="py-12">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            {stats.map((stat) => (
              <div key={stat.label} className="rounded-2xl border border-white/10 bg-white/5 px-6 py-8 text-center">
                <p className="text-3xl font-black text-white">{stat.value}</p>
                <p className="text-sm uppercase tracking-wide text-slate-400 mt-2">{stat.label}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="features" className="py-16">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-10 gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-blue-300">Why teams choose us</p>
              <h2 className="text-3xl md:text-4xl font-black mt-2">Landing insights, not dashboards.</h2>
            </div>
            <button
              onClick={primaryCta}
              className="px-6 py-3 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-sm font-semibold w-full md:w-auto"
            >
              See ReviewSense in action
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature) => (
              <div key={feature.title} className="rounded-3xl border border-white/10 bg-gradient-to-b from-white/10 to-white/5 p-8 shadow-lg shadow-black/20">
                <p className="text-xs uppercase tracking-[0.3em] text-blue-200">{feature.highlight}</p>
                <h3 className="text-2xl font-bold mt-4 mb-3">{feature.title}</h3>
                <p className="text-slate-200 text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="testimonials" className="py-16">
          <div className="rounded-[32px] border border-white/10 bg-white/5 p-10">
            <p className="text-xs uppercase tracking-[0.3em] text-blue-200 mb-8">Loved by CX, Product & Ops</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              {testimonials.map((testimonial) => (
                <div key={testimonial.name} className="bg-slate-900/40 rounded-3xl p-6 border border-white/5">
                  <p className="text-lg text-slate-50 mb-6">“{testimonial.quote}”</p>
                  <div>
                    <p className="font-semibold text-white">{testimonial.name}</p>
                    <p className="text-sm text-slate-400">{testimonial.role}</p>
                  </div>
                </div>
              ))}
            </div>
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
            <button onClick={() => handleScrollTo('features')} className="hover:text-white transition">Features</button>
            <button onClick={() => handleScrollTo('proof')} className="hover:text-white transition">Customers</button>
            <button onClick={() => navigate('/signup')} className="hover:text-white transition">Get Started</button>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
