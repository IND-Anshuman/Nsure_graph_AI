import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import { Github, ArrowRight, Brain, Share2, ShieldCheck, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/card";
import { NsureLogo } from "@/components/NsureLogo";
import { KGPipelineAnimation } from "@/components/KGPipelineAnimation";

export function LandingPage() {
  return (
    <div className="min-h-screen relative overflow-hidden bg-background selection:bg-accent/40">
      <div className="paper-overlay" />
      <div className="relative z-10">
        <header className="container mx-auto px-6 py-8">
          <nav className="flex items-center justify-between border-b border-white/5 pb-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 flex items-center justify-center icon-glow">
                <NsureLogo className="w-full h-full text-primary" />
              </div>
              <span className="text-2xl font-serif font-bold tracking-tight text-foreground uppercase">Nsure AI</span>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="flex items-center gap-6"
            >
              <Button
                variant="ghost"
                className="gap-2 hover:text-accent transition-colors font-sans text-sm"
                onClick={() => window.open("https://github.com", "_blank")}
              >
                <Github className="w-4 h-4" />
                Documentation
              </Button>
              <Link to="/agent">
                <Button size="lg" className="gap-2 rounded-none border border-accent/50 bg-accent/10 text-accent hover:bg-accent hover:text-accent-foreground transition-all duration-500 font-serif px-8 shadow-[0_0_15px_rgba(212,175,55,0.1)]">
                  ACCESS SYSTEM
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </motion.div>
          </nav>
        </header>

        <main className="container mx-auto px-6 py-24">
          <div className="flex flex-col items-center text-center max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8"
            >
              <Badge variant="outline" className="px-4 py-1 border-accent text-accent font-serif tracking-widest uppercase text-[10px]">
                Institutional Intelligence System
              </Badge>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="text-4xl md:text-5xl font-serif font-bold mb-4 leading-tight text-foreground uppercase tracking-tighter"
            >
              The Architecture of <br />
              <span className="italic accent-underline text-accent">Knowledge</span>
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="space-y-4 mb-10 max-w-2xl mx-auto"
            >
              <p className="text-lg text-muted-foreground leading-relaxed font-serif italic selection:bg-accent/30 lowercase first-letter:uppercase">
                A prestigious autonomous GraphRAG environment for deep document understanding and authoritative knowledge extraction.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.6 }}
              className="mb-12"
            >
              <Link to="/agent">
                <Button size="lg" className="text-lg px-8 py-6 h-auto gap-3 rounded-none border border-accent bg-transparent text-accent hover:bg-accent hover:text-accent-foreground transition-all duration-500 font-serif shadow-[0_0_20px_rgba(212,175,55,0.15)]">
                  <Brain className="w-5 h-5 icon-glow" />
                  INITIATE RESEARCH
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
              className="w-full mb-24"
            >
              <div className="mb-12 flex flex-col items-center">
                <div className="h-px w-24 bg-accent/30 mb-8" />
                <h2 className="text-sm font-serif font-bold uppercase tracking-[0.3em] text-accent/60 italic underline-offset-8 underline decoration-accent/20">The Processing Pipeline</h2>
              </div>
              <KGPipelineAnimation />
            </motion.div>

            {/* Novelty & Trust Sections restorted */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-24 w-full"
            >
              <div className="flex flex-col items-center text-center space-y-4 group">
                <div className="w-16 h-16 rounded-full border border-white/5 bg-secondary/50 flex items-center justify-center mb-2 group-hover:border-accent/40 transition-colors icon-glow">
                  <Brain className="w-6 h-6 text-accent" />
                </div>
                <h3 className="text-2xl font-serif font-bold text-primary italic">Precision Retrieval</h3>
                <p className="text-xs text-muted-foreground leading-relaxed px-4 font-serif">
                  Extract answers from complex corporate and legal policies with unparalleled accuracy and institutional context.
                </p>
              </div>

              <div className="flex flex-col items-center text-center space-y-4 group">
                <div className="w-16 h-16 rounded-full border border-white/5 bg-secondary/50 flex items-center justify-center mb-2 group-hover:border-accent/40 transition-colors icon-glow">
                  <Share2 className="w-6 h-6 text-accent" />
                </div>
                <h3 className="text-2xl font-serif font-bold text-primary italic">Graph Synthesis</h3>
                <p className="text-xs text-muted-foreground leading-relaxed px-4 font-serif">
                  A hybrid GraphRAG architecture that preserves semantic relationships while eliminating artificial hallucinations.
                </p>
              </div>

              <div className="flex flex-col items-center text-center space-y-4 group">
                <div className="w-16 h-16 rounded-full border border-white/5 bg-secondary/50 flex items-center justify-center mb-2 group-hover:border-accent/40 transition-colors icon-glow">
                  <ShieldCheck className="w-6 h-6 text-accent" />
                </div>
                <h3 className="text-2xl font-serif font-bold text-primary italic">Verifiable Evidence</h3>
                <p className="text-xs text-muted-foreground leading-relaxed px-4 font-serif">
                  Every insight is cross-referenced with citations and evidence snippets for immediate auditability and trust.
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="w-full max-w-4xl"
            >
              <Card className="p-12 rounded-none border border-white/5 bg-secondary/20 realistic-shadow relative overflow-hidden backdrop-blur-md">
                <div className="absolute top-0 left-0 w-1 h-full bg-accent icon-glow" />
                <div className="flex items-center gap-3 mb-8 justify-center">
                  <Sparkles className="w-5 h-5 text-accent" />
                  <h3 className="text-2xl font-serif font-bold text-primary uppercase tracking-wider italic">Trust through Architecture</h3>
                </div>
                <div className="grid md:grid-cols-2 gap-12 text-left">
                  <div className="space-y-6">
                    <div className="flex items-start gap-4">
                      <div className="w-6 h-6 rounded-full border border-accent flex items-center justify-center flex-shrink-0 mt-1">
                        <span className="text-[10px] text-accent font-bold">I</span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Ingestion protocols that construct high-fidelity knowledge graphs of entities and regulatory relations.
                      </p>
                    </div>
                    <div className="flex items-start gap-4">
                      <div className="w-6 h-6 rounded-full border border-accent flex items-center justify-center flex-shrink-0 mt-1">
                        <span className="text-[10px] text-accent font-bold">II</span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Semantic indexing that reconciles complex queries with institutional policy hierarchies.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-6">
                    <div className="flex items-start gap-4">
                      <div className="w-6 h-6 rounded-full border border-accent flex items-center justify-center flex-shrink-0 mt-1">
                        <span className="text-[10px] text-accent font-bold">III</span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        GraphRAG reranking engine designed to preserve critical context and cross-document references.
                      </p>
                    </div>
                    <div className="flex items-start gap-4">
                      <div className="w-6 h-6 rounded-full border border-accent flex items-center justify-center flex-shrink-0 mt-1">
                        <span className="text-[10px] text-accent font-bold">IV</span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Synthesis of final intelligence with embedded citations for instant verification across the corpus.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </main>

        <footer className="container mx-auto px-6 py-12 border-t border-blue-900/20 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-serif italic">
            Nsure AI &bull; Intelligence Beyond Parameters &bull; Ver 1.0.4
          </p>
        </footer>
      </div>
    </div>
  );
}
