import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Github, ArrowRight, Sparkles, Zap, Brain, Search, Layers, FileText, Link2, CheckCircle } from "lucide-react";
import { BackgroundParticles } from "@/components/BackgroundParticles";
import { FloatingShapes, AnimatedWave, OrbitingElements } from "@/components/VibrantEffects";
import { NsureLogo } from "@/components/NsureLogo";

export function LandingPage() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-[#0f1729] via-[#1a1f35] to-[#0a0e1a]" />
      <div className="absolute inset-0 animated-grid opacity-30" />
      <FloatingShapes />
      <BackgroundParticles />
      <AnimatedWave color="rgba(94, 234, 212, 0.05)" delay={0} />
      <AnimatedWave color="rgba(139, 92, 246, 0.05)" delay={0.5} />

      <div className="relative z-10">
        <header className="container mx-auto px-6 py-6">
          <nav className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center gap-3"
            >
              <motion.div 
                className="w-12 h-12 rounded-lg flex items-center justify-center shadow-lg relative overflow-hidden"
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
              >
                <NsureLogo className="w-full h-full relative z-10" />
                <motion.div
                  className="absolute inset-0 rounded-lg"
                  animate={{
                    boxShadow: [
                      "0 0 20px rgba(94, 234, 212, 0.5)",
                      "0 0 30px rgba(139, 92, 246, 0.5)",
                      "0 0 20px rgba(94, 234, 212, 0.5)",
                    ],
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
              <span className="text-xl font-bold tracking-tight">Nsure AI</span>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="flex items-center gap-4"
            >
              <Button
                variant="ghost"
                className="gap-2 hover:text-primary transition-colors"
                onClick={() =>
                  window.open("https://github.com", "_blank")
                }
              >
                <Github className="w-4 h-4" />
                Star on GitHub
              </Button>
              <Link to="/agent">
                <Button size="lg" className="gap-2 glow-effect">
                  Launch System
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </motion.div>
          </nav>
        </header>

        <main className="container mx-auto px-6 py-20">
          <div className="flex flex-col items-center text-center max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Badge variant="success" className="mb-8 shimmer neon-border">
                <motion.div
                  className="flex items-center gap-2"
                  animate={{ opacity: [1, 0.7, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <Sparkles className="w-3 h-3" />
                  ● SYSTEM ACTIVE · v1.0.4 PRODUCTION
                  <Zap className="w-3 h-3" />
                </motion.div>
              </Badge>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight relative"
            >
              <motion.span
                animate={{ 
                  textShadow: [
                    "0 0 20px rgba(94, 234, 212, 0.5)",
                    "0 0 40px rgba(139, 92, 246, 0.5)",
                    "0 0 20px rgba(94, 234, 212, 0.5)",
                  ]
                }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                Intelligent
              </motion.span>
              <br />
              <span className="gradient-text relative inline-block">
                Graph Knowledge
                <motion.div
                  className="absolute -inset-2 bg-gradient-to-r from-cyan-400/20 via-violet-400/20 to-fuchsia-400/20 blur-2xl -z-10"
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0.8, 0.5],
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                />
              </span>
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="space-y-4 mb-12 max-w-3xl"
            >
              <p className="text-xl text-muted-foreground leading-relaxed">
                <span className="font-semibold text-foreground">Nsure AI</span> is an advanced
              </p>
              <p className="text-xl text-muted-foreground leading-relaxed">
                autonomous GraphRAG system for deep document understanding and knowledge extraction.
              </p>
              <p className="text-xl text-muted-foreground leading-relaxed">
                Combining graph-based reasoning with cutting-edge language models.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.6 }}
              className="mb-16 relative"
            >
              <OrbitingElements />
              <Link to="/agent">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button size="lg" className="text-lg px-10 py-6 h-auto gap-3 group relative overflow-hidden">
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-violet-600 via-fuchsia-600 to-pink-600"
                      initial={{ x: "-100%" }}
                      whileHover={{ x: "100%" }}
                      transition={{ duration: 0.6 }}
                    />
                    <Brain className="w-5 h-5 relative z-10" />
                    <span className="relative z-10">Launch Knowledge System</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform relative z-10" />
                  </Button>
                </motion.div>
              </Link>
            </motion.div>

            {/* Novelty & Benefits for Policy Documents */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16"
            >
              <Card className="p-6 glass-effect border-2 border-cyan-500/20 bg-secondary/30">
                <div className="flex items-center gap-3 mb-3">
                  <Search className="w-5 h-5 text-cyan-400" />
                  <h3 className="font-semibold">Get Answers From Policies</h3>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Ask natural-language questions and receive precise answers sourced from large policy documents, without skimming hundreds of pages. The system understands policy language, sections, and definitions.
                </p>
              </Card>

              <Card className="p-6 glass-effect border-2 border-violet-500/20 bg-secondary/30">
                <div className="flex items-center gap-3 mb-3">
                  <Layers className="w-5 h-5 text-violet-400" />
                  <h3 className="font-semibold">GraphRAG Novelty</h3>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  We combine semantic search with a knowledge graph of entities and relationships. This hybrid retrieval (GraphRAG) preserves context, reduces hallucinations, and surfaces relevant clauses and cross-references.
                </p>
              </Card>

              <Card className="p-6 glass-effect border-2 border-fuchsia-500/20 bg-secondary/30">
                <div className="flex items-center gap-3 mb-3">
                  <FileText className="w-5 h-5 text-fuchsia-400" />
                  <h3 className="font-semibold">Traceable Evidence</h3>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Every answer includes evidence snippets and citations, so users can verify the source in the policy document. Confidence labels and insufficiency notes improve trust and auditability.
                </p>
              </Card>
            </motion.div>

            {/* How It Works */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="mb-16"
            >
              <Card className="p-8 glass-effect border-2 border-primary/20 bg-secondary/30">
                <div className="flex items-center gap-2 mb-6">
                  <Link2 className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-bold">How Nsure AI extracts answers from large policy documents</h3>
                </div>
                <div className="grid md:grid-cols-2 gap-6 text-sm">
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                      <p className="text-muted-foreground">
                        Ingests documents and builds a knowledge graph of entities, definitions, and relations (sections, clauses, references).
                      </p>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                      <p className="text-muted-foreground">
                        Creates a semantic index to understand queries beyond keywords, aligning them to relevant policy context.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                      <p className="text-muted-foreground">
                        Reranks candidates with GraphRAG to preserve relationships and reduce off-topic results.
                      </p>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                      <p className="text-muted-foreground">
                        Synthesizes a final answer with citations, confidence, and evidence snippets for instant verification.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="flex items-center gap-6 text-sm text-muted-foreground flex-wrap justify-center"
            >
              <motion.div 
                className="flex items-center gap-2"
                whileHover={{ scale: 1.1, color: "rgb(94, 234, 212)" }}
              >
                <motion.div 
                  className="w-2 h-2 rounded-full bg-primary"
                  animate={{ 
                    scale: [1, 1.5, 1],
                    boxShadow: [
                      "0 0 0px rgba(94, 234, 212, 0)",
                      "0 0 10px rgba(94, 234, 212, 0.8)",
                      "0 0 0px rgba(94, 234, 212, 0)",
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <span>React 18</span>
              </motion.div>
              <div className="w-1 h-1 rounded-full bg-border" />
              <motion.div 
                className="flex items-center gap-2"
                whileHover={{ scale: 1.1, color: "rgb(16, 185, 129)" }}
              >
                <motion.div 
                  className="w-2 h-2 rounded-full bg-emerald-400"
                  animate={{ 
                    scale: [1, 1.5, 1],
                    boxShadow: [
                      "0 0 0px rgba(16, 185, 129, 0)",
                      "0 0 10px rgba(16, 185, 129, 0.8)",
                      "0 0 0px rgba(16, 185, 129, 0)",
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity, delay: 0.3 }}
                />
                <span>GraphRAG</span>
              </motion.div>
              <div className="w-1 h-1 rounded-full bg-border" />
              <motion.div 
                className="flex items-center gap-2"
                whileHover={{ scale: 1.1, color: "rgb(34, 211, 238)" }}
              >
                <motion.div 
                  className="w-2 h-2 rounded-full bg-cyan-400"
                  animate={{ 
                    scale: [1, 1.5, 1],
                    boxShadow: [
                      "0 0 0px rgba(34, 211, 238, 0)",
                      "0 0 10px rgba(34, 211, 238, 0.8)",
                      "0 0 0px rgba(34, 211, 238, 0)",
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity, delay: 0.6 }}
                />
                <span>LLM Powered</span>
              </motion.div>
            </motion.div>
          </div>
        </main>
      </div>
    </div>
  );
}
