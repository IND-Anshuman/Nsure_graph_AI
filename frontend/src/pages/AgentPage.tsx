import { Link } from "react-router-dom";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Home, Settings, Network, Send, Sparkles, Cpu, Layers, Brain, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { FileUploadZone, FileList } from "@/components/FileUpload";
import { FloatingShapes } from "@/components/VibrantEffects";
import { queryFromFile, type Answer, type QueryResponse } from "@/lib/api";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file?: File;
}

interface DisplayAnswer {
  question: string;
  answer: string;
  extracted_facts?: Array<{
    fact: string;
    evidence_ids: string[];
  }>;
  insufficiency_note?: string;
  confidence?: string;
  used_evidence?: string[];
}

export function AgentPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [answers, setAnswers] = useState<DisplayAnswer[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [queryStats, setQueryStats] = useState<{
    nodes: number;
    edges: number;
    timing: { total: number; graph_build: number; index_build: number; qa_total: number };
  } | null>(null);

  const handleFilesUploaded = (newFiles: UploadedFile[]) => {
    setFiles((prev) => [...prev, ...newFiles]);
  };

  const handleRemoveFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please enter a question");
      return;
    }

    if (files.length === 0) {
      setError("Please upload at least one document");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setAnswers([]);

    try {
      // Use the first uploaded file (you can modify to handle multiple files)
      const firstFile = files.find(f => f.file);
      if (!firstFile?.file) {
        throw new Error("No valid file found");
      }

      const response: QueryResponse = await queryFromFile(
        firstFile.file,
        query,
        {
          build: {
            skip_neo4j: true,
          },
          qa: {
            top_n_semantic: 20,
            top_k_final: 40,
            rerank_top_k: 12,
          },
          cache: {
            enabled: true,
            rebuild_graph: false, // Use cached graph if available
            rebuild_index: false, // Use cached index if available
          },
          verbose: true,
        }
      );

      // Transform answers to flatten the result structure
      const transformedAnswers = response.answers.map(ans => ({
        question: ans.question,
        answer: ans.result.answer,
        extracted_facts: ans.result.extracted_facts,
        insufficiency_note: ans.result.insufficiency_note,
        confidence: ans.result.confidence,
        used_evidence: ans.result.used_evidence,
      }));

      setAnswers(transformedAnswers);
      setQueryStats({
        nodes: response.counts.nodes,
        edges: response.counts.edges,
        timing: response.timing_s,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process query");
      console.error("Query error:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0f1729] to-[#0a0e1a] relative">
      <div className="absolute inset-0 animated-grid opacity-20 pointer-events-none" />
      <FloatingShapes />
      
      <motion.div
        className="absolute top-20 left-10 w-64 h-64 bg-gradient-to-br from-violet-500/10 to-transparent rounded-full blur-3xl"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      <motion.div
        className="absolute bottom-20 right-10 w-96 h-96 bg-gradient-to-br from-cyan-500/10 to-transparent rounded-full blur-3xl"
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{ duration: 10, repeat: Infinity, delay: 1 }}
      />
      
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/30 sticky top-0 z-50 glass-effect relative">
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent"
          animate={{
            opacity: [0.3, 1, 0.3],
            scaleX: [0.8, 1, 0.8],
          }}
          transition={{ duration: 3, repeat: Infinity }}
        />
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-3"
              >
                <motion.div 
                  className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-emerald-600 flex items-center justify-center shadow-lg relative overflow-hidden"
                  whileHover={{ rotate: 180, scale: 1.1 }}
                  transition={{ duration: 0.5 }}
                >
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-br from-violet-500 to-fuchsia-500"
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  />
                  <Network className="w-5 h-5 text-background relative z-10" />
                  <motion.div
                    className="absolute inset-0"
                    animate={{
                      boxShadow: [
                        "0 0 15px rgba(94, 234, 212, 0.4)",
                        "0 0 25px rgba(139, 92, 246, 0.4)",
                        "0 0 15px rgba(94, 234, 212, 0.4)",
                      ],
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.div>
                <span className="text-lg font-bold tracking-tight">Nsure AI</span>
              </motion.div>
              <motion.div
                animate={{
                  boxShadow: [
                    "0 0 0 0 rgba(94, 234, 212, 0)",
                    "0 0 0 4px rgba(94, 234, 212, 0.1)",
                    "0 0 0 0 rgba(94, 234, 212, 0)",
                  ],
                }}
                transition={{ duration: 2, repeat: Infinity }}
                className="rounded-full"
              >
                <Badge variant="outline" className="text-xs glass-effect">
                  <Cpu className="w-3 h-3 mr-1" />
                  GraphRAG Knowledge System
                </Badge>
              </motion.div>
            </div>

            <div className="flex items-center gap-2">
              <Link to="/">
                <Button variant="ghost" size="icon" className="hover:text-primary transition-colors">
                  <Home className="w-4 h-4" />
                </Button>
              </Link>
              <Button variant="ghost" size="icon" className="hover:text-primary transition-colors">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 relative">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="relative">
                <motion.div
                  className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 via-violet-500 to-fuchsia-500 rounded-xl opacity-20 blur"
                  animate={{
                    opacity: [0.2, 0.4, 0.2],
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                />
                <Card className="p-8 glass-effect border-2 border-primary/20 hover:border-primary/40 transition-all relative backdrop-blur-xl bg-secondary/30">
                  <div className="flex items-start gap-4 mb-6">
                    <motion.div
                      className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 flex items-center justify-center border border-cyan-500/30"
                      whileHover={{ rotate: 90, scale: 1.1 }}
                      transition={{ type: "spring", stiffness: 200 }}
                    >
                      <Layers className="w-6 h-6 text-cyan-400" />
                    </motion.div>
                    <div className="flex-1">
                      <h2 className="text-2xl font-bold mb-2 bg-gradient-to-r from-foreground to-foreground/60 bg-clip-text">
                        Document Upload
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        Drag & drop your files or browse to begin analysis
                      </p>
                    </div>
                  </div>
                  <FileUploadZone onFilesUploaded={handleFilesUploaded} />
                </Card>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
            >
              <div className="relative">
                <motion.div
                  className="absolute -inset-1 bg-gradient-to-r from-violet-600 via-fuchsia-600 to-pink-600 rounded-2xl opacity-20 blur-xl"
                  animate={{
                    rotate: [0, 180, 360],
                    scale: [1, 1.1, 1],
                  }}
                  transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                />
                <Card className="glass-effect border-2 border-violet-500/20 hover:border-violet-500/40 transition-all relative backdrop-blur-xl bg-secondary/30 overflow-hidden">
                  <div className="grid md:grid-cols-3 gap-6 p-8">
                    <div className="md:col-span-1 flex flex-col items-center justify-center border-r border-border/50 pr-6">
                      <motion.div
                        className="relative mb-4"
                        animate={{
                          y: [0, -10, 0],
                        }}
                        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                      >
                        <div className="w-28 h-28 rounded-3xl bg-gradient-to-br from-violet-500/30 via-fuchsia-500/30 to-pink-500/30 flex items-center justify-center border-2 border-violet-500/40 relative">
                          <motion.div
                            className="absolute inset-0 rounded-3xl bg-gradient-to-br from-cyan-500/20 to-transparent"
                            animate={{ opacity: [0.3, 0.7, 0.3] }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                          <Brain className="w-14 h-14 text-violet-400 relative z-10" />
                        </div>
                        <motion.div
                          className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full bg-emerald-500/80 border-2 border-background flex items-center justify-center"
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 2, repeat: Infinity }}
                        >
                          <Sparkles className="w-4 h-4 text-white" />
                        </motion.div>
                      </motion.div>
                      
                      <h3 className="text-xl font-bold mb-2 text-center">Nsure AI</h3>
                      <Badge className="glass-effect border border-violet-500/30 bg-violet-500/10">
                        <Cpu className="w-3 h-3 mr-1" />
                        GraphRAG
                      </Badge>
                    </div>

                    <div className="md:col-span-2 flex flex-col">
                      <div className="mb-6">
                        <h4 className="text-3xl font-bold mb-3">
                          <span className="bg-gradient-to-r from-violet-400 via-fuchsia-400 to-pink-400 bg-clip-text text-transparent">
                            Ask Anything
                          </span>
                        </h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          Advanced knowledge extraction using graph-based reasoning. Query complex documents with contextual understanding.
                        </p>
                      </div>

                      <div className="flex-1">
                        <div className="relative">
                          <Textarea
                            placeholder="Type your question here..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="min-h-[160px] text-base bg-background/50 border-2 border-border/50 focus:border-violet-500/50 rounded-xl resize-none transition-all"
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit();
                              }
                            }}
                          />
                          <div className="flex items-center justify-between mt-3">
                            <div className="flex gap-2">
                              <motion.div
                                className="px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-xs text-violet-400"
                                whileHover={{ scale: 1.05, borderColor: "rgba(139, 92, 246, 0.4)" }}
                              >
                                {query.length} chars
                              </motion.div>
                            </div>
                            <AnimatePresence>
                              {query.trim() && (
                                <motion.div
                                  initial={{ opacity: 0, x: 20 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  exit={{ opacity: 0, x: 20 }}
                                >
                                  <Button
                                    onClick={handleSubmit}
                                    disabled={isSubmitting}
                                    className="gap-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500"
                                  >
                                    {isSubmitting ? (
                                      <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                      >
                                        <Cpu className="w-4 h-4" />
                                      </motion.div>
                                    ) : (
                                      <>
                                        <span>Submit Query</span>
                                        <Send className="w-4 h-4" />
                                      </>
                                    )}
                                  </Button>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            </motion.div>

            {/* Loading State */}
            {isSubmitting && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <Card className="p-8 glass-effect border-2 border-violet-500/30 bg-secondary/30 overflow-hidden relative">
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-violet-500/10 via-fuchsia-500/10 to-cyan-500/10"
                    animate={{
                      x: ["-100%", "100%"],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  />
                  <div className="relative z-10">
                    <div className="flex items-center justify-center gap-4 mb-6">
                      <motion.div
                        animate={{
                          rotate: 360,
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: "linear",
                        }}
                      >
                        <Network className="w-8 h-8 text-violet-400" />
                      </motion.div>
                      <h3 className="text-xl font-bold bg-gradient-to-r from-violet-400 via-fuchsia-400 to-cyan-400 bg-clip-text text-transparent">
                        Building Knowledge Graph
                      </h3>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="flex items-center gap-3">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-emerald-500"
                          animate={{
                            scale: [1, 1.5, 1],
                            opacity: [1, 0.5, 1],
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                          }}
                        />
                        <span className="text-sm text-muted-foreground">Extracting document content...</span>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-cyan-500"
                          animate={{
                            scale: [1, 1.5, 1],
                            opacity: [1, 0.5, 1],
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: 0.3,
                          }}
                        />
                        <span className="text-sm text-muted-foreground">Identifying entities and relationships...</span>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-violet-500"
                          animate={{
                            scale: [1, 1.5, 1],
                            opacity: [1, 0.5, 1],
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: 0.6,
                          }}
                        />
                        <span className="text-sm text-muted-foreground">Building semantic index...</span>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-fuchsia-500"
                          animate={{
                            scale: [1, 1.5, 1],
                            opacity: [1, 0.5, 1],
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: 0.9,
                          }}
                        />
                        <span className="text-sm text-muted-foreground">Generating intelligent answer...</span>
                      </div>
                    </div>

                    <div className="mt-6 p-4 rounded-lg bg-violet-500/10 border border-violet-500/30">
                      <p className="text-xs text-center text-violet-400">
                        This may take 30-60 seconds depending on document size
                      </p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Error Display */}
            {error && !isSubmitting && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <Card className="p-6 glass-effect border-2 border-red-500/30 bg-red-500/5">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-red-400 mb-1">Error</h3>
                      <p className="text-sm text-muted-foreground">{error}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Answer Display */}
            {answers.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Stats Card */}
                {queryStats && (
                  <Card className="p-6 glass-effect border-2 border-emerald-500/20 bg-secondary/30">
                    <div className="flex items-center gap-2 mb-4">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <h3 className="font-bold text-lg">Query Completed</h3>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Nodes</div>
                        <div className="text-xl font-bold text-emerald-400">{queryStats.nodes}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Edges</div>
                        <div className="text-xl font-bold text-cyan-400">{queryStats.edges}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Build Time</div>
                        <div className="text-xl font-bold text-violet-400">{queryStats.timing.graph_build.toFixed(2)}s</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Total Time</div>
                        <div className="text-xl font-bold text-fuchsia-400">{queryStats.timing.total.toFixed(2)}s</div>
                      </div>
                    </div>
                  </Card>
                )}

                {/* Answers */}
                {answers.map((answer, idx) => (
                  <Card key={idx} className="p-6 glass-effect border-2 border-primary/20 bg-secondary/30">
                    <div className="space-y-4">
                      <div>
                        <div className="text-sm text-muted-foreground mb-2">Question</div>
                        <div className="text-lg font-semibold">{answer.question}</div>
                      </div>
                      
                      <div>
                        <div className="text-sm text-muted-foreground mb-2">Answer</div>
                        <div className="prose prose-invert max-w-none">
                          <p className="text-foreground leading-relaxed whitespace-pre-wrap">{answer.answer}</p>
                        </div>
                      </div>

                      {answer.confidence && (
                        <div className="flex items-center gap-2">
                          <div className="text-sm text-muted-foreground">Confidence:</div>
                          <Badge className={`${
                            answer.confidence === 'high' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                            answer.confidence === 'medium' ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400' :
                            'bg-red-500/10 border-red-500/30 text-red-400'
                          }`}>
                            {answer.confidence.toUpperCase()}
                          </Badge>
                        </div>
                      )}

                      {answer.extracted_facts && answer.extracted_facts.length > 0 && (
                        <div>
                          <div className="text-sm text-muted-foreground mb-2">Key Facts</div>
                          <ul className="space-y-2">
                            {answer.extracted_facts.map((fact, factIdx) => (
                              <li key={factIdx} className="text-sm flex gap-2">
                                <span className="text-cyan-400 flex-shrink-0">•</span>
                                <span className="text-foreground/80">{fact.fact}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {answer.insufficiency_note && (
                        <div className="mt-4 p-4 rounded-lg bg-amber-500/10 border border-amber-500/30">
                          <p className="text-sm text-amber-400">{answer.insufficiency_note}</p>
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </motion.div>
            )}
          </div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="lg:col-span-1"
          >
            <div className="sticky top-24 space-y-4">
              <Card className="p-6 glass-effect border-2 border-emerald-500/20 hover:border-emerald-500/40 transition-all backdrop-blur-xl bg-secondary/30">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center border border-emerald-500/30">
                      <Layers className="w-4 h-4 text-emerald-400" />
                    </div>
                    Sources
                  </h3>
                  <Badge className="bg-emerald-500/10 border-emerald-500/30 text-emerald-400">
                    {files.length}
                  </Badge>
                </div>
                <FileList files={files} onRemove={handleRemoveFile} />
              </Card>
              
              <Card className="p-5 glass-effect border border-border/50 backdrop-blur-xl bg-secondary/20">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-sm font-medium">System Status</span>
                </div>
                <div className="space-y-2 text-xs text-muted-foreground">
                  <div className="flex justify-between">
                    <span>GraphRAG</span>
                    <span className="text-emerald-400">Active</span>
                  </div>
                  <div className="flex justify-between">
                    <span>LLM Model</span>
                    <span className="text-cyan-400">Online</span>
                  </div>
                </div>
              </Card>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
