import { Link } from "react-router-dom";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Home, Settings, Network, Send, AlertCircle, Search, BookOpen, Quote, Layers, Cpu } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { FileUploadZone, FileList } from "@/components/FileUpload";
import { NsureLogo } from "@/components/NsureLogo";
import { cn } from "@/lib/utils";
import { queryFromFile, queryFromUrl, type QueryResponse } from "@/lib/api";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file?: File;
}

interface DisplayAnswer {
  question: string;
  answer: unknown;
  extracted_facts?: Array<{
    fact: string;
    evidence_ids: string[];
  }>;
  insufficiency_note?: string;
  confidence?: string;
  evidence_texts?: string[];
}

function renderAnswerText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function AgentPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [answers, setAnswers] = useState<DisplayAnswer[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isUrlMode, setIsUrlMode] = useState(false);
  const [queryStats, setQueryStats] = useState<{
    nodes: number;
    edges: number;
    timing: { total: number; graph_build: number; index_build: number; qa_total: number };
  } | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [rebuildGraph, setRebuildGraph] = useState(false);
  const [envOverrides, setEnvOverrides] = useState({
    KG_DOC_WORKERS: "4",
    KG_QA_WORKERS: "4",
    KG_RELATION_WORKERS: "4",
    KG_COMMUNITY_SUMMARY_WORKERS: "4",
    KG_RELATION_BATCH_SIZE: "10",
    KG_TOP_N_SEMANTIC: "60",
    KG_TOP_K_FINAL: "60",
    KG_RERANK_TOP_K: "25",
    GEMINI_MODEL: "gemini-2.0-flash",
    KG_EXTRACTION_STRATEGY: "hybrid",
    KG_EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2",
    KG_EMBEDDING_DIM: "384",
  });

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

    if (files.length === 0 && (!isUrlMode || !url.trim())) {
      setError("Please upload at least one document or provide a URL");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setAnswers([]);

    try {
      let response: QueryResponse;

      const apiOptions = {
        build: { skip_neo4j: true },
        qa: {
          top_n_semantic: Number(envOverrides.KG_TOP_N_SEMANTIC) || Number(import.meta.env.VITE_TOP_N_SEMANTIC) || 60,
          top_k_final: Number(envOverrides.KG_TOP_K_FINAL) || Number(import.meta.env.VITE_TOP_K_FINAL) || 60,
          rerank_top_k: Number(envOverrides.KG_RERANK_TOP_K) || Number(import.meta.env.VITE_RERANK_TOP_K) || 30,
        },
        cache: {
          enabled: true,
          rebuild_graph: rebuildGraph,
          rebuild_index: rebuildGraph,
        },
        env_overrides: envOverrides,
        verbose: true,
      };

      if (isUrlMode && url.trim()) {
        response = await queryFromUrl(url.trim(), query, apiOptions);
      } else {
        const firstFile = files.find(f => f.file);
        if (!firstFile?.file) {
          throw new Error("No valid file found");
        }
        response = await queryFromFile(firstFile.file, query, apiOptions);
      }

      const transformedAnswers = response.answers.map(ans => ({
        question: ans.question,
        answer: ans.result.answer,
        extracted_facts: ans.result.extracted_facts,
        insufficiency_note: ans.result.insufficiency_note,
        confidence: ans.result.confidence,
        evidence_texts: ans.result.evidence_texts || ans.result.used_evidence,
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
    <div className="min-h-screen bg-background text-foreground selection:bg-accent/40 relative">
      <div className="paper-overlay" />
      <header className="border-b border-white/5 bg-background/95 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link to="/" className="flex items-center gap-3 group">
                <div className="w-10 h-10 flex items-center justify-center icon-glow">
                  <NsureLogo className="w-full h-full text-primary" />
                </div>
                <span className="text-xl font-serif font-bold tracking-tight text-foreground uppercase group-hover:text-accent transition-colors">NSURE AI</span>
              </Link>
              <div className="h-4 w-px bg-white/5 mx-2" />
              <Badge variant="outline" className="rounded-none border-accent/30 bg-accent/5 text-accent font-serif italic py-1 px-3">
                Research Workbench v1.0.4
              </Badge>
            </div>

            <div className="flex items-center gap-3">
              <Link to="/">
                <Button variant="ghost" size="sm" className="gap-2 font-serif uppercase tracking-widest text-[10px] hover:text-accent">
                  <Home className="w-3.5 h-3.5" />
                  Terminal
                </Button>
              </Link>
              <Button
                variant="ghost"
                size="sm"
                className={cn("gap-2 font-serif uppercase tracking-widest text-[10px] hover:text-accent", showAdvanced && "text-accent bg-accent/10")}
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <Settings className="w-3.5 h-3.5" />
                Advanced
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          {/* Main Sidebar (Upload & Status) */}
          <div className="lg:col-span-3 space-y-8">
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <BookOpen className="w-4 h-4 text-accent" />
                <h2 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Library & Sources</h2>
              </div>
              <Card className="rounded-none border border-white/5 bg-secondary/60 shadow-2xl p-4 backdrop-blur-md relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-1 h-0 bg-accent group-hover:h-full transition-all duration-500" />
                <div className="flex gap-2 mb-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsUrlMode(false)}
                    className={cn(
                      "flex-1 rounded-none font-serif uppercase tracking-widest text-[10px]",
                      !isUrlMode ? "bg-accent/10 text-accent" : "text-muted-foreground"
                    )}
                  >
                    Upload Docs
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsUrlMode(true)}
                    className={cn(
                      "flex-1 rounded-none font-serif uppercase tracking-widest text-[10px]",
                      isUrlMode ? "bg-accent/10 text-accent" : "text-muted-foreground"
                    )}
                  >
                    URL Ingest
                  </Button>
                </div>

                {isUrlMode ? (
                  <div className="space-y-4 py-4">
                    <input
                      type="url"
                      placeholder="https://example.com/policy.pdf"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      className="w-full bg-background/50 border border-white/10 rounded-none p-3 text-xs font-serif italic focus:border-accent outline-none transition-colors"
                    />
                    <p className="text-[9px] text-muted-foreground uppercase tracking-widest italic text-center">
                      Supports Web Pages & Remote PDFs
                    </p>
                  </div>
                ) : (
                  <>
                    <FileUploadZone onFilesUploaded={handleFilesUploaded} />
                    <div className="mt-4 pt-4 border-t border-white/5">
                      <FileList files={files} onRemove={handleRemoveFile} />
                    </div>
                  </>
                )}
              </Card>
            </div>

            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="space-y-6 mb-2">
                    <div className="flex items-center gap-2 mb-4">
                      <Settings className="w-4 h-4 text-accent" />
                      <h2 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Engine Customization</h2>
                    </div>

                    <Card className="rounded-none border border-accent/20 bg-secondary/80 p-6 space-y-6 backdrop-blur-md relative overflow-hidden">
                      {/* Build Settings Section */}
                      <div className="space-y-4">
                        <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                          <Layers className="w-3.5 h-3.5 text-accent/60" />
                          <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold">Build & Extraction</span>
                        </div>

                        <div className="flex items-center justify-between group/toggle">
                          <div className="flex flex-col gap-1">
                            <span className="text-[10px] font-serif uppercase tracking-wider text-primary">Force Rebuild Graph</span>
                            <span className="text-[8px] text-muted-foreground uppercase tracking-widest leading-tight max-w-[180px]">
                              Discards cached knowledge and reconstructs from scratch.
                            </span>
                          </div>
                          <input
                            type="checkbox"
                            checked={rebuildGraph}
                            onChange={(e) => setRebuildGraph(e.target.checked)}
                            className="w-4 h-4 accent-accent cursor-pointer"
                          />
                        </div>

                        <div className="flex items-center justify-between group/toggle">
                        </div>

                        <div className="space-y-3 pt-2">
                          <div className="flex justify-between items-center">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Embedding Intelligence
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Fast (MiniLM) for CPU, High-Quality (MPNet) for GPU</span>
                            </label>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setEnvOverrides(prev => ({
                                ...prev,
                                KG_EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2",
                                KG_EMBEDDING_DIM: "384"
                              }))}
                              className={cn(
                                "rounded-none text-[8px] h-7 border-white/10 uppercase tracking-widest font-serif",
                                envOverrides.KG_EMBEDDING_DIM === "384" && "border-accent bg-accent/10 text-accent"
                              )}
                            >
                              High Speed
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setEnvOverrides(prev => ({
                                ...prev,
                                KG_EMBEDDING_MODEL: "sentence-transformers/all-mpnet-base-v2",
                                KG_EMBEDDING_DIM: "768"
                              }))}
                              className={cn(
                                "rounded-none text-[8px] h-7 border-white/10 uppercase tracking-widest font-serif",
                                envOverrides.KG_EMBEDDING_DIM === "768" && "border-accent bg-accent/10 text-accent"
                              )}
                            >
                              Max Context
                            </Button>
                          </div>
                        </div>
                      </div>

                      {/* Worker Settings Section */}
                      <div className="space-y-4 pt-2">
                        <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                          <Cpu className="w-3.5 h-3.5 text-accent/60" />
                          <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold">Parallel Processing</span>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Doc Workers
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Parallel doc ingestion (1-16)</span>
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="16"
                              value={envOverrides.KG_DOC_WORKERS}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_DOC_WORKERS: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              QA Workers
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Parallel query handling (1-16)</span>
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="16"
                              value={envOverrides.KG_QA_WORKERS}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_QA_WORKERS: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className={cn("space-y-2")}>
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Rel Workers
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Relation identification (1-16)</span>
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="16"
                              value={envOverrides.KG_RELATION_WORKERS}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_RELATION_WORKERS: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                          <div className={cn("space-y-2")}>
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Rel Batch Size
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Items per API call (1-100)</span>
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="100"
                              value={envOverrides.KG_RELATION_BATCH_SIZE}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_RELATION_BATCH_SIZE: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Comm Workers
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Community summarization (1-16)</span>
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="16"
                              value={envOverrides.KG_COMMUNITY_SUMMARY_WORKERS}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_COMMUNITY_SUMMARY_WORKERS: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Retrieval Settings Section */}
                      <div className="space-y-4 pt-2">
                        <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                          <Search className="w-3.5 h-3.5 text-accent/60" />
                          <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold">Retrieval & Reranking</span>
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Top-N Sem
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Initial pool (10-200)</span>
                            </label>
                            <input
                              type="number"
                              min="10"
                              max="200"
                              value={envOverrides.KG_TOP_N_SEMANTIC}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_TOP_N_SEMANTIC: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Top-K Fin
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Final context (5-100)</span>
                            </label>
                            <input
                              type="number"
                              min="5"
                              max="100"
                              value={envOverrides.KG_TOP_K_FINAL}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_TOP_K_FINAL: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                          <div className="space-y-2">
                            <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                              Rerank K
                              <span className="text-[7px] normal-case text-muted-foreground/60 italic">Precision pool (5-50)</span>
                            </label>
                            <input
                              type="number"
                              min="5"
                              max="50"
                              value={envOverrides.KG_RERANK_TOP_K}
                              onChange={(e) => setEnvOverrides(prev => ({ ...prev, KG_RERANK_TOP_K: e.target.value }))}
                              className="w-full bg-background/50 border border-white/10 p-2 text-xs font-serif focus:border-accent/50 transition-colors"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Model Settings Section */}
                      <div className="space-y-4 pt-2">
                        <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                          <Network className="w-3.5 h-3.5 text-accent/60" />
                          <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold">Intelligence Model</span>
                        </div>

                        <div className="space-y-2">
                          <label className="text-[9px] uppercase tracking-widest text-muted-foreground flex flex-col gap-1">
                            Gemini Model Version
                            <span className="text-[7px] normal-case text-muted-foreground/60 italic">Specify model used for extraction & synthesis</span>
                          </label>
                          <input
                            type="text"
                            value={envOverrides.GEMINI_MODEL}
                            onChange={(e) => setEnvOverrides(prev => ({ ...prev, GEMINI_MODEL: e.target.value }))}
                            className="w-full bg-background/50 border border-white/10 p-3 text-xs font-serif focus:border-accent/50 transition-colors"
                            placeholder="e.gemini-2.0-flash"
                          />
                        </div>
                      </div>
                    </Card>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Research Area */}
          <div className="lg:col-span-6 space-y-8">
            <Card className="rounded-none border-none bg-transparent overflow-hidden">
              <div className="mb-6 space-y-2">
                <h1 className="text-4xl font-serif font-bold text-primary">Intelligence Console</h1>
                <p className="text-sm text-muted-foreground italic font-serif leading-relaxed">
                  Query the knowledge graph for synthesized policy intelligence and cross-referenced evidence.
                </p>
              </div>

              <div className="relative group">
                <div className="absolute -inset-0.5 bg-accent/20 opacity-0 group-focus-within:opacity-100 transition-opacity blur-md" />
                <Textarea
                  placeholder="Inquire about institutional policy..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="min-h-[200px] text-lg font-serif italic bg-secondary/40 backdrop-blur-xl border-white/5 rounded-none resize-none focus:ring-0 focus:border-accent p-6 placeholder:text-muted-foreground/20 transition-all shadow-[inset_0_2px_15px_rgba(0,0,0,0.6)]"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                />
                <div className="absolute bottom-4 right-4 flex items-center gap-4">
                  <span className="text-[10px] font-serif uppercase tracking-widest text-muted-foreground">
                    {query.length} Manifestations
                  </span>
                  <Button
                    onClick={handleSubmit}
                    disabled={isSubmitting || !query.trim()}
                    className="rounded-none border border-accent/50 bg-accent/10 text-accent hover:bg-accent hover:text-accent-foreground transition-all duration-500 font-serif px-8 uppercase tracking-widest text-[10px] h-10 shadow-[0_0_15px_rgba(212,175,55,0.1)]"
                  >
                    {isSubmitting ? "SYNTHESIZING..." : "EXECUTE QUERY"}
                    {!isSubmitting && <Send className="w-3.5 h-3.5 ml-2" />}
                  </Button>
                </div>
              </div>
            </Card>

            <AnimatePresence mode="wait">
              {isSubmitting && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-6 pt-12"
                >
                  <div className="flex flex-col items-center text-center space-y-6">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                      className="w-12 h-12 border-2 border-accent border-t-transparent rounded-full"
                    />
                    <div className="space-y-2">
                      <h3 className="text-sm font-serif font-bold uppercase tracking-[0.2em] text-accent italic">Traversing knowledge structure</h3>
                      <p className="text-[11px] text-muted-foreground uppercase tracking-widest animate-pulse">Consulting high-fidelity knowledge graph...</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {error && !isSubmitting && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Card className="rounded-none border-2 border-destructive/20 bg-destructive/5 p-6">
                    <div className="flex gap-3">
                      <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0" />
                      <div>
                        <h4 className="font-serif font-bold text-destructive uppercase tracking-widest text-xs mb-1">Operational Error</h4>
                        <p className="text-sm text-muted-foreground">{error}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              )}

              {answers.length > 0 && !isSubmitting && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-12 pb-12"
                >
                  {answers.map((answer, idx) => (
                    <div key={idx} className="space-y-8">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <Search className="w-4 h-4 text-accent" />
                          <h3 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Synthesized Intelligence</h3>
                        </div>
                        <div className="prose-custom max-w-none text-primary/90 font-sans leading-relaxed selection:bg-accent/30">
                          <ReactMarkdown
                            components={{
                              h1: ({ node, ...props }) => <h1 className="text-accent font-serif font-bold mb-6 mt-8 text-4xl border-b border-accent/20 pb-2" {...props} />,
                              h2: ({ node, ...props }) => <h2 className="text-sky-400 font-serif font-bold mb-4 mt-6 text-3xl" {...props} />,
                              h3: ({ node, ...props }) => <h3 className="text-emerald-400 font-serif font-bold mb-3 mt-5 text-2xl" {...props} />,
                              h4: ({ node, ...props }) => <h4 className="text-amber-400 font-serif font-bold mb-2 mt-4 text-xl" {...props} />,
                              h5: ({ node, ...props }) => <h5 className="text-rose-400 font-serif font-bold mb-2 mt-3 text-lg" {...props} />,
                              h6: ({ node, ...props }) => <h6 className="text-indigo-400 font-serif font-bold mb-1 mt-2 text-base" {...props} />,
                              strong: ({ node, ...props }) => <strong className="text-accent font-bold" {...props} />,
                              p: ({ node, ...props }) => <p className="mb-4 text-primary/80" {...props} />,
                              ul: ({ node, ...props }) => <ul className="list-disc list-outside ml-6 mb-6 space-y-2 text-primary/80" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal list-outside ml-6 mb-6 space-y-2 text-primary/80" {...props} />,
                              li: ({ node, ...props }) => <li className="pl-2" {...props} />,
                            }}
                          >
                            {renderAnswerText(answer.answer)}
                          </ReactMarkdown>
                        </div>
                      </div>

                      {answer.extracted_facts && answer.extracted_facts.length > 0 && (
                        <div className="space-y-4">
                          <div className="flex items-center gap-3">
                            <Layers className="w-4 h-4 text-accent" />
                            <h3 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Structural Evidence</h3>
                          </div>
                          <div className="grid grid-cols-1 gap-4">
                            {answer.extracted_facts.map((fact, fIdx) => (
                              <Card key={fIdx} className="rounded-none border border-white/5 bg-secondary/40 p-6 relative overflow-hidden group hover:bg-secondary/60 transition-all">
                                <div className="absolute top-0 left-0 w-1 h-full bg-accent/30 group-hover:bg-accent group-hover:icon-glow transition-all duration-500" />
                                <div className="flex items-start gap-4">
                                  <Quote className="w-4 h-4 text-accent/40 mt-1" />
                                  <p className="text-sm text-muted-foreground font-sans leading-relaxed">
                                    {fact.fact}
                                  </p>
                                </div>
                              </Card>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Investigation Panel (Right Sidebar) */}
          <div className="lg:col-span-3 space-y-8">
            {/* Intelligence Rulebook (Protocol & Constraints) */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <ShieldCheck className="w-4 h-4 text-accent" />
                <h2 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Protocol & Constraints</h2>
              </div>
              <Card className="rounded-none border border-white/5 bg-secondary/40 p-5 space-y-5 backdrop-blur-md relative overflow-hidden group hover:bg-secondary/50 transition-colors duration-500">
                <div className="absolute top-0 right-0 w-24 h-24 bg-accent/5 rounded-full -mr-12 -mt-12 blur-2xl" />

                <div className="space-y-4 relative z-10">
                  <div className="space-y-2">
                    <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold flex items-center gap-2">
                      <div className="w-1 h-1 bg-accent rotate-45" />
                      Ingestion Guardrails
                    </span>
                    <p className="text-[10px] text-muted-foreground leading-relaxed uppercase tracking-wider font-serif italic">
                      To prevent <span className="text-destructive">429 (Rate Limit)</span> or <span className="text-destructive">503 (Overload)</span> errors, maintain parallel workers (Doc/Rel/QA) below <span className="text-primary font-bold">8</span> and Batch Size under <span className="text-primary font-bold">25</span> for free-tier API keys.
                    </p>
                  </div>

                  <div className="h-px bg-white/5" />

                  <div className="space-y-2">
                    <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold flex items-center gap-2">
                      <div className="w-1 h-1 bg-accent rotate-45" />
                      Memory Architecture
                    </span>
                    <p className="text-[10px] text-muted-foreground leading-relaxed uppercase tracking-wider font-serif italic">
                      The current intelligence model is <span className="underline decoration-accent/30 underline-offset-4">stateless</span>. It does not maintain a context window of previous inquiries. Each query is treated as an independent research session.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <span className="text-[10px] font-serif uppercase tracking-widest text-accent font-bold flex items-center gap-2">
                      <div className="w-1 h-1 bg-accent rotate-45" />
                      Optimal Parameter Rulebook
                    </span>
                    <div className="text-[9px] text-muted-foreground leading-relaxed uppercase tracking-wider font-serif italic space-y-1">
                      <div className="flex justify-between border-b border-white/5 pb-1">
                        <span>Extraction Mode</span>
                        <span className="text-primary font-bold">Hybrid Global</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                        <span>Parallel Workers</span>
                        <span className="text-primary font-bold">4 (Stable)</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                        <span>Top-N Semantic</span>
                        <span className="text-primary font-bold">60 - 100</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                        <span>Rerank-K pool</span>
                        <span className="text-primary font-bold">20 - 40</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                        <span>Top-K Final</span>
                        <span className="text-primary font-bold">30 - 60</span>
                      </div>
                    </div>
                  </div>

                  <div className="h-px bg-white/5" />

                  <div className="flex items-center gap-2 text-[9px] text-accent/60 uppercase tracking-widest font-serif italic">
                    <AlertCircle className="w-3 h-3" />
                    Adherence to protocols ensures pipeline stability.
                  </div>
                </div>
              </Card>
            </div>
            {answers.length > 0 && !isSubmitting ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-8"
              >
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <ShieldCheck className="w-4 h-4 text-accent" />
                    <h2 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Integrity Metrics</h2>
                  </div>
                  <Card className="rounded-none border border-blue-900/20 bg-secondary/60 p-6 space-y-6 backdrop-blur-xl shadow-2xl">
                    <div>
                      <span className="text-[10px] font-serif uppercase tracking-wider text-muted-foreground mb-3 block">Confidence Manifestation</span>
                      <div className={`text-center py-4 border font-serif italic text-lg ${answers[0]?.confidence === 'high'
                        ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400'
                        : answers[0]?.confidence === 'medium'
                          ? 'border-accent/30 bg-accent/5 text-accent'
                          : 'border-red-500/30 bg-red-500/5 text-red-400'
                        }`}>
                        {answers[0]?.confidence?.toUpperCase() || 'UNVERIFIED'}
                      </div>
                    </div>

                    {queryStats && (
                      <div className="space-y-3 border-t border-blue-900/20 pt-6">
                        <div className="flex justify-between items-end">
                          <span className="text-[10px] font-serif uppercase text-muted-foreground">Synthesis Time</span>
                          <span className="text-sm font-serif font-bold italic text-primary">{queryStats?.timing?.total?.toFixed(2)}s</span>
                        </div>
                        <div className="flex justify-between items-end">
                          <span className="text-[10px] font-serif uppercase text-muted-foreground">Graph Depth</span>
                          <span className="text-sm font-serif font-bold italic text-primary">L-24</span>
                        </div>
                      </div>
                    )}
                  </Card>
                </div>

                {answers[0]?.evidence_texts && answers[0]?.evidence_texts?.length > 0 && (
                  <div className="space-y-4 text-left">
                    <div className="flex items-center gap-2 mb-4">
                      <QUOTE className="w-4 h-4 text-accent" />
                      <h2 className="font-serif font-bold uppercase tracking-widest text-xs text-primary">Contextual Proofs</h2>
                    </div>
                    <div className="max-h-[500px] overflow-y-auto custom-scrollbar space-y-4 pr-2">
                      {answers[0].evidence_texts.slice(0, 12).map((text, idx) => (
                        <div key={idx} className="p-4 bg-secondary/40 border-b border-blue-900/20 hover:bg-secondary/60 transition-all relative group">
                          <span className="absolute -left-2 top-4 w-5 h-5 bg-accent text-background text-[8px] flex items-center justify-center font-bold font-serif opacity-0 group-hover:opacity-100 transition-opacity">
                            {idx + 1}
                          </span>
                          <p className="text-[11px] leading-relaxed text-muted-foreground font-sans">
                            "{text}"
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-center space-y-4 border border-dashed border-border p-8 grayscale opacity-50">
                <Network className="w-8 h-8 text-muted-foreground" />
                <p className="text-[10px] font-serif uppercase tracking-widest text-muted-foreground">Waiting for query initiation</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="container mx-auto px-6 py-12 border-t border-blue-900/20 mt-24">
        <div className="flex flex-col items-center gap-4">
          <NsureLogo className="w-6 h-6 text-accent/20" />
          <p className="text-[10px] text-muted-foreground uppercase tracking-[0.3em] font-serif italic text-center">
            Institutional Graph Intelligence Platform &bull; Nsure AI &bull; Est. 2026
          </p>
        </div>
      </footer>
    </div >
  );
}

// Helper icons not imported
const ShieldCheck = (props: any) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-shield-check"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" /><path d="m9 12 2 2 4-4" /></svg>
);

const QUOTE = (props: any) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-quote"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 2.5 1 4.5 3 6" /><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 6" /></svg>
);
