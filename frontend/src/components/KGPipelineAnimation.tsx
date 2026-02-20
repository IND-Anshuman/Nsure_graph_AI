import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    FileText, Database, Share2, Users, Search,
    Sparkles, Terminal,
    Cpu, Activity, Globe, Zap
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

type Stage = "INGESTION" | "EXTRACTION" | "LEIDEN" | "SUMMARY" | "HYBRID" | "SYNTHESIS";

const STAGES: Stage[] = ["INGESTION", "EXTRACTION", "LEIDEN", "SUMMARY", "HYBRID", "SYNTHESIS"];

const STAGE_CONFIG = {
    INGESTION: {
        title: "Data Ingestion",
        desc: "Protocols parsing unstructured policy corpora into atomic data segments.",
        icon: FileText,
        logs: [
            "Accessing document store...",
            "Reading policy_v2.pdf [742kb]",
            "Normalizing encoding structure",
            "Chunking: 1,402 atomic segments created",
            "Attaching metadata headers"
        ]
    },
    EXTRACTION: {
        title: "Entity & Relation Extraction",
        desc: "Autonomous identification of agents, provisions, and regulatory dependencies.",
        icon: Database,
        logs: [
            "Running semantic scanning...",
            "Extracted Entity: [Policy Holder]",
            "Extracted Entity: [Exclusion 4.2]",
            "Mapping Relation: [CLAIM] -> [PROCESS]",
            "Confidence check: 98.4%"
        ]
    },
    LEIDEN: {
        title: "Leiden Community Detection",
        desc: "Hierarchical partitioning of the graph into dense thematic communities.",
        icon: Share2,
        logs: [
            "Calculating node modularity...",
            "Optimizing cluster density",
            "Detected 12 macro-communities",
            "Re-partitioning level=1 nodes",
            "Global graph coherence: 0.89"
        ]
    },
    SUMMARY: {
        title: "Community Summary Formation",
        desc: "LLM-driven synthesis of high-level intelligence for each knowledge cluster.",
        icon: Users,
        logs: [
            "Synthesizing Cluster [CID: 402]",
            "Thematic focus: Medical Exclusions",
            "Drafting community report...",
            "Indexing micro-summaries",
            "Cache warmed for structural queries"
        ]
    },
    HYBRID: {
        title: "Hybrid Retrieval",
        desc: "Cross-referencing semantic proximity with graph-based neighbor expansion.",
        icon: Search,
        logs: [
            "Query: 'What if pre-existing?'",
            "Semantic hit: [Sentence_42]",
            "Graph traversal: [Exclusion_A_1]",
            "Reranking 60 candidates...",
            "Cross-encoder score: 0.94"
        ]
    },
    SYNTHESIS: {
        title: "Intelligence Synthesis",
        desc: "Final filtering and grounded report generation with technical citations.",
        icon: Sparkles,
        logs: [
            "Awaiting LLM synthesis...",
            "Applying grounded constraints",
            "Mapping used_evidence IDs",
            "Final review: Policy Alignment OK",
            "Report generation complete"
        ]
    },
};

export function KGPipelineAnimation() {
    const [stageIndex, setStageIndex] = useState(0);
    const [logLines, setLogLines] = useState<string[]>([]);
    const currentStage = STAGES[stageIndex];
    const config = STAGE_CONFIG[currentStage];

    // Rotating log lines
    useEffect(() => {
        let logIdx = 0;
        const interval = 800;
        setLogLines([]);

        const logTimer = setInterval(() => {
            if (logIdx < config.logs.length) {
                setLogLines(prev => [...prev, config.logs[logIdx]]);
                logIdx++;
            }
        }, interval);

        const stageTimer = setTimeout(() => {
            setStageIndex((prev) => (prev + 1) % STAGES.length);
        }, 5500);

        return () => {
            clearInterval(logTimer);
            clearTimeout(stageTimer);
        };
    }, [stageIndex]);

    return (
        <div className="w-full max-w-5xl mx-auto py-12 px-6">
            <div className="grid lg:grid-cols-12 gap-12 items-start">

                {/* Left: Animation Area (7 Cols) */}
                <div className="flex w-full lg:col-span-12 xl:col-span-7 relative aspect-[16/10] bg-secondary/20 rounded-none border border-white/5 realistic-shadow overflow-hidden flex flex-col min-h-[300px] lg:min-h-auto">
                    <div className="absolute inset-0 paper-overlay opacity-[0.05]" />

                    <div className="flex-1 relative flex items-center justify-center">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={currentStage}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 1.05 }}
                                transition={{ duration: 0.8 }}
                                className="relative w-full h-full flex items-center justify-center p-12"
                            >
                                {/* STAGE: INGESTION */}
                                {currentStage === "INGESTION" && (
                                    <div className="relative w-full h-full flex items-center justify-center">
                                        <motion.div
                                            className="absolute inset-0 flex items-center justify-center"
                                            initial={{ scale: 0.8 }}
                                            animate={{ scale: 1 }}
                                        >
                                            <div className="relative">
                                                <motion.div
                                                    animate={{ rotate: 360 }}
                                                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                                                    className="w-48 h-48 rounded-full border border-accent/20 border-dashed"
                                                />
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <Cpu size={48} className="text-accent/60 icon-glow" />
                                                </div>
                                            </div>
                                        </motion.div>

                                        {[...Array(6)].map((_, i) => (
                                            <motion.div
                                                key={i}
                                                initial={{ opacity: 0, x: -200, y: (i - 2.5) * 40 }}
                                                animate={{ opacity: [0, 1, 1, 0], x: 0, scale: [1, 1, 0.5] }}
                                                transition={{ delay: i * 0.3, duration: 2, repeat: Infinity }}
                                                className="absolute text-accent/40 flex items-center gap-2"
                                            >
                                                <FileText size={24} />
                                                <span className="text-[8px] font-mono whitespace-nowrap uppercase tracking-tighter">DATA_CHUNK_{i}</span>
                                            </motion.div>
                                        ))}
                                    </div>
                                )}

                                {/* STAGE: EXTRACTION */}
                                {currentStage === "EXTRACTION" && (
                                    <div className="relative w-full h-full">
                                        <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 gap-4 p-8 opacity-20">
                                            {[...Array(9)].map((_, i) => (
                                                <div key={i} className="border border-white/10 flex items-center justify-center text-[10px] font-mono">
                                                    CHUNK_{i}
                                                </div>
                                            ))}
                                        </div>
                                        {/* Scanning Beam */}
                                        <motion.div
                                            animate={{ top: ["0%", "100%", "0%"] }}
                                            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                                            className="absolute left-0 right-0 h-px bg-accent shadow-[0_0_15px_rgba(212,175,55,0.8)] z-10"
                                        />

                                        <div className="relative w-full h-full flex items-center justify-center">
                                            {[
                                                { label: "Policy Holder", x: -100, y: -60, color: "text-accent" },
                                                { label: "Exclusion 4.2", x: 120, y: 40, color: "text-rose-400" },
                                                { label: "Claim Process", x: -80, y: 80, color: "text-emerald-400" },
                                                { label: "Premium Rate", x: 60, y: -100, color: "text-sky-400" }
                                            ].map((node, i) => (
                                                <motion.div
                                                    key={i}
                                                    initial={{ scale: 0, opacity: 0 }}
                                                    animate={{ scale: 1, opacity: 1 }}
                                                    transition={{ delay: i * 0.5 }}
                                                    className="absolute flex flex-col items-center gap-1"
                                                    style={{ marginLeft: node.x, marginTop: node.y }}
                                                >
                                                    <div className={`w-3 h-3 rounded-full bg-current ${node.color} shadow-lg`} />
                                                    <span className={`text-[9px] font-bold uppercase ${node.color} whitespace-nowrap`}>{node.label}</span>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* STAGE: LEIDEN */}
                                {currentStage === "LEIDEN" && (
                                    <div className="relative w-full h-full flex items-center justify-center">
                                        <motion.div animate={{ rotate: 360 }} transition={{ duration: 60, repeat: Infinity, ease: "linear" }} className="absolute w-full h-full border border-white/5 rounded-full" />
                                        {[
                                            { name: "Legal", count: 8, color: "bg-sky-400", x: "30%", y: "40%" },
                                            { name: "Medical", count: 6, color: "bg-emerald-400", x: "70%", y: "45%" },
                                            { name: "Financial", count: 12, color: "bg-amber-400", x: "50%", y: "75%" }
                                        ].map((cluster, ci) => (
                                            <motion.div
                                                key={ci}
                                                animate={{
                                                    scale: [0.95, 1.05, 0.95],
                                                    x: [0, Math.sin(ci) * 20, 0]
                                                }}
                                                transition={{ duration: 5, repeat: Infinity }}
                                                className="absolute group"
                                                style={{ left: cluster.x, top: cluster.y }}
                                            >
                                                <div className={`w-28 h-28 rounded-full blur-3xl opacity-20 ${cluster.color}`} />
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <div className="relative">
                                                        <div className={`w-12 h-12 rounded-full border border-white/10 backdrop-blur-sm flex items-center justify-center flex-col`}>
                                                            <span className="text-[10px] font-bold uppercase tracking-widest text-white/80">{cluster.name}</span>
                                                            <span className="text-[8px] opacity-40">mod={0.88}</span>
                                                        </div>
                                                        {[...Array(cluster.count)].map((_, nodeI) => (
                                                            <div
                                                                key={nodeI}
                                                                className={`absolute w-1.5 h-1.5 rounded-full ${cluster.color} opacity-60`}
                                                                style={{
                                                                    left: `${50 + 35 * Math.cos((nodeI * 2 * Math.PI) / cluster.count)}%`,
                                                                    top: `${50 + 35 * Math.sin((nodeI * 2 * Math.PI) / cluster.count)}%`,
                                                                }}
                                                            />
                                                        ))}
                                                    </div>
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                )}

                                {/* STAGE: SUMMARY */}
                                {currentStage === "SUMMARY" && (
                                    <div className="relative w-full h-full flex items-center justify-center">
                                        <div className="absolute inset-0 border border-accent/5 rounded-none m-8" />
                                        <motion.div
                                            animate={{ opacity: [0, 1, 0], scale: [1, 1.2, 1] }}
                                            transition={{ duration: 2, repeat: Infinity }}
                                            className="absolute w-full h-full bg-accent/5 blur-3xl rounded-full"
                                        />
                                        <div className="relative flex gap-8">
                                            {[0, 1].map((i) => (
                                                <motion.div
                                                    key={i}
                                                    initial={{ opacity: 0, y: 20 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: i * 0.4 }}
                                                    className="w-40 bg-secondary/80 border border-white/10 p-4 relative backdrop-blur-xl"
                                                >
                                                    <div className="absolute top-0 right-0 p-1 opacity-20"><Activity size={12} /></div>
                                                    <div className="flex items-center gap-2 mb-3">
                                                        <Users size={14} className="text-accent" />
                                                        <span className="text-[9px] font-bold uppercase text-accent/60">Community {400 + i}</span>
                                                    </div>
                                                    <div className="space-y-2">
                                                        <div className="h-1.5 w-full bg-white/10 rounded-full" />
                                                        <div className="h-1.5 w-5/6 bg-white/5 rounded-full" />
                                                        <div className="h-1.5 w-full bg-white/10 rounded-full" />
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* STAGE: HYBRID */}
                                {currentStage === "HYBRID" && (
                                    <div className="relative w-full h-full flex flex-col items-center justify-center">
                                        <motion.div
                                            className="bg-accent/10 border border-accent/20 px-6 py-2 mb-12 backdrop-blur-md relative"
                                            animate={{ scale: [1, 1.02, 1] }}
                                            transition={{ duration: 2, repeat: Infinity }}
                                        >
                                            <span className="text-xs font-serif italic text-accent line-clamp-1">"How does Exclusion 4.2 affect my claim?"</span>
                                            <div className="absolute -bottom-1 -right-1 opacity-40"><Zap size={10} className="text-accent" /></div>
                                        </motion.div>

                                        <div className="relative w-full h-32 flex items-center justify-center overflow-visible">
                                            {/* Paths */}
                                            <div className="absolute w-2/3 h-px bg-white/5 dashed" />

                                            <div className="grid grid-cols-2 gap-24 relative z-10 w-full px-12">
                                                <motion.div
                                                    className="flex flex-col items-center gap-3"
                                                    animate={{ opacity: [0.4, 1, 0.4] }}
                                                    transition={{ duration: 1.5, repeat: Infinity }}
                                                >
                                                    <div className="w-12 h-12 bg-sky-500/10 border border-sky-500/20 flex items-center justify-center rounded-none shadow-lg">
                                                        <Globe size={20} className="text-sky-400" />
                                                    </div>
                                                    <span className="text-[9px] font-bold uppercase text-sky-400 tracking-widest text-center">Semantic<br />Path</span>
                                                </motion.div>

                                                <motion.div
                                                    className="flex flex-col items-center gap-3"
                                                    animate={{ opacity: [0.4, 1, 0.4] }}
                                                    transition={{ duration: 1.5, repeat: Infinity, delay: 0.75 }}
                                                >
                                                    <div className="w-12 h-12 bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center rounded-none shadow-lg">
                                                        <Share2 size={20} className="text-emerald-400" />
                                                    </div>
                                                    <span className="text-[9px] font-bold uppercase text-emerald-400 tracking-widest text-center">Graph<br />Expansion</span>
                                                </motion.div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* STAGE: SYNTHESIS */}
                                {currentStage === "SYNTHESIS" && (
                                    <div className="relative w-full h-full flex items-center justify-center">
                                        <motion.div
                                            initial={{ scale: 0.8, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            className="w-full max-w-md bg-secondary border border-accent/40 p-10 realistic-shadow relative overflow-hidden"
                                        >
                                            <div className="absolute top-0 right-0 p-2"><Sparkles size={16} className="text-accent/40 animate-pulse" /></div>
                                            <div className="absolute top-0 left-0 w-1.5 h-full bg-accent icon-glow" />

                                            <div className="space-y-6">
                                                <div className="flex items-center gap-3 border-b border-white/5 pb-4">
                                                    <div className="w-8 h-8 rounded-full border border-accent/20 flex items-center justify-center">
                                                        <span className="text-[10px] text-accent font-serif font-bold italic">N</span>
                                                    </div>
                                                    <h4 className="text-xs font-serif font-bold uppercase tracking-widest text-primary italic">Intelligence Summary</h4>
                                                </div>

                                                <div className="space-y-3">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-1 h-1 bg-accent rounded-full" />
                                                        <div className="h-1.5 w-full bg-white/10 rounded-full" />
                                                    </div>
                                                    <div className="h-1.5 w-[90%] bg-white/5 rounded-full ml-3" />
                                                    <div className="h-1.5 w-[85%] bg-white/5 rounded-full ml-3" />
                                                    <div className="flex items-center gap-2 pt-2">
                                                        <div className="w-1 h-1 bg-accent rounded-full" />
                                                        <div className="h-1.5 w-[80%] bg-white/10 rounded-full" />
                                                    </div>
                                                </div>

                                                <div className="pt-4 flex gap-2">
                                                    {[1, 2, 3].map(i => (
                                                        <div key={i} className="px-2 py-1 bg-accent/5 border border-accent/20 text-[7px] text-accent font-mono">CIT_{i}F2</div>
                                                    ))}
                                                </div>
                                            </div>
                                        </motion.div>
                                    </div>
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Terminal Area */}
                    <div className="h-32 bg-black/40 border-t border-white/5 p-4 font-mono text-[10px] relative overflow-hidden flex flex-col gap-1">
                        <div className="absolute top-1 right-2 opacity-10"><Terminal size={14} /></div>
                        <div className="flex items-center gap-2 mb-2 pb-1 border-b border-white/5">
                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="text-[9px] uppercase tracking-widest text-muted-foreground italic">Pipeline Logs :: {currentStage}</span>
                        </div>
                        <div className="flex-1 overflow-hidden space-y-1">
                            {logLines.map((line, i) => (
                                <motion.div
                                    key={line + i}
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="flex items-center gap-2"
                                >
                                    <span className="text-accent/40 select-none">›</span>
                                    <span className="text-primary/70">{line}</span>
                                </motion.div>
                            ))}
                            {logLines.length < 5 && (
                                <motion.div
                                    animate={{ opacity: [0, 1, 0] }}
                                    transition={{ repeat: Infinity, duration: 1 }}
                                    className="w-1.5 h-3 bg-accent/40 inline-block ml-1"
                                />
                            )}
                        </div>
                    </div>
                </div>

                {/* Right: Info Area (5 Cols) */}
                <div className="lg:col-span-12 xl:col-span-5 space-y-10 xl:pl-4">
                    <div className="space-y-6">
                        <Badge variant="outline" className="border-accent/40 text-accent uppercase tracking-widest text-[10px] px-4 py-1.5 bg-accent/5 font-serif italic">
                            Stage 0{stageIndex + 1}: {currentStage}
                        </Badge>

                        <div className="space-y-4">
                            <motion.h3
                                key={config.title}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-4xl font-serif font-bold text-primary italic leading-tight uppercase tracking-tighter"
                            >
                                {config.title}
                            </motion.h3>

                            <div className="relative">
                                <div className="absolute left-0 top-0 bottom-0 w-px bg-gradient-to-b from-accent/40 via-accent/10 to-transparent" />
                                <motion.p
                                    key={config.desc}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.1 }}
                                    className="text-lg text-muted-foreground font-serif leading-relaxed italic pl-8 py-2 selection:bg-accent/30 lowercase first-letter:uppercase"
                                >
                                    {config.desc}
                                </motion.p>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-6 gap-2">
                        {STAGES.map((s, i) => (
                            <div key={s} className="space-y-2 group cursor-pointer" onClick={() => setStageIndex(i)}>
                                <div className={`h-1.5 transition-all duration-700 ${i === stageIndex ? "bg-accent shadow-[0_0_15px_rgba(212,175,55,0.6)]" : "bg-white/5 group-hover:bg-white/10"}`} />
                                <span className={`text-[8px] uppercase tracking-widest font-bold block text-center transition-colors ${i === stageIndex ? "text-accent" : "text-muted-foreground/40"}`}>
                                    P{i + 1}
                                </span>
                            </div>
                        ))}
                    </div>

                    <div className="pt-8 space-y-4">
                        <div className="flex items-center gap-4 p-5 bg-secondary/30 border border-white/5 group transition-colors">
                            <div className="w-12 h-12 flex items-center justify-center bg-accent/5 border border-accent/20 group-hover:border-accent/50 transition-colors">
                                <config.icon size={24} className="text-accent/60 group-hover:text-accent transition-colors" />
                            </div>
                            <div>
                                <span className="text-[10px] uppercase font-serif tracking-[0.2em] text-accent/80 block mb-1 underline underline-offset-4 decoration-accent/20">Active Module</span>
                                <span className="text-xs text-muted-foreground font-serif italic">Subsystem monitoring active • 99.9% fidelity</span>
                            </div>
                        </div>
                        <div className="pt-2 text-center">
                            <p className="text-[10px] text-accent/60 font-serif italic uppercase tracking-[0.15em]">
                                Processing Note: The first query requires 3-4 minutes due to graph building. Subsequent queries are faster.
                            </p>
                        </div>
                        <p className="text-[10px] text-muted-foreground/40 font-serif italic text-center uppercase tracking-widest">
                            Autonomous Knowledge Engineering Frame
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
