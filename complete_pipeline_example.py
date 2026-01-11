"""
Complete end-to-end Knowledge Graph pipeline with proper function usage.
This example demonstrates the full workflow:
1. Data ingestion from PDF/URLs
2. Document preprocessing into sentences
3. Entity extraction with LLM clustering
4. Building entity catalog and nodes
5. Co-occurrence and semantic relation edges
6. Multi-level community detection
7. Enhanced retrieval index building
8. Hybrid semantic search
9. LLM-based reranking
10. LLM-based answer synthesis

Usage:
    python complete_pipeline_example.py
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Core modules
from graph_maker.data_corpus import (
    KnowledgeGraph,
    build_corpus_from_sources,
    add_document_and_sentence_nodes,
    SentenceInfo,
)

# NER and entity extraction
from graph_maker.ner import (
    extract_semantic_entities_for_doc,
    build_entity_catalog,
    add_entity_nodes,
    add_mention_and_cooccurrence_edges,
    add_semantic_relation_edges,
)

# Community detection
from graph_maker.Community_processing import (
    compute_multilevel_communities,
    build_and_add_community_nodes,
)

# Retrieval
from answer_synthesis.retrieval import (
    build_retrieval_index_enhanced,
    IndexItem,
)
from answer_synthesis.hybrid_search import search_and_expand, RetrievalCandidate
from answer_synthesis.llm_rerank import llm_rerank_candidates
from answer_synthesis.llm_synthesis import llm_synthesize_answer

# Setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompleteKnowledgeGraphPipeline:
    """
    Full end-to-end pipeline for knowledge graph construction and querying.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.kg = KnowledgeGraph()
        self.corpus: Dict[str, str] = {}
        self.sent_index: Dict[str, SentenceInfo] = {}
        self.all_entities_per_doc: Dict[str, List] = {}
        self.partition: Dict[str, int] = {}
        self.index_items: List[IndexItem] = []
        self.embeddings = None

        logger.info("Knowledge Graph Pipeline initialized")

    def phase_1_ingestion(self, sources: List[str]) -> None:
        """
        Phase 1: Ingest documents from sources (URLs or file paths).
        
        Parameters
        ----------
        sources : List[str]
            List of URLs or file paths to ingest.
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: DOCUMENT INGESTION")
        logger.info("=" * 80)

        try:
            self.corpus = build_corpus_from_sources(sources)
            logger.info(f"✓ Ingested {len(self.corpus)} documents")
            for doc_id, text in self.corpus.items():
                logger.info(f"  - {doc_id}: {len(text)} characters")
        except Exception as e:
            logger.error(f"✗ Ingestion failed: {e}")
            raise

    def phase_2_preprocessing(self) -> None:
        """
        Phase 2: Preprocess documents into sentences and create document nodes.
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: DOCUMENT PREPROCESSING")
        logger.info("=" * 80)

        try:
            self.sent_index = add_document_and_sentence_nodes(self.kg, self.corpus)
            logger.info(f"✓ Created document and sentence nodes")
            logger.info(f"  - Total sentences: {len(self.sent_index)}")
            logger.info(f"  - Total documents: {len(self.corpus)}")
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {e}")
            raise

    def phase_3_ner_extraction(self) -> None:
        """
        Phase 3: Extract entities using semantic extraction with LLM clustering.
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: ENTITY EXTRACTION & NER")
        logger.info("=" * 80)

        total_entities = 0
        try:
            for doc_id, text in self.corpus.items():
                logger.info(f"Processing {doc_id}...")
                entities = extract_semantic_entities_for_doc(doc_id, text)
                self.all_entities_per_doc[doc_id] = entities
                total_entities += len(entities)
                logger.info(f"  - {len(entities)} entities extracted")

            logger.info(f"✓ Entity extraction complete")
            logger.info(f"  - Total entities across all docs: {total_entities}")
        except Exception as e:
            logger.error(f"✗ Entity extraction failed: {e}")
            raise

    def phase_4_entity_catalog(self) -> None:
        """
        Phase 4: Build entity catalog and add entity nodes to graph.
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: ENTITY CATALOG & NODES")
        logger.info("=" * 80)

        try:
            # Build catalog (deduplication and entity linking)
            catalog = build_entity_catalog(self.all_entities_per_doc)
            logger.info(f"✓ Entity catalog built")
            logger.info(f"  - Canonical entities: {len(catalog)}")

            # Add entity nodes to graph
            add_entity_nodes(self.kg, catalog)
            logger.info(f"✓ Added {len(catalog)} entity nodes to graph")
        except Exception as e:
            logger.error(f"✗ Entity catalog building failed: {e}")
            raise

    def phase_5_mention_edges(self) -> None:
        """
        Phase 5: Add mention edges and co-occurrence edges.
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: MENTION & CO-OCCURRENCE EDGES")
        logger.info("=" * 80)

        try:
            add_mention_and_cooccurrence_edges(
                self.kg, self.all_entities_per_doc, self.sent_index
            )
            logger.info(f"✓ Added mention and co-occurrence edges")
            logger.info(f"  - Total edges so far: {len(self.kg.edges)}")
        except Exception as e:
            logger.error(f"✗ Edge addition failed: {e}")
            raise

    def phase_6_semantic_relations(self) -> None:
        """
        Phase 6: Extract semantic relations between entities.
        """
        logger.info("=" * 80)
        logger.info("PHASE 6: SEMANTIC RELATIONS")
        logger.info("=" * 80)

        try:
            add_semantic_relation_edges(
                self.kg, self.all_entities_per_doc, self.sent_index
            )
            logger.info(f"✓ Added semantic relation edges")
            logger.info(f"  - Total edges after semantic relations: {len(self.kg.edges)}")
        except Exception as e:
            logger.error(f"✗ Semantic relation extraction failed: {e}")
            raise

    def phase_7_community_detection(self) -> None:
        """
        Phase 7: Detect multi-level communities in the entity graph.
        """
        logger.info("=" * 80)
        logger.info("PHASE 7: COMMUNITY DETECTION")
        logger.info("=" * 80)

        try:
            # Detect multi-level communities
            community_results = compute_multilevel_communities(
                self.kg,
                max_levels=None,
                min_comm_size=1,
                verbose=True,
            )
            logger.info(f"✓ Community detection complete")
            logger.info(f"  - Levels detected: {len(community_results)}")

            # Build and add community nodes
            build_and_add_community_nodes(
                self.kg,
                community_results,
                create_member_edges=True,
                create_partof_edges=True,
                verbose=True,
            )
            logger.info(f"✓ Community nodes added to graph")
            logger.info(f"  - Total nodes in graph: {len(self.kg.nodes)}")
        except Exception as e:
            logger.error(f"✗ Community detection failed: {e}")
            raise

    def phase_8_retrieval_index(self) -> None:
        """
        Phase 8: Build enhanced retrieval index for semantic search.
        """
        logger.info("=" * 80)
        logger.info("PHASE 8: RETRIEVAL INDEX BUILDING")
        logger.info("=" * 80)

        try:
            self.index_items, self.embeddings = build_retrieval_index_enhanced(
                graph=self.kg,
                context_window_tokens=50,
                include_entity_contexts=True,
                chunk_size_words=160,
                chunk_overlap_words=40,
                min_sentence_chars=25,
                min_text_chars=40,
                verbose=True,
            )
            logger.info(f"✓ Retrieval index built")
            logger.info(f"  - Index items: {len(self.index_items)}")
            logger.info(f"  - Embedding dimension: {self.embeddings.shape[1]}")
        except Exception as e:
            logger.error(f"✗ Index building failed: {e}")
            raise

    def build_graph(self, sources: List[str]) -> None:
        """
        Run the complete pipeline to build the knowledge graph.
        
        Parameters
        ----------
        sources : List[str]
            List of URLs or file paths to ingest.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE KNOWLEDGE GRAPH PIPELINE")
        logger.info("=" * 80 + "\n")

        try:
            self.phase_1_ingestion(sources)
            self.phase_2_preprocessing()
            self.phase_3_ner_extraction()
            self.phase_4_entity_catalog()
            self.phase_5_mention_edges()
            self.phase_6_semantic_relations()
            self.phase_7_community_detection()
            self.phase_8_retrieval_index()

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETE - GRAPH READY FOR QUERYING")
            logger.info("=" * 80)
            logger.info(f"Final Graph Statistics:")
            logger.info(f"  - Total nodes: {len(self.kg.nodes)}")
            logger.info(f"  - Total edges: {len(self.kg.edges)}")
            logger.info(f"  - Index items: {len(self.index_items)}")
            logger.info("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"\n✗ PIPELINE FAILED: {e}")
            raise

    def query(self, question: str, top_k: int = 5, verbose: bool = True) -> str:
        """
        Query the knowledge graph with multi-stage retrieval and synthesis.
        
        Parameters
        ----------
        question : str
            The question to answer.
        top_k : int
            Number of top evidence items to use (default 5).
        verbose : bool
            Whether to print detailed progress (default True).
            
        Returns
        -------
        str
            The synthesized answer with citations.
        """
        if not self.index_items or self.embeddings is None:
            logger.error("Graph not built. Run build_graph() first.")
            return "Error: Graph not built yet."

        logger.info("\n" + "=" * 80)
        logger.info("QUERYING KNOWLEDGE GRAPH")
        logger.info("=" * 80)
        logger.info(f"Question: {question}\n")

        try:
            # Stage 1: Hybrid semantic search
            logger.info("[Stage 1] Hybrid Semantic Search")
            logger.info("-" * 80)
            candidates = search_and_expand(
                query=question,
                graph=self.kg,
                index_items=self.index_items,
                embeddings=self.embeddings,
                top_n_semantic=20,
                top_k_final=40,
                alpha=0.7,
                beta=0.3,
                expansion_hops=1,
                include_community_expansion=True,
                min_hybrid_score=0.18,
                dedup_overlap_threshold=0.9,
                community_boost=0.15,
                verbose=verbose,
            )
            logger.info(f"✓ Retrieved {len(candidates)} candidates\n")

            if not candidates:
                logger.warning("✗ No candidates retrieved")
                return "Unable to find relevant information to answer your question."

            # Stage 2: LLM-based reranking
            logger.info("[Stage 2] LLM-Based Reranking")
            logger.info("-" * 80)
            rerank_result = llm_rerank_candidates(
                query=question,
                candidates=candidates,
                top_k=top_k,
                model_name="gemini-2.0-flash",
                temperature=0.1,
                use_cache=True,
                verbose=verbose,
            )
            ranked = rerank_result["ranked_candidates"]
            logger.info(f"✓ Reranked to top {len(ranked)} evidence items\n")

            if not ranked:
                logger.warning("✗ No ranked candidates")
                return "Unable to find relevant information to answer your question."

            # Stage 3: LLM-based answer synthesis
            logger.info("[Stage 3] LLM-Based Answer Synthesis")
            logger.info("-" * 80)
            synthesis_result = llm_synthesize_answer(
                query=question,
                evidence_candidates=ranked,
                model_name="gemini-2.0-flash",
                temperature=0.2,
                use_cache=True,
                verbose=verbose,
            )
            logger.info(f"✓ Answer synthesized\n")

            # Format and display result
            answer = synthesis_result.get("answer", "")
            confidence = synthesis_result.get("confidence", "unknown")
            used_evidence = synthesis_result.get("used_evidence", [])

            logger.info("=" * 80)
            logger.info("ANSWER")
            logger.info("=" * 80)
            logger.info(f"Confidence: {confidence}")
            logger.info(f"Evidence used: {len(used_evidence)} items")
            logger.info("=" * 80 + "\n")

            return answer

        except Exception as e:
            logger.error(f"✗ Query failed: {e}")
            return f"Error during query processing: {str(e)}"


def main():
    """Main entry point with example usage."""
    
    # Define sources (URLs or file paths)
    sources = [
        "https://www.ijrti.org/papers/IJRTI2304061.pdf"
        # Add more sources as needed
    ]

    # Create pipeline and build graph
    pipeline = CompleteKnowledgeGraphPipeline()
    pipeline.build_graph(sources)

    # Example queries
    queries = [
        "What are the major application domains of Artificial Intelligence discussed in the text, and how are they grouped into related communities?",
        "Explain the role of Natural Language Processing (NLP) in enabling machines to understand human communication. Give two real-world applications mentioned in the paper and justify their impact.",
        "The paper discusses applications of AI in multiple sectors. Choose any two sectors and critically analyze how AI transforms decision-making within them.",
        "Based on the paper, what are the major challenges and limitations of deploying AI systems in real-world environments? Suggest practical solutions for each challenge.",
    ]

    print("\n" + "=" * 80)
    print("QUERYING THE KNOWLEDGE GRAPH")
    print("=" * 80 + "\n")

    for query in queries:
        print(f"\nQ: {query}")
        answer = pipeline.query(query, top_k=5, verbose=False)
        print(f"\nA: {answer}")
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
