# Changelog

All notable changes to the Knowledge Graph Agent project.

## [2.0.0] - 2024 - Major Restructuring

### Added

#### Core Architecture
- Created modular phase-based architecture with 7 distinct pipeline phases
- New `src/core/` module with:
  - `graph_schema.py`: KGNode, KGEdge, KnowledgeGraph classes
  - `entity_schema.py`: Entity, SentenceInfo, EntityCatalogEntry schemas
  - `config.py`: Centralized configuration management

#### Utilities
- New `src/utils/` module with:
  - `cache.py`: Thread-safe disk-based JSON caching
  - `embeddings.py`: Embedding model loading and caching
  - `llm_client.py`: Unified Gemini/OpenAI client interface

#### Phase Modules
- **Phase 1 - Ingestion** (`src/phase1_ingestion/`):
  - `text_extractors.py`: URL, PDF, file extraction
  - `corpus_builder.py`: Multi-source corpus creation

- **Phase 2 - Preprocessing** (`src/phase2_preprocessing/`):
  - `document_processor.py`: Document and sentence node creation
  - `text_cleaner.py`: Text cleaning utilities

- **Phase 3 - NER** (`src/phase3_ner/`):
  - `entity_extractor.py`: Entity and relation extraction
  - `entity_catalog.py`: Canonical entity management
  - `spacy_utils.py`: spaCy model utilities

- **Phase 4 - Relations** (`src/phase4_relations/`):
  - `cooccurrence_edges.py`: Co-occurrence edge creation
  - `semantic_relations.py`: Semantic relation extraction

- **Phase 5 - Communities** (`src/phase5_communities/`):
  - `community_detector.py`: Louvain and label propagation
  - `community_builder.py`: Community node creation
  - `community_summarizer.py`: LLM-based summarization

- **Phase 6 - Retrieval** (`src/phase6_retrieval/`):
  - `index_builder.py`: Vector index for semantic search
  - `hybrid_search.py`: Combined vector and keyword search
  - `reranker.py`: LLM-based result reranking
  - `synthesizer.py`: Answer synthesis from context

- **Phase 7 - Persistence** (`src/phase7_persistence/`):
  - `graphml_exporter.py`: GraphML export/import
  - `neo4j_exporter.py`: Neo4j database integration

#### Pipeline
- New `KnowledgeGraphPipeline` orchestrator class
- Simplified end-to-end workflow
- Built-in query interface for Q&A

#### Documentation
- Comprehensive `README.md` with examples
- `MIGRATION.md` guide for transitioning from old structure
- `RESTRUCTURING_SUMMARY.md` with complete overview
- `.env.example` template for configuration

#### Examples
- `example_wikipedia.py`: Wikipedia article processing
- `example_pdf.py`: PDF processing with interactive queries
- `quickstart.py`: Environment setup and validation

#### Testing
- `tests/test_structure.py`: Import verification and basic tests
- Test infrastructure ready for expansion

### Changed
- Migrated from flat file structure to modular architecture
- Centralized all configuration in `Config` class
- Standardized import paths using phase-based organization
- Updated `requirements.txt` with version numbers and categories
- Enhanced error handling across all modules
- Improved logging with consistent formatting

### Fixed
- Windows file permission issues in cache handling (unlink → rename)
- Neo4j 5.x constraint syntax (FOR...REQUIRE instead of ON...ASSERT)
- GraphML serialization for complex types (lists, dicts)
- Robust JSON parsing for LLM outputs with fallback extraction

### Preserved
- All original functionality from flat structure
- NER_Extraction_Agent integration
- Full compatibility with existing .env configuration
- All bug fixes from previous versions

## [1.0.0] - Previous Version

### Features
- Basic knowledge graph construction pipeline
- NER extraction with spaCy and LLMs
- Community detection with Louvain algorithm
- Hybrid search and retrieval
- GraphML and Neo4j export

### Known Issues (Fixed in 2.0.0)
- Monolithic file structure difficult to maintain
- Scattered configuration across files
- Windows-specific file handling issues
- Neo4j syntax compatibility

---

## Migration from 1.x to 2.0

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

### Key Changes
1. Import paths updated (phase-based organization)
2. Configuration centralized in `Config` class
3. Graph operations use `KnowledgeGraph` class
4. Pipeline usage simplified with `KnowledgeGraphPipeline`

### Example Migration

**Before (1.x):**
```python
from data_corpus import build_corpus_from_sources
from NER import extract_entities
```

**After (2.0):**
```python
from src.phase1_ingestion import build_corpus_from_sources
from src.phase3_ner import process_document_for_ner
```

---

## Version Numbering

- **Major version** (2.x): Breaking changes, architecture changes
- **Minor version** (x.1): New features, backward compatible
- **Patch version** (x.x.1): Bug fixes, minor improvements
