# Knowledge Graph Agent

A modular, phase-based pipeline for constructing and querying knowledge graphs from unstructured text.

## 🏗️ Architecture

The project is organized into 7 phases, each handling a specific aspect of knowledge graph construction:

```
src/
├── core/              # Core data structures and configuration
│   ├── graph_schema.py       # KGNode, KGEdge, KnowledgeGraph
│   ├── entity_schema.py      # Entity, SentenceInfo, EntityCatalogEntry
│   └── config.py             # Centralized configuration
│
├── utils/             # Shared utilities
│   ├── cache.py              # Disk-based JSON caching
│   ├── embeddings.py         # Embedding model and caching
│   └── llm_client.py         # Gemini/OpenAI client wrappers
│
├── phase1_ingestion/  # Phase 1: Document ingestion
│   ├── text_extractors.py    # URL, PDF, file extraction
│   └── corpus_builder.py     # Corpus creation from sources
│
├── phase2_preprocessing/  # Phase 2: Text preprocessing
│   ├── document_processor.py # Document/sentence node creation
│   └── text_cleaner.py       # Text cleaning utilities
│
├── phase3_ner/        # Phase 3: Named Entity Recognition
│   ├── entity_extractor.py   # Entity & relation extraction
│   ├── entity_catalog.py     # Canonical entity management
│   └── spacy_utils.py        # spaCy model utilities
│
├── phase4_relations/  # Phase 4: Relationship building
│   ├── cooccurrence_edges.py # Co-occurrence edge creation
│   └── semantic_relations.py # Semantic relation extraction
│
├── phase5_communities/  # Phase 5: Community detection
│   ├── community_detector.py # Louvain, label propagation
│   ├── community_builder.py  # Community node creation
│   └── community_summarizer.py # LLM-based summarization
│
├── phase6_retrieval/  # Phase 6: Search and retrieval
│   ├── index_builder.py      # Vector index creation
│   ├── hybrid_search.py      # Vector + keyword search
│   ├── reranker.py           # LLM-based reranking
│   └── synthesizer.py        # Answer synthesis
│
├── phase7_persistence/  # Phase 7: Export and storage
│   ├── graphml_exporter.py   # GraphML export/import
│   └── neo4j_exporter.py     # Neo4j database export
│
└── pipeline.py        # Main orchestrator (KnowledgeGraphPipeline)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Knowledge-Graph-Agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
from src.pipeline import KnowledgeGraphPipeline

# Define data sources
sources = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    {"file": "data/research_paper.pdf"},
    {"url": "https://example.com/article"},
]

# Create and run pipeline
pipeline = KnowledgeGraphPipeline()
pipeline.build_graph(sources, export_neo4j_flag=True)

# Query the knowledge graph
answer = pipeline.query("What is artificial intelligence?")
print(answer)
```

## 📋 Pipeline Phases

### Phase 1: Ingestion
Extract text from various sources (URLs, PDFs, local files).

```python
from src.phase1_ingestion import build_corpus_from_sources

corpus = build_corpus_from_sources(sources, kg)
```

### Phase 2: Preprocessing
Clean text and split into sentences.

```python
from src.phase2_preprocessing import add_document_and_sentence_nodes

sentences = add_document_and_sentence_nodes(kg, doc_id, text)
```

### Phase 3: NER
Extract entities and relations using NER_Extraction_Agent.

```python
from src.phase3_ner import process_document_for_ner

result = process_document_for_ner(kg, doc_id, text)
```

### Phase 4: Relations
Build co-occurrence and semantic relationships.

```python
from src.phase4_relations import add_cooccurrence_edges

add_cooccurrence_edges(kg, window_size=3)
```

### Phase 5: Communities
Detect communities and generate summaries.

```python
from src.phase5_communities import (
    detect_communities_louvain,
    summarize_all_communities,
    add_community_nodes
)

partition = detect_communities_louvain(kg)
summaries = summarize_all_communities(kg, partition)
add_community_nodes(kg, partition, summaries)
```

### Phase 6: Retrieval
Build indices and answer questions.

```python
from src.phase6_retrieval import (
    build_entity_index,
    hybrid_search,
    rerank_with_llm,
    synthesize_answer
)

index = build_entity_index(kg)
results = hybrid_search(query, index)
reranked = rerank_with_llm(query, results)
answer = synthesize_answer(query, reranked)
```

### Phase 7: Persistence
Export to GraphML and Neo4j.

```python
from src.phase7_persistence import export_to_graphml, export_to_neo4j

export_to_graphml(kg, "outputs/graph.graphml")
export_to_neo4j(kg)  # Requires Neo4j configuration
```

## ⚙️ Configuration

Configuration is managed through environment variables (`.env` file):

```bash
# LLM APIs
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_MODEL=gemini-2.0-flash
OPENAI_MODEL=gpt-4o-mini

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Models
SPACY_MODEL=en_core_web_sm
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Paths
CACHE_DIR=.cache
OUTPUT_DIR=outputs
```

## 📦 Dependencies

- **NLP**: spacy, sentence-transformers
- **LLMs**: google-generativeai, openai
- **Graph**: networkx, python-louvain
- **Database**: neo4j (optional)
- **Utils**: requests, beautifulsoup4, pypdf

See [requirements.txt](requirements.txt) for complete list.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run specific phase tests
pytest tests/test_phase1_ingestion.py
```

## 📊 Output Formats

### GraphML
Standard graph format, compatible with Gephi, Cytoscape, etc.

```python
from src.phase7_persistence import import_from_graphml

kg = import_from_graphml("outputs/knowledge_graph.graphml")
```

### Neo4j
Full-featured graph database with Cypher query language.

```cypher
// Find all entities of type PERSON
MATCH (e:Entity {type: "PERSON"})
RETURN e.name, e.mention_count

// Find communities
MATCH (c:Community)-[:HAS_MEMBER]->(e:Entity)
RETURN c.summary, count(e) as size
ORDER BY size DESC
```

## 🔧 Extending the Pipeline

### Adding a Custom Phase

1. Create a new directory: `src/phase8_custom/`
2. Implement your phase logic
3. Add to `pipeline.py`:

```python
def run_custom_phase(self):
    from src.phase8_custom import custom_function
    custom_function(self.kg)
```

### Custom Entity Extraction

Replace the default NER agent:

```python
from src.phase3_ner import entity_extractor

# Override the extraction function
def my_custom_extractor(text, doc_id, chunk_id):
    # Your custom logic
    return {...}

entity_extractor.extract_entities_from_text = my_custom_extractor
```

## 🐛 Troubleshooting

### Neo4j Connection Issues
- Ensure Neo4j is running: `systemctl status neo4j`
- Check credentials in `.env`
- Use correct syntax for Neo4j 5.x (REQUIRE instead of ASSERT)

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Out of Memory
- Reduce batch sizes in config
- Process documents one at a time
- Use smaller embedding model

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the modular phase-based structure
4. Add tests for new functionality
5. Submit a pull request

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{knowledge_graph_agent,
  title = {Knowledge Graph Agent: Modular Pipeline for Graph Construction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Knowledge-Graph-Agent}
}
```

## 🔗 Related Projects

- [NER_Extraction_Agent](NER_Extraction_Agent.py) - Entity and semantic extraction
- [GraphRAG](https://github.com/microsoft/graphrag) - Microsoft's graph-based RAG
- [AutoGraphRAG](https://arxiv.org/abs/xxxx) - Automated knowledge graph construction

## 📧 Contact

For questions and support, please open an issue on GitHub.
