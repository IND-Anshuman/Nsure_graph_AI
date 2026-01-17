# Nsure AI: The Architecture of Knowledge

Nsure AI is a prestigious, high-fidelity **GraphRAG (Graph-based Retrieval-Augmented Generation)** environment designed for deep document understanding and authoritative policy intelligence. It transcends traditional RAG by combining the semantic power of vector search with the structural precision of a Knowledge Graph.

![Nsure AI Pipeline](https://img.shields.io/badge/Architecture-GraphRAG-gold?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-Python_%7C_React_%7C_Gemini-blue?style=for-the-badge)

---

## 🏛 Core Philosophy
Traditional AI often hallucinates because it lacks structural "contextual grounding." Nsure AI solves this by:
1.  **Extracting** entities and regulatory relations from unstructured text.
2.  **Clustering** knowledge into thematic communities using the Leiden algorithm.
3.  **Synthesizing** answers using a **Hierarchy of Evidence** that prioritizes reranked, high-confidence sources.

---

## 🚀 Key Features

### 1. Narrative KG Pipeline Visualization
A dynamic, 6-stage operational animation on the homepage that demonstrates the system's inner workings:
- **Ingestion**: Multi-protocol parsing of PDFs and URLs.
- **Extraction**: Node-edge mapping of agents, provisions, and dependencies.
- **Leiden Detection**: Hierarchical community partitioning.
- **Hybrid Retrieval**: Simultaneous semantic scanning and graph neighbor expansion.

### 2. Scenario-Based Reasoning
The system is uniquely optimized for complex "What-If" queries. It identifies conditional rules and policy exclusions (e.g., "Section 4.2", "Clause B") to provide precise assessments for situational questions.

### 3. High-Fidelity Synthesis
- **Evidence Scaling**: Accesses up to 60+ candidates with a 4,000-character context budget.
- **Grounded Assertions**: Explicitly instructed to avoid hedging ("it seems") in favor of direct, evidence-backed statements and technical citations.

### 4. Professional Intelligence Console
- **Premium Aesthetics**: A dark, glassmorphic UI with vibrant heading hierarchies and gold-accented terminology.
- **Citable Output**: Every answer includes a list of used evidence IDs and extracted key facts for instant auditability.

---

## 🛠 Technical Architecture

### Backend (Python/FastAPI)
- **Knowledge Graph**: Built using `spacy` for NLP and custom relationship schemas.
- **LLM Engine**: Powered by Google Gemini (Flash/Pro) for extraction and answer synthesis.
- **Reranking**: Custom cross-encoder logic for prioritizing policy relevance over simple keyword overlap.
- **Vector Search**: Integrated semantic indexing of document chunks.

### Frontend (React/TypeScript/Vite)
- **UI Framework**: Tailwind CSS with custom thematic extensions.
- **Animations**: Framer Motion for premium stage transitions.
- **Components**: Fully custom Markdown renderer with high-contrast accessibility.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Google AI API Key](https://aistudio.google.com/app/apikey) (Gemini)

### Backend Setup
1. Clone the repository and navigate to the root.
2. Create a `.env` file from `.env.example`:
   ```bash
   CP .env.example .env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the engine:
   ```bash
   python main.py
   ```

### Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Launch the console:
   ```bash
   npm run dev
   ```

---

## 📖 Usage Guide

1.  **Ingestion**: Drag and drop policy PDFs or provide a URL in the Intelligence Console.
2.  **Graph Construction**: The system will automatically parse and build a local knowledge structure.
3.  **Inquiry**: Ask complex questions (e.g., *"Does my policy cover pre-existing conditions if I haven't claimed in 12 months?"*).
4.  **Verification**: Review the "Structural Evidence" cards and "Used Evidence" IDs to verify the AI's logic.

---

## ⚖ License & Versioning
**Version**: 1.0.4  
**Philosophy**: Intelligence Beyond Parameters.  
*Nsure AI is designed for institutional accuracy and legal-grade policy analysis.*
