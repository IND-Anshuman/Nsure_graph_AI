#python -m spacy download en_core_web_sm
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os
import io
import re
import pathlib
import requests
import pdfplumber
import trafilatura
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")



@dataclass
class Entity:
    text: str
    label: str
    start: int 
    end: int
    source: str 
    canonical: Optional[str]= None
    description: Optional[str]= None
    context: Optional[str]= None
    doc_id: Optional[str]= None

@dataclass
class KGNode:
    id: str                       # e.g., "ent:neo4j", "doc:paper1"
    label: str                    # ENTITY, DOCUMENT, SENTENCE, COMMUNITY
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KGEdge:
    id:str
    source: str 
    target: str 
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, KGNode] = field(default_factory=dict)
    edges: List[KGEdge] = field(default_factory=list)

    def add_node(self, node: KGNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge):
        self.edges.append(edge)



@dataclass
class SentenceInfo:
    sent_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


# ---------- TEXT CLEANING HELPERS ----------

def _clean_whitespace(text: str) -> str:
    # Remove excessive whitespace, normalize newlines
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    text = re.sub(r"\r\n?", "\n", text)  # normalize CRLF -> LF
    text = re.sub(r"[ \t]+", " ", text)  # collapse spaces and tabs
    text = re.sub(r"\n{2,}", "\n\n", text)  # max 2 consecutive newlines
    return text.strip()


# ---------- PDF EXTRACTION ----------

def _extract_text_from_pdf_bytes(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages_text = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        text = "\n\n".join(pages_text)
    return _clean_whitespace(text)


def _extract_text_from_pdf_file(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return _extract_text_from_pdf_bytes(data)


# ---------- HTML / WEB PAGE EXTRACTION ----------

def _extract_text_with_trafilatura(url: str, html: str | None = None) -> str | None:
    """
    Try trafilatura's main-content extractor.
    If html is given, use it; otherwise trafilatura will fetch.
    """
    if html is None:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    else:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        return None
    return _clean_whitespace(text)


def _extract_text_from_html_fallback(html: str) -> str:
    """
    If trafilatura fails, use a simple BeautifulSoup fallback.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return _clean_whitespace(text)


def _extract_text_from_url(url: str) -> str:
    # Add User-Agent to avoid 403 Forbidden errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # If it's a PDF URL, download and treat as PDF
    if url.lower().endswith(".pdf"):
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        return _extract_text_from_pdf_bytes(resp.content)

    # Otherwise treat as HTML / web page
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    html = resp.text

    # Try trafilatura first (better main-content extraction)
    text = _extract_text_with_trafilatura(url, html)
    if text:
        return text

    # Fallback: soup-based full text
    return _extract_text_from_html_fallback(html)


# ---------- SOURCE DISPATCHER ----------

def _is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def build_corpus_from_sources(sources: List[str]) -> Dict[str, str]:
    """
    Robust replacement for simple_corpus().

    Parameters
    ----------
    sources : list of str
        Each item can be:
        - Local PDF file path
        - HTTP/HTTPS URL to a PDF or HTML page

    Returns
    -------
    corpus : Dict[str, str]
        {doc_id: text} where:
        - doc_id is derived from filename or URL slug
        - text is the cleaned raw text
    """
    corpus: Dict[str, str] = {}

    for src in sources:
        if _is_url(src):
            text = _extract_text_from_url(src)
            # Derive doc_id from URL path
            name = src.rstrip("/").split("/")[-1] or "index"
            name = name.split("?")[0] or "index"
            doc_id = pathlib.Path(name).stem or "web_doc"
        else:
            # Local file
            path = os.path.abspath(src)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            ext = pathlib.Path(path).suffix.lower()
            if ext == ".pdf":
                text = _extract_text_from_pdf_file(path)
            else:
                # If you want to support .txt, .md etc., read as text
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text = _clean_whitespace(text)

            doc_id = pathlib.Path(path).stem

        corpus[doc_id] = text

    return corpus


# ---------- SENTENCE SPLITTING (unchanged) ----------

def add_document_and_sentence_nodes(
    graph: KnowledgeGraph,
    corpus: Dict[str, str],
) -> Dict[str, SentenceInfo]:
    """
    Create DOCUMENT and SENTENCE nodes using spaCy sentence splitter.
    Returns a sentence index {sent_id: SentenceInfo}.
    """
    sent_index: Dict[str, SentenceInfo] = {}

    for doc_id, text in corpus.items():
        # Document node
        doc_node_id = f"doc:{doc_id}"
        graph.add_node(
            KGNode(
                id=doc_node_id,
                label="DOCUMENT",
                properties={"doc_id": doc_id, "text": text},
            )
        )

        # Sentence nodes
        doc = nlp(text)
        for i, sent in enumerate(doc.sents):
            sent_id = f"sent:{doc_id}:{i}"
            sent_index[sent_id] = SentenceInfo(
                sent_id=sent_id,
                doc_id=doc_id,
                text=sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char,
            )
            graph.add_node(
                KGNode(
                    id=sent_id,
                    label="SENTENCE",
                    properties={
                        "doc_id": doc_id,
                        "index": i,
                        "text": sent.text,
                    },
                )
            )

    return sent_index
