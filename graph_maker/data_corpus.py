#python -m spacy download en_core_web_sm
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import os
import io
import re
import pathlib
try:
    import requests  # type: ignore
    from requests import RequestException  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    RequestException = Exception  # type: ignore

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover
    trafilatura = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

nlp = None
if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None



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


def prune_graph(graph: KnowledgeGraph, min_degree: int = 1, preserve_labels: Optional[Set[str]] = None) -> None:
    """Remove orphan/low-degree nodes to keep the KG compact."""
    preserve_labels = preserve_labels or set()
    degree: Dict[str, int] = {}
    for e in graph.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1

    to_remove = []
    for node_id, node in graph.nodes.items():
        if node.label in preserve_labels:
            continue
        if degree.get(node_id, 0) < min_degree:
            to_remove.append(node_id)

    if not to_remove:
        return

    for node_id in to_remove:
        graph.nodes.pop(node_id, None)
    graph.edges = [e for e in graph.edges if e.source not in to_remove and e.target not in to_remove]



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


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for dedup checks."""
    cleaned = _clean_whitespace(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------- PDF EXTRACTION ----------

def _extract_text_from_pdf_bytes(data: bytes) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for PDF ingestion. Install requirements.txt.")
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
    if trafilatura is None:
        return None
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
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is required for HTML fallback extraction. Install requirements.txt.")
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return _clean_whitespace(text)


def _extract_text_from_url(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests is required for URL ingestion. Install requirements.txt.")
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

    def _url_basename(u: str) -> str:
        name = u.rstrip("/").split("/")[-1] or "index"
        name = name.split("?")[0] or "index"
        return name

    def _find_cached_url_file(u: str) -> Optional[str]:
        """Try to find a cached local copy for a URL.

        Looks in:
        - env KG_SOURCES_CACHE_DIR (if set)
        - ./outputs
        - current working directory
        """
        base = _url_basename(u)
        # For safety, avoid creating directories implicitly here.
        search_dirs: List[str] = []
        cache_dir = os.getenv("KG_SOURCES_CACHE_DIR")
        if cache_dir:
            search_dirs.append(cache_dir)
        search_dirs.append(os.path.join(os.getcwd(), "outputs"))
        search_dirs.append(os.getcwd())

        for d in search_dirs:
            try:
                candidate = os.path.join(d, base)
            except Exception:
                continue
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate
        return None

    def _maybe_cache_pdf_bytes(u: str, data: bytes) -> None:
        """Best-effort cache of downloaded PDFs for offline reruns."""
        try:
            if not u.lower().endswith(".pdf"):
                return
            out_dir = os.getenv("KG_SOURCES_CACHE_DIR") or os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            filename = _url_basename(u)
            path = os.path.join(out_dir, filename)
            # Don't overwrite if already exists.
            if os.path.exists(path):
                return
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            # Cache failures should never fail ingestion.
            return

    for src in sources:
        if _is_url(src):
            # Derive doc_id from URL path early (also used by cache lookup).
            name = _url_basename(src)
            doc_id = pathlib.Path(name).stem or "web_doc"

            try:
                # Special-case: if it's a PDF, fetch bytes so we can optionally cache them.
                if src.lower().endswith(".pdf"):
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    resp = requests.get(src, timeout=30, headers=headers)
                    resp.raise_for_status()
                    _maybe_cache_pdf_bytes(src, resp.content)
                    text = _extract_text_from_pdf_bytes(resp.content)
                else:
                    text = _extract_text_from_url(src)
            except RequestException as e:
                cached = _find_cached_url_file(src)
                if cached and os.path.exists(cached):
                    ext = pathlib.Path(cached).suffix.lower()
                    if ext == ".pdf":
                        text = _extract_text_from_pdf_file(cached)
                    else:
                        with open(cached, "r", encoding="utf-8", errors="ignore") as f:
                            text = _clean_whitespace(f.read())
                else:
                    raise RuntimeError(
                        "Failed to fetch URL source and no local cache was found. "
                        f"URL: {src}\n"
                        "To run offline, download the file and add it as a local path in main_pipeline.py sources, "
                        "or place it in ./outputs (or set KG_SOURCES_CACHE_DIR)."
                    ) from e
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

        corpus[doc_id] = _dedup_paragraphs(text)

    return corpus


def _dedup_paragraphs(text: str, min_len: int = 60, overlap_threshold: float = 0.92) -> str:
    """Remove near-duplicate paragraphs to reduce noisy edges."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: List[str] = []
    seen: List[Set[str]] = []

    for p in paragraphs:
        if len(p) < min_len:
            continue
        tokens = set(_normalize_text(p).split())
        if not tokens:
            continue
        duplicate = False
        for prev in seen:
            overlap = len(tokens & prev) / max(1, len(tokens | prev))
            if overlap >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(p)
            seen.append(tokens)

    if not kept:
        return text
    return "\n\n".join(kept)


# ---------- SENTENCE SPLITTING (unchanged) ----------

def add_document_and_sentence_nodes(
    graph: KnowledgeGraph,
    corpus: Dict[str, str],
    min_sentence_chars: int = 25,
    dedup_overlap_threshold: float = 0.9,
) -> Dict[str, SentenceInfo]:
    """
    Create DOCUMENT and SENTENCE nodes using spaCy sentence splitter.
    Returns a sentence index {sent_id: SentenceInfo}.
    """
    sent_index: Dict[str, SentenceInfo] = {}

    if nlp is None:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required for sentence splitting. "
            "Run: python -m spacy download en_core_web_sm"
        )

    spacy_n_process = int(os.getenv("KG_SPACY_N_PROCESS", "1") or 1)
    spacy_batch_size = int(os.getenv("KG_SPACY_BATCH_SIZE", "0") or 0)

    items = list(corpus.items())

    # Create document nodes first so sentence insertion can be streamed.
    for doc_id, text in items:
        doc_node_id = f"doc:{doc_id}"
        graph.add_node(
            KGNode(
                id=doc_node_id,
                label="DOCUMENT",
                properties={"doc_id": doc_id, "text": text},
            )
        )

    # Use nlp.pipe only when explicitly enabled and there are multiple documents.
    use_pipe = spacy_n_process > 1 and len(items) > 1
    if use_pipe:
        pipe_kwargs = {"n_process": spacy_n_process}
        if spacy_batch_size and spacy_batch_size > 0:
            pipe_kwargs["batch_size"] = spacy_batch_size
        spacy_docs = nlp.pipe((text for _, text in items), **pipe_kwargs)
    else:
        spacy_docs = (nlp(text) for _, text in items)

    for (doc_id, _text), doc in zip(items, spacy_docs):
        doc_node_id = f"doc:{doc_id}"
        seen_sentence_tokens: List[Set[str]] = []

        # Sentence nodes
        for i, sent in enumerate(doc.sents):
            cleaned_sent = sent.text.strip()
            if len(cleaned_sent) < min_sentence_chars:
                # Some legal PDFs encode provision headings as extremely short
                # sentences (e.g., "3." or "31A.") which are critical anchors
                # for PROVISION mentions and PROVISION_CONTEXT windows.
                keep_short = False

                # Bare numbered heading at line start: 3. / 3.— / 31A.
                if re.match(r"^\s*\d{1,3}[A-Za-z]?\s*[\.\u2013\u2014\-]{1,2}\s*$", cleaned_sent):
                    keep_short = True

                # Explicit legal citation without surrounding text: Article 3
                if not keep_short and re.match(
                    r"^\s*(?:Article|Section|Clause|Chapter|Part|Schedule|Rule|Regulation)\s+\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*\s*$",
                    cleaned_sent,
                    flags=re.IGNORECASE,
                ):
                    keep_short = True

                if not keep_short:
                    continue

            tokens = set(_normalize_text(cleaned_sent).split())
            if tokens:
                duplicate = False
                for prev in seen_sentence_tokens:
                    overlap = len(tokens & prev) / max(1, len(tokens | prev))
                    if overlap >= dedup_overlap_threshold:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_sentence_tokens.append(tokens)

            sent_id = f"sent:{doc_id}:{i}"
            sent_index[sent_id] = SentenceInfo(
                sent_id=sent_id,
                doc_id=doc_id,
                text=cleaned_sent,
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
                        "text": cleaned_sent,
                    },
                )
            )

            # Evidence edge back to document for traceability
            graph.add_edge(
                KGEdge(
                    id=f"e:doc_sent:{doc_id}:{i}",
                    source=doc_node_id,
                    target=sent_id,
                    type="HAS_SENTENCE",
                    properties={"doc_id": doc_id, "index": i},
                )
            )

    return sent_index
