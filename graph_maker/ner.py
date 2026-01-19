from __future__ import annotations
# ai_setup.py

import os
from dataclasses import dataclass, field
from dataclasses import asdict
from functools import lru_cache
import logging
from dotenv import load_dotenv
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
from typing import List, Dict, Set, Tuple, Optional, Any
# NLP & Embeddings (Lazy Loaded)
_NLP = None
_EMB_MODEL = None
from sklearn.cluster import MiniBatchKMeans
import json
import re
import numpy as np
from graph_maker.data_corpus import Entity, KGEdge, KnowledgeGraph, KGNode ,SentenceInfo
from utils.genai_compat import generate_text as genai_generate_text
from utils.cache_utils import DiskJSONCache
from itertools import combinations
import hashlib
import math
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from bisect import bisect_right

def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


load_dotenv()

_RELATION_SCHEMA_CACHE: Dict[str, Any] | None = None
_RELATION_SCHEMA_LOCK = threading.Lock()


# Small, fast embedding model
_EMB_CACHE = DiskJSONCache("cache_embeddings.json")
_CLUSTER_LABEL_CACHE = DiskJSONCache("cache_cluster_labels.json")
_ENTITIES_CACHE = DiskJSONCache("cache_entities.json")
_ONESHOT_RELATION_CACHE = DiskJSONCache("cache_oneshot_relations.json")
_SEMANTIC_REL_CACHE = DiskJSONCache("cache_semantic_relations.json")


def get_emb_model():
    """Lazy-load the local embedding model."""
    global _EMB_MODEL
    if _EMB_MODEL is not None:
        return _EMB_MODEL
    
    # If Gemini provider is used, we don't need the local model at all.
    provider = (os.getenv("KG_EMBEDDING_PROVIDER", "local") or "local").strip().lower()
    if provider == "gemini":
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("KG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        local_only = (os.getenv("KG_EMBEDDING_LOCAL_ONLY", "1") or "1").strip() != "0"
        _EMB_MODEL = SentenceTransformer(model_name, local_files_only=local_only)
        return _EMB_MODEL
    except Exception as exc:
        logging.warning("Embedding model unavailable (%s); will use hashing fallback embeddings.", exc)
        return None


def _hash_embed(texts: List[str], *, dim: int = 384) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = (t or "").lower()
        if not s:
            continue
        for tok in s.split():
            h = hashlib.sha256(tok.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(h[:4], "little", signed=False) % dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            mat[i, idx] += sign
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


def _normalize_surface(surface: str) -> str:
    """Fast normalization used for cache keys/canonicalization.

    Default path avoids running spaCy on every short surface form (major speed win).
    Set env `KG_NORMALIZE_LEMMAS=1` to enable lemma-based normalization.
    """
    s = (surface or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    if os.getenv("KG_NORMALIZE_LEMMAS", "0") == "1":
        try:
            doc = _get_nlp()(s)
            lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
            s = (" ".join(lemmas) or s).strip()
        except Exception:
            # If spaCy fails for any reason, fall back to simple normalization.
            pass
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _text_digest(text: str) -> str:
    # Hash text without storing it in cache keys.
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _find_acronym_pairs(text: str) -> Dict[str, str]:
    """Return mapping of acronym -> longform for patterns like 'Long Form (LF)'"""
    pairs: Dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z][A-Za-z\s]{3,})\s*\(([A-Z]{2,})\)")
    for match in pattern.finditer(text):
        long_form = match.group(1).strip()
        acronym = match.group(2).strip()
        pairs[acronym] = long_form
    return pairs


# -------------------------
# Utilities
# -------------------------
def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def _parse_json_safely(raw: str, default):
    """Best-effort JSON parsing that tolerates fenced blocks and extra text.

    - Strips triple backtick fences and leading 'json'
    - Tries full parse
    - Then tries to extract first top-level JSON object or array
    - Returns `default` on failure
    """
    try:
        cleaned = _strip_code_fences(raw)
        if not cleaned:
            return default
        # First attempt: direct parse
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        cleaned = _strip_code_fences(raw)
        # Try to find a JSON object
        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")

        candidates = []
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(cleaned[obj_start:obj_end+1])
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(cleaned[arr_start:arr_end+1])

        for cand in candidates:
            try:
                # Try simple parse
                return json.loads(cand)
            except Exception:
                # Try repair if it looks like a truncated object/list
                try:
                    repaired = _robust_json_repair(cand)
                    return json.loads(repaired)
                except Exception:
                    continue
    except Exception:
        pass

    return default


def _robust_json_repair(s: str) -> str:
    """Attempts to fix truncated JSON by closing open braces/brackets/quotes.
    
    Extremely simple heuristic for truncated LLM responses.
    """
    s = s.strip()
    if not s:
        return s
    
    # If it ends with a comma, strip it
    if s.endswith(","):
        s = s[:-1].strip()
        
    # Stack-based closure
    stack = []
    in_string = False
    escaped = False
    
    for char in s:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack:
                    top = stack[-1]
                    if (char == "}" and top == "{") or (char == "]" and top == "["):
                        stack.pop()
                        
    # Close open string
    if in_string:
        s += '"'
        
    # Close stack in reverse
    while stack:
        top = stack.pop()
        if top == "{":
            s += "}"
        elif top == "[":
            s += "]"
            
    return s



@dataclass
class EntityCatalogEntry:
    canonical: str
    label: str
    aliases: Set[str] = field(default_factory=set)
    descriptions: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

def get_embeddings_with_cache(texts: List[str]) -> np.ndarray:
    """Compatibility shim.

    Route embedding computation to the centralized embedding cache.
    This keeps embedding values consistent while enabling a much faster cache backend.
    """
    from graph_maker.embedding_cache import get_embeddings_with_cache as _get

    return _get(texts)


_GENERIC_ENTITY_STOPWORDS: Set[str] = {
    # Common high-degree hubs in legal/technical corpora that often add noise
    # to co-occurrence edges. We avoid filtering them out of the KG entirely;
    # we just use this list as a signal for down-weighting/edge filtering.
    "act",
    "acts",
    "law",
    "laws",
    "rule",
    "rules",
    "court",
    "courts",
    "government",
    "state",
    "union",
    "constitution",
    "section",
    "article",
    "clause",
    "schedule",
    "chapter",
    "part",

    # Finance / insurance / legal boilerplate hubs
    "policy",
    "policies",
    "insurer",
    "insurers",
    "insured",
    "insureds",
    "claim",
    "claims",
    "premium",
    "premiums",
    "liability",
    "liabilities",
    "loss",
    "losses",
    "damage",
    "damages",
    "party",
    "parties",
    "third party",
    "contract",
    "agreement",
    "terms",
    "conditions",
    "coverage",
    "covered",
    "exclusion",
    "exclusions",
    "exception",
    "exceptions",
    "endorsement",
    "endorsements",
    "schedule",
    "annexure",
    "appendix",
    "regulator",
    "authority",
    "bank",
    "company",
    "corporation",
}


def _iter_text_chunks(text: str, max_chunk_chars: int) -> List[Tuple[int, int, str]]:
    """Split large text into manageable chunks without losing coverage.

    Returns list of (start_char, end_char, chunk_text) in original document coordinates.

    Strategy:
    - Prefer paragraph boundaries (\n\n)
    - Fall back to sliding windows if a paragraph is too large
    """
    t = text or ""
    if not t:
        return []
    if max_chunk_chars <= 0 or len(t) <= max_chunk_chars:
        return [(0, len(t), t)]

    chunks: List[Tuple[int, int, str]] = []
    paras: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\n+", t):
        end = m.start()
        if end > start:
            paras.append((start, end))
        start = m.end()
    if start < len(t):
        paras.append((start, len(t)))

    for ps, pe in paras:
        if pe - ps <= max_chunk_chars:
            chunk = t[ps:pe]
            chunks.append((ps, pe, chunk))
            continue

        # Paragraph is too large: split with an overlap.
        overlap = min(600, max_chunk_chars // 6)
        step = max(1, max_chunk_chars - overlap)
        i = ps
        while i < pe:
            j = min(pe, i + max_chunk_chars)
            chunks.append((i, j, t[i:j]))
            if j >= pe:
                break
            i = j - overlap

    return chunks


_LEGAL_REF_RE = re.compile(
    r"\b(?:Article|Section|Clause|Chapter|Part|Schedule|Rule|Regulation)\s+"  # keyword
    r"(?:\d+[A-Za-z]?)(?:\([0-9A-Za-z]+\))*"  # number + optional subsections
    r"\b",
    flags=re.IGNORECASE,
)

# --- Constitution-aware structural entities (India Constitution PDFs) ---
_ARTICLE_REF_RE = re.compile(
    r"\bArticle\s+(?P<num>\d{1,3})(?P<suffix>[A-Za-z]?)\b",
    flags=re.IGNORECASE,
)

# Many Constitution PDFs format Articles as numbered headings like:
#   "3. Formation of new States and alteration of areas, boundaries or names of existing States."
# rather than "Article 3". Capture these headings at line start.
_ARTICLE_HEADING_RE = re.compile(
    r"(?m)^\s*(?P<num>\d{1,3})(?P<suffix>[A-Za-z]?)\s*(?:[\.\-\u2013\u2014]{1,2})\s*(?P<title>[^\n]{4,160})$",
    flags=re.IGNORECASE,
)

_PART_HEADING_RE = re.compile(
    r"(?m)^\s*PART\s+(?P<roman>[IVXLCDM]{1,7})\s*(?:\—|\-|\:)?\s*(?P<title>[^\n]{0,80})$",
    flags=re.IGNORECASE,
)

_SCHEDULE_HEADING_RE = re.compile(
    r"(?m)^\s*(?P<ord>(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH))\s+SCHEDULE\b.*$",
    flags=re.IGNORECASE,
)

_INSTITUTION_SURFACES: List[Tuple[str, str]] = [
    (r"\bPresident\b", "president"),
    (r"\bParliament\b", "parliament"),
    (r"\bState\s+Legislature\b", "state legislature"),
    (r"\bLegislature\s+of\s+the\s+State\b", "state legislature"),
    (r"\bLegislature\s+of\s+each\s+of\s+the\s+affected\s+States\b", "affected state legislature"),
    (r"\bGovernor\b", "governor"),
]

_LEGAL_CONCEPT_SURFACES: List[Tuple[str, str]] = [
    (r"\bFundamental\s+Rights\b", "fundamental rights"),
    (r"\bDirective\s+Principles\s+of\s+State\s+Policy\b", "directive principles of state policy"),
    (r"\bDirective\s+Principles\b", "directive principles"),
    (r"\bDPSP\b", "directive principles"),
]

_PROCEDURAL_REQ_SURFACES: List[Tuple[str, str]] = [
    (r"\brecommendation\s+of\s+the\s+President\b", "presidential recommendation"),
    (r"\brefer\s+the\s+Bill\s+to\s+the\s+Legislature\b", "legislature consultation"),
    (r"\bfor\s+expressing\s+its\s+views\b", "views of affected state"),
    (r"\bconsultation\b", "consultation"),
]


def _ord_to_key(word: str) -> str:
    w = _normalize_surface(word)
    # keep common ordinal words as-is (first, second, ...)
    return w.replace(" ", "_")


def _extract_constitution_structure_entities(text: str, doc_id: str) -> List[Entity]:
    """Extract constitution-structure entities deterministically.

    This is intentionally rule-based and conservative: it creates first-class anchors
    for ARTICLE / PART / SCHEDULE / INSTITUTION / LEGAL_CONCEPT / PROCEDURAL_REQUIREMENT.

    These anchors prevent catastrophic collapse into generic entities like `ent:article`.
    """
    if not text:
        return []

    out: List[Entity] = []

    # Articles (explicit "Article 3" form)
    for m in _ARTICLE_REF_RE.finditer(text):
        num = m.group("num")
        suffix = (m.group("suffix") or "").lower()
        canon = f"article {num}{suffix}".strip()
        out.append(
            Entity(
                text=m.group(0),
                label="ARTICLE",
                start=m.start(),
                end=m.end(),
                source="constitution_regex",
                canonical=_normalize_surface(canon),
                description=None,
                context=text[max(0, m.start()-100):min(len(text), m.end()+160)],
                doc_id=doc_id,
            )
        )

    # Articles (numbered heading form: "3." / "3—" at line start)
    for m in _ARTICLE_HEADING_RE.finditer(text):
        title = (m.group("title") or "").strip()
        # Heuristic: avoid pulling in tiny numbered bullets; require some alphabetic content.
        if len(re.findall(r"[A-Za-z]", title)) < 6:
            continue
        num = m.group("num")
        suffix = (m.group("suffix") or "").lower()
        canon = f"article {num}{suffix}".strip()
        out.append(
            Entity(
                text=m.group(0),
                label="ARTICLE",
                start=m.start(),
                end=m.end(),
                source="constitution_regex",
                canonical=_normalize_surface(canon),
                description=title if title else None,
                context=text[max(0, m.start()-60):min(len(text), m.end()+220)],
                doc_id=doc_id,
            )
        )

    # Parts (PART III — FUNDAMENTAL RIGHTS)
    for m in _PART_HEADING_RE.finditer(text):
        roman = (m.group("roman") or "").strip().lower()
        title = (m.group("title") or "").strip()
        canon = f"part {roman}".strip()
        desc = title if title else None
        out.append(
            Entity(
                text=m.group(0),
                label="PART",
                start=m.start(),
                end=m.end(),
                source="constitution_regex",
                canonical=_normalize_surface(canon),
                description=desc,
                context=text[max(0, m.start()-60):min(len(text), m.end()+120)],
                doc_id=doc_id,
            )
        )

    # Schedules (FIRST SCHEDULE)
    for m in _SCHEDULE_HEADING_RE.finditer(text):
        ord_word = (m.group("ord") or "").strip().lower()
        canon = f"{ord_word} schedule".strip()
        out.append(
            Entity(
                text=m.group(0),
                label="SCHEDULE",
                start=m.start(),
                end=m.end(),
                source="constitution_regex",
                canonical=_normalize_surface(canon),
                description=None,
                context=text[max(0, m.start()-60):min(len(text), m.end()+120)],
                doc_id=doc_id,
            )
        )

    # Institutions / actors
    for pat, canon in _INSTITUTION_SURFACES:
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
        except Exception:
            continue
        for m in rx.finditer(text):
            out.append(
                Entity(
                    text=m.group(0),
                    label="INSTITUTION",
                    start=m.start(),
                    end=m.end(),
                    source="constitution_regex",
                    canonical=_normalize_surface(canon),
                    description=None,
                    context=text[max(0, m.start()-80):min(len(text), m.end()+140)],
                    doc_id=doc_id,
                )
            )

    # High-value constitutional concepts
    for pat, canon in _LEGAL_CONCEPT_SURFACES:
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
        except Exception:
            continue
        for m in rx.finditer(text):
            out.append(
                Entity(
                    text=m.group(0),
                    label="LEGAL_CONCEPT",
                    start=m.start(),
                    end=m.end(),
                    source="constitution_regex",
                    canonical=_normalize_surface(canon),
                    description=None,
                    context=text[max(0, m.start()-80):min(len(text), m.end()+140)],
                    doc_id=doc_id,
                )
            )

    # Procedural requirement anchors
    for pat, canon in _PROCEDURAL_REQ_SURFACES:
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
        except Exception:
            continue
        for m in rx.finditer(text):
            out.append(
                Entity(
                    text=m.group(0),
                    label="PROCEDURAL_REQUIREMENT",
                    start=m.start(),
                    end=m.end(),
                    source="constitution_regex",
                    canonical=_normalize_surface(canon),
                    description=None,
                    context=text[max(0, m.start()-80):min(len(text), m.end()+140)],
                    doc_id=doc_id,
                )
            )

    return out


# Some legal PDFs (including constitutions) format provision headings as just a
# number at the start of a line (e.g. `3.` or `3.—`) rather than `Article 3`.
# We synthesize canonical forms like `article 3` but anchor spans to the real
# heading text so mentions and PROVISION_CONTEXT windows work.
_NUMBERED_PROVISION_HEADING_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]{0,3})(?P<num>\d{1,3})(?P<suffix>[A-Za-z]?)\s*(?:[\.\u2013\u2014\-]{1,2})\s*(?=(?:\(|[A-Za-z]))"
)


def _extract_numbered_provision_headings(text: str, doc_id: str) -> List[Entity]:
    """Create PROVISION entities for numbered heading styles like `3.`.

    Guardrails:
    - Enabled only when the document appears to be a constitution (heuristic)
    - Caps total headings per doc to avoid exploding on enumerated lists
    """
    t = text or ""
    if not t.strip():
        return []

    if os.getenv("KG_ENABLE_NUMBERED_ARTICLE_HEADINGS", "1") != "1":
        return []

    # Heuristic: only enable for constitutions (reduces false positives in other corpora).
    looks_like_constitution = bool(re.search(r"\bconstitution\b", t, flags=re.IGNORECASE))
    looks_like_constitution = looks_like_constitution or bool(re.search(r"constitution", str(doc_id or ""), flags=re.IGNORECASE))
    if not looks_like_constitution:
        return []

    max_headings = int(os.getenv("KG_MAX_NUMBERED_ARTICLE_HEADINGS", "240"))
    out: List[Entity] = []
    per_canon_cap = 8
    canon_counts: Dict[str, int] = {}

    for m in _NUMBERED_PROVISION_HEADING_RE.finditer(t):
        if max_headings > 0 and len(out) >= max_headings:
            break

        try:
            n = int(m.group("num"))
        except Exception:
            continue
        if n <= 0 or n > 450:
            continue

        suffix = (m.group("suffix") or "").strip()
        canon = f"article {n}{suffix.lower()}".strip()

        # avoid runaway duplicates (e.g., repeated headers/footers)
        if canon_counts.get(canon, 0) >= per_canon_cap:
            continue
        canon_counts[canon] = canon_counts.get(canon, 0) + 1

        start = m.start("num")
        end = m.end()
        if end <= start:
            continue

        out.append(
            Entity(
                text=t[start:end],
                label="PROVISION",
                start=start,
                end=end,
                source="rule_numbered_heading",
                canonical=canon,
                description=None,
                context=t[max(0, start - 120) : min(len(t), end + 200)],
                doc_id=doc_id,
            )
        )

    return out


_LEGAL_ID_CANON_RE = re.compile(
    r"^(article|section|clause|chapter|part|schedule|rule|regulation)\s+\d+[a-z]?(?:\([0-9a-z]+\))*$",
    flags=re.IGNORECASE,
)


def _is_legal_id_like(surface_or_canon: str) -> bool:
    return bool(_LEGAL_ID_CANON_RE.match(_normalize_surface(surface_or_canon or "")))


def _expand_legal_ranges(text: str) -> List[Tuple[str, int]]:
    """Expand patterns like 'Articles 31A–31C' into Article 31A/31B/31C."""
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    rx = re.compile(
        r"\bArticles?\s+(\d+)([A-Za-z])\s*[\-\u2013\u2014]\s*(\d+)?([A-Za-z])\b",
        flags=re.IGNORECASE,
    )
    for m in rx.finditer(text):
        base_num_1 = m.group(1)
        letter1 = (m.group(2) or "").upper()
        base_num_2 = m.group(3) or base_num_1
        letter2 = (m.group(4) or "").upper()

        if base_num_1 != base_num_2:
            continue
        if not (letter1.isalpha() and letter2.isalpha()):
            continue

        a = ord(letter1)
        b = ord(letter2)
        if a > b:
            a, b = b, a
        if b - a > 10:
            continue

        for code in range(a, b + 1):
            out.append((f"Article {base_num_1}{chr(code)}", 6))

    return out


_DEFINED_TERM_RE_LIST: List[re.Pattern] = [
    # “X” means ... / 'X' means ...
    re.compile(r"(?:^|\s)[\"']([^\"']{2,80}?)[\"']\s+(?:shall\s+)?means\b", flags=re.IGNORECASE),
    # X means ... (Title Case or Capitalized phrase)
    re.compile(r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,6})\s+(?:shall\s+)?mean(?:s)?\b", flags=re.IGNORECASE),
    # referred to as X
    re.compile(r"\breferred\s+to\s+as\s+[\"']?([^\"'\n\r]{2,80})[\"']?", flags=re.IGNORECASE),
]


def _extract_rule_candidates(text: str) -> List[Tuple[str, int]]:
    """High-recall, low-cost regex-based candidates."""
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    for m in _LEGAL_REF_RE.finditer(text):
        out.append((m.group(0).strip(), 4))

    # Expand legal ranges (e.g., Articles 31A–31C)
    out.extend(_expand_legal_ranges(text))

    for pat in _DEFINED_TERM_RE_LIST:
        for m in pat.finditer(text):
            term = (m.group(1) or "").strip()
            if term:
                out.append((term, 6))

    # Capitalized multiword sequences (often organizations / named concepts)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b", text):
        out.append((m.group(1).strip(), 2))

    return out


def _compute_adaptive_candidate_budget(text: str) -> int:
    """Adaptive max candidate budget; can still be overridden via env."""
    explicit = os.getenv("KG_MAX_ENTITY_CANDIDATES")
    if explicit is not None and explicit.strip() != "":
        try:
            return max(0, int(explicit))
        except Exception:
            pass

    t = text or ""
    # Allow more candidates for longer documents without exploding.
    base = 350
    # ~ +150 for ~200k chars, +250 for ~1M chars (log-ish growth)
    extra = int(120 * math.log10(max(10, len(t)) / 10000.0 + 1.0))
    return int(max(200, min(1400, base + extra)))


def extract_candidate_phrases(text: str) -> List[str]:
    """Extract high-recall candidate entity surfaces from the full document.

    Improvements vs the earlier version:
    - No head/middle/tail sampling (coverage is preserved)
    - Chunking keeps spaCy bounded
    - Adds rule-based candidates (definitions, legal refs, capitalized phrases)
    - Adaptive budget instead of a fixed hard ceiling
    """
    t = text or ""
    if not t.strip():
        return []

    max_chars = int(os.getenv("KG_ENTITY_EXTRACT_MAX_CHARS", "200000"))
    # Interpret as per-chunk cap (not total doc cap).
    max_chunk_chars = max(5000, min(max_chars, 45000)) if max_chars > 0 else 25000
    chunks = _iter_text_chunks(t, max_chunk_chars=max_chunk_chars)

    # norm -> (best_surface, score, freq)
    candidates: Dict[str, Tuple[str, int, int]] = {}

    acronym_map = _find_acronym_pairs(t)

    def _try_add(raw: str, score_add: int = 1):
        cleaned = (raw or "").strip()
        if not cleaned or len(cleaned) > 120 or len(cleaned) < 2:
            return
        norm = _normalize_surface(cleaned)
        if not norm:
            return
        # avoid exploding with extremely generic terms
        if len(norm) <= 2:
            return
        prev = candidates.get(norm)
        if prev is None:
            candidates[norm] = (cleaned, score_add, 1)
        else:
            surface, score, freq = prev
            candidates[norm] = (surface, score + score_add, freq + 1)

    # Rule-based candidates from full text (cheap)
    for surface, score in _extract_rule_candidates(t):
        _try_add(surface, score_add=score)

    # spaCy-based candidates per chunk
    # Use nlp.pipe for batching (and optional multiprocessing) to reduce overhead.
    chunk_texts = [chunk for (_cs, _ce, chunk) in chunks if (chunk or "").strip()]
    if chunk_texts:
        spacy_batch_size = int(os.getenv("KG_SPACY_BATCH_SIZE", "32") or 32)
        spacy_n_process = int(os.getenv("KG_SPACY_N_PROCESS", "1") or 1)
        if spacy_batch_size <= 0:
            spacy_batch_size = 32
        if spacy_n_process <= 0:
            spacy_n_process = 1

        pipe_kwargs: Dict[str, Any] = {"batch_size": spacy_batch_size}
        if spacy_n_process > 1:
            pipe_kwargs["n_process"] = spacy_n_process

        try:
            docs_iter = nlp.pipe(chunk_texts, **pipe_kwargs)
        except TypeError:
            # Older spaCy may not accept n_process.
            pipe_kwargs.pop("n_process", None)
            docs_iter = nlp.pipe(chunk_texts, **pipe_kwargs)
        except Exception:
            docs_iter = (nlp(c) for c in chunk_texts)

        for doc in docs_iter:
            try:
                for chunk_np in doc.noun_chunks:
                    _try_add(chunk_np.text, score_add=1)
                for ent in doc.ents:
                    _try_add(ent.text, score_add=3)
            except Exception:
                continue

    # Acronym expansions (doc-wide)
    for acro, long_form in acronym_map.items():
        _try_add(acro, score_add=3)
        _try_add(long_form, score_add=3)

    # Rank: score first, then freq, then length
    max_candidates = _compute_adaptive_candidate_budget(t)
    ranked = sorted(
        candidates.values(),
        key=lambda x: (x[1], x[2], len(x[0])),
        reverse=True,
    )
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    return [surface for (surface, _score, _freq) in ranked]


CLUSTER_LABEL_PROMPT = """
You are an expert Ontologist building a high-fidelity knowledge graph for LEGAL and INSURANCE domains.

You will receive a cluster of surface forms (strings) that refer to the same concept.

Task:
1. Determine if this cluster represents a meaningful Entity or Domain Concept.
2. If YES:
   - "is_entity": true
   - "label": A precise semantic type. For insurance, use types like:
     POLICY_TYPE, COVERAGE, EXCLUSION, CLAIM_PROCEDURE, REGULATORY_BODY, 
     LEGAL_ACT, FINANCIAL_INSTRUMENT, ORGANIZATION, PERSON, LOCATION,
     PROVISION, PENALTY, LIMITATION, etc.
   - "canonical": The most standard, formal name for this concept (Title Case).
   - "description": A high-quality, 2-3 sentence summary. Include WHAT it is and its significance in the document's context.
3. If NO (junk/parsing artifact):
   - "is_entity": false

Return ONLY a single JSON object:
{
  "is_entity": true,
  "label": "COVERAGE",
  "canonical": "Third Party Liability",
  "description": "Coverage provided to the insured against legal liability for death, bodily injury to third parties or damage to third party property caused by the motor vehicle."
}
"""


def label_cluster_with_llm(cluster_items: List[str]) -> Dict:
    """Label a surface-form cluster using Gemini.

    Design choice: be *permissive* so we don't under-generate entities.
    If the LLM call fails or returns malformed JSON, we treat the cluster
    as an entity by default (is_entity=True).
    """

    # Cache by stable signature to avoid repeated LLM calls across runs.
    signature = json.dumps(sorted(set(cluster_items)), ensure_ascii=False)
    cache_key = DiskJSONCache.hash_key("cluster_label_v2", signature)
    cached = _CLUSTER_LABEL_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return cached

    prompt = f"{CLUSTER_LABEL_PROMPT}\n\nCluster items:\n{signature}"

    raw = ""
    for attempt in range(2):
        try:
            raw = genai_generate_text(None, prompt, temperature=0.1, purpose="ENTITY")
            if raw:
                break
        except Exception as exc:  # pragma: no cover - network / API issues
            logging.warning("label_cluster_with_llm: LLM call failed (attempt %s): %s", attempt + 1, exc)
            raw = ""
            # brief backoff helps with transient rate limits
            time.sleep(1.5 * (attempt + 1))

    # If parsing fails, default to treating the cluster as an entity
    data = _parse_json_safely(raw, default={"is_entity": True})
    if not isinstance(data, dict):
        data = {"is_entity": True}

    # Be generous: unless explicitly false, accept as entity
    data.setdefault("is_entity", True)
    if data["is_entity"] is False:
        return {"is_entity": False}

    # Ensure we always have a coarse label and canonical
    label = data.get("label") or "ENTITY"
    canon = data.get("canonical") or (cluster_items[0] if cluster_items else "")
    canon = _normalize_surface(canon)

    data["label"] = label
    data["canonical"] = canon
    _CLUSTER_LABEL_CACHE.set(cache_key, data)
    return data


def cluster_candidates(candidates: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
    if not candidates:
        return {}

    if len(candidates) == 1:
        return {0: candidates}

    embeddings = get_embeddings_with_cache(candidates)

    # AgglomerativeClustering is O(n^2) and was configured to create ~n/2 clusters,
    # which explodes runtime and triggers far too many LLM label calls.
    n = len(candidates)
    min_clusters = max(2, int(os.getenv("KG_MIN_ENTITY_CLUSTERS", "12")))
    max_clusters = max(2, int(os.getenv("KG_MAX_ENTITY_CLUSTERS", "120")))

    # Higher granularity improves recall on legal/technical corpora.
    # Heuristic: ~n/6 clusters with bounds.
    desired_clusters = int(round(n / 6.0))
    desired_clusters = max(desired_clusters, min_clusters, n_clusters)
    desired_clusters = min(desired_clusters, max_clusters, n)

    if desired_clusters <= 1:
        labels = np.zeros((n,), dtype=int)
    else:
        clustering = MiniBatchKMeans(
            n_clusters=desired_clusters,
            random_state=42,
            n_init=10,
            batch_size=1024,
        )
        labels = clustering.fit_predict(embeddings)

    clusters: Dict[int, List[str]] = {}
    for cand, lbl in zip(candidates, labels):
        clusters.setdefault(int(lbl), []).append(cand)
    return clusters


def _canonical_to_node_id(canonical: str) -> str:
    """Stable KG node id for an entity canonical string.

    Legal-first deterministic IDs (non-negotiable for constitutional QA):
    - Article:  "article 31a" -> "art:31a"
    - Part:     "part iii"    -> "part:iii"
    - Schedule: "first schedule" -> "sched:first"
    - Institution: "president" -> "inst:president"
    - Concepts: "fundamental rights" -> "concept:fundamental_rights"
    Falls back to: "ent:<canonical>".
    """
    canon = _normalize_surface(canonical)
    if not canon:
        return "ent:unknown"

    # Articles / Sections / Clauses (keep close to query phrasing, but IDs are compact)
    m = re.match(r"^(article)\s+(\d{1,3})([a-z]?)((?:\([0-9a-z]+\))*)$", canon, flags=re.IGNORECASE)
    if m:
        num = m.group(2)
        suffix = (m.group(3) or "").lower()
        subs = (m.group(4) or "")
        subs = re.sub(r"[()]+", "_", subs).strip("_")
        tail = f"{num}{suffix}" + (f"_{subs}" if subs else "")
        return f"art:{tail}"

    m = re.match(r"^(section)\s+(\d{1,4})([a-z]?)((?:\([0-9a-z]+\))*)$", canon, flags=re.IGNORECASE)
    if m:
        num = m.group(2)
        suffix = (m.group(3) or "").lower()
        subs = (m.group(4) or "")
        subs = re.sub(r"[()]+", "_", subs).strip("_")
        tail = f"{num}{suffix}" + (f"_{subs}" if subs else "")
        return f"sec:{tail}"

    m = re.match(r"^part\s+([ivxlcdm]{1,7})$", canon, flags=re.IGNORECASE)
    if m:
        return f"part:{m.group(1).lower()}"

    m = re.match(r"^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+schedule$", canon, flags=re.IGNORECASE)
    if m:
        return f"sched:{_ord_to_key(m.group(1))}"

    # Institutions / concepts / procedural requirements
    if canon in {"president", "parliament", "governor", "state legislature", "affected state legislature"}:
        return f"inst:{canon.replace(' ', '_')}"

    if canon in {"fundamental rights", "directive principles", "directive principles of state policy"}:
        return f"concept:{canon.replace(' ', '_')}"

    if canon in {"presidential recommendation", "legislature consultation", "views of affected state", "consultation"}:
        return f"proc:{canon.replace(' ', '_')}"

    # default entity id
    canon = canon.replace(" ", "_")
    return f"ent:{canon}" if canon else "ent:unknown"


def _entity_to_node_id(e: Entity) -> str:
    canon = (e.canonical or e.text or "")
    return _canonical_to_node_id(canon)


def _canon_for_payload(e: Entity) -> str:
    """Canonical string used in LLM payloads (human-readable, space-separated)."""
    canon = _normalize_surface(e.canonical or e.text)
    return canon


@lru_cache(maxsize=50000)
def _pattern_from_surface(surface: str) -> Optional[re.Pattern]:
    s = (surface or "").strip()
    if not s:
        return None
    # Escape regex metacharacters, but make whitespace flexible.
    esc = re.escape(s)
    esc = re.sub(r"\\\s+", r"\\s+", esc)

    # If the surface is alphanumeric-ish, apply word boundaries.
    if re.fullmatch(r"[A-Za-z0-9_\- ]+", s):
        # Word boundary around the whole phrase is tricky; use \b on ends.
        esc = r"\b" + esc + r"\b"
    return re.compile(esc, flags=re.IGNORECASE)


@lru_cache(maxsize=50000)
def _surface_variants(surface: str) -> List[str]:
    """Generate surface variants to improve mention linking in finance/legal text.

    This is intentionally lightweight (regex-only) to avoid token→offset remapping.
    """
    s = (surface or "").strip()
    if not s:
        return []

    # Normalize common quote characters and whitespace.
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip()

    out: Set[str] = {s}

    # Hyphen/space variants: third-party <-> third party
    if "-" in s:
        out.add(s.replace("-", " "))
    if " " in s:
        out.add(s.replace(" ", "-"))

    # Comma variants: Act, 1988 <-> Act 1988
    if "," in s:
        out.add(s.replace(",", " "))

    # Dot spacing: s. 149(2) <-> s 149(2)
    if "." in s:
        out.add(s.replace(".", " "))

    # Ampersand variants: S&P <-> S and P
    if "&" in s:
        out.add(s.replace("&", " and "))
        out.add(s.replace("&", " "))

    # Collapse repeated whitespace in variants.
    cleaned = []
    for v in out:
        v2 = re.sub(r"\s+", " ", v).strip()
        if v2:
            cleaned.append(v2)

    # Prefer longer/more specific first.
    cleaned = sorted(set(cleaned), key=lambda x: (-len(x), x.lower()))
    return cleaned


def find_spans(
    text: str,
    surface: str,
    *,
    max_spans: int,
    word_set: Optional[Set[str]] = None,
) -> List[Tuple[int, int]]:
    """Find multiple occurrences of a surface form in text.

    - Case-insensitive
    - Uses word boundaries for simple alphanumeric phrases
    - Caps results to avoid exploding mentions
    """
    if not text or not surface:
        return []
    spans: List[Tuple[int, int]] = []

    # Optional fast reject: if any meaningful token from the variant cannot
    # exist in the document, skip regex scanning entirely.
    # This is conservative (only rejects when a token is absent), and is meant
    # to reduce the common case of many candidate surfaces that never appear.
    enable_prefilter = (os.getenv("KG_SPAN_PREFILTER", "1") or "1").strip() != "0"

    # Try a small set of robust variants before giving up.
    for variant in _surface_variants(surface):
        if enable_prefilter and word_set is not None:
            try:
                toks = re.findall(r"[a-z0-9]+", (variant or "").lower())
                # Only enforce for tokens length>=3 to avoid rejecting short
                # acronyms/single letters that are noisy to index.
                if any((len(t) >= 3 and t not in word_set) for t in toks):
                    continue
            except Exception:
                pass
        pat = _pattern_from_surface(variant)
        if pat is None:
            continue
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
            if max_spans > 0 and len(spans) >= max_spans:
                return spans

    # Dedupe overlaps while preserving order.
    if len(spans) <= 1:
        return spans
    spans.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int]] = []
    last = (-1, -1)
    for a, b in spans:
        if a == last[0] and b == last[1]:
            continue
        out.append((a, b))
        last = (a, b)
        if max_spans > 0 and len(out) >= max_spans:
            break
    return out


def extract_semantic_entities_for_doc(doc_id: str, text: str) -> List[Entity]:
    """
    Fully automated:
    - extracts candidates
    - clusters them
    - labels clusters via LLM
    - creates Entity objects aligned to text
    """
    # Cache full extraction per document content so reruns are fast.
    # IMPORTANT: bump the version whenever canonicalization / span logic changes,
    # otherwise stale cached entities can collapse legal provisions (e.g., all
    # articles under canonical "article").
    strategy = (os.getenv("KG_EXTRACTION_STRATEGY", "cluster") or "cluster").strip().lower()
    if strategy == "oneshot":
        return extract_oneshot_kg_from_doc(doc_id, text)

    canon_mode = (os.getenv("KG_ENTITY_CANONICAL_MODE", "cluster") or "cluster").strip().lower()
    # Cache key includes canonicalization mode so switching modes takes effect immediately.
    ent_cache_key = DiskJSONCache.hash_key("entities_v6", doc_id, canon_mode, _text_digest(text))
    cached = _ENTITIES_CACHE.get(ent_cache_key)
    if isinstance(cached, list):
        try:
            ents = []
            for item in cached:
                if isinstance(item, dict):
                    ents.append(Entity(**item))
            if ents:
                return ents
        except Exception:
            pass

    # Seed with numbered heading provisions so `ent:article_3` exists even when
    # the PDF uses bare numeric headings like `3.` instead of `Article 3`.
    seeded_entities = []
    try:
        seeded_entities.extend(_extract_numbered_provision_headings(text, doc_id))
    except Exception:
        pass
    try:
        seeded_entities.extend(_extract_constitution_structure_entities(text, doc_id))
    except Exception:
        pass

    # Build one token-set index for the whole document and reuse it for all span lookups.
    # This dramatically reduces time spent scanning text for non-present surfaces.
    doc_word_set: Optional[Set[str]] = None
    if (os.getenv("KG_SPAN_PREFILTER", "1") or "1").strip() != "0":
        try:
            doc_word_set = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        except Exception:
            doc_word_set = None

    candidates = extract_candidate_phrases(text)
    # Increase cluster count for better granularity (richer nodes)
    clusters = cluster_candidates(candidates, n_clusters=int(os.getenv("KG_ENTITY_CLUSTERS", "12") or 12))
    acronym_map = _find_acronym_pairs(text)

    all_entities: List[Entity] = list(seeded_entities)

    # Helper to filter out obviously junk clusters (stopwords/numerics/one-char)
    def _cluster_is_junk(items: List[str]) -> bool:
        for s in items:
            cleaned = _normalize_surface(s)
            if not cleaned:
                continue
            # Skip purely numeric or single-character tokens
            if cleaned.isdigit() or len(cleaned) <= 1:
                continue
            # Has some usable content
            return False
        return True

    disable_llm = os.getenv("KG_DISABLE_LLM_ENTITY_LABELING", "0") == "1"
    # Increase LLM budget for richer entity metadata
    max_llm_clusters = int(os.getenv("KG_MAX_LLM_ENTITY_CLUSTERS", "60") or 60)
    # Prefer labeling larger clusters first (more value per LLM call)
    cluster_items_ranked = sorted(clusters.values(), key=lambda it: len(it), reverse=True)
    llm_allowed_signatures = set(
        DiskJSONCache.hash_key("cluster_sig_v1", json.dumps(sorted(set(items)), ensure_ascii=False))
        for items in (cluster_items_ranked[:max_llm_clusters] if max_llm_clusters > 0 else [])
    )

    # Pre-label LLM-eligible clusters in parallel to reduce wall-clock time.
    # This does not change which clusters are labeled; it only overlaps network latency.
    llm_meta_by_cluster: Dict[int, Dict[str, Any]] = {}
    if (not disable_llm) and llm_allowed_signatures:
        label_workers = int(os.getenv("KG_LLM_LABEL_WORKERS", "2") or 2)
        if label_workers < 1:
            label_workers = 1

        def _default_meta(items: List[str]) -> Dict[str, Any]:
            return {
                "is_entity": True,
                "label": "ENTITY",
                "canonical": _normalize_surface(items[0] if items else ""),
                "description": None,
            }

        if label_workers > 1:
            with ThreadPoolExecutor(max_workers=label_workers) as ex:
                futs = {}
                for cluster_id, items in clusters.items():
                    sig = DiskJSONCache.hash_key("cluster_sig_v1", json.dumps(sorted(set(items)), ensure_ascii=False))
                    if sig in llm_allowed_signatures:
                        futs[ex.submit(label_cluster_with_llm, items)] = (cluster_id, items)

                for fut in as_completed(futs):
                    cluster_id, items = futs[fut]
                    try:
                        meta = fut.result()
                        if isinstance(meta, dict):
                            llm_meta_by_cluster[cluster_id] = meta
                        else:
                            llm_meta_by_cluster[cluster_id] = _default_meta(items)
                    except Exception:
                        llm_meta_by_cluster[cluster_id] = _default_meta(items)

    max_mentions_per_surface = int(os.getenv("KG_MAX_MENTIONS_PER_SURFACE", "8"))
    max_mentions_per_doc = int(os.getenv("KG_MAX_MENTIONS_PER_DOC", "2500"))

    # Production hardening: allow adaptive caps on large documents to avoid
    # truncating late-document provisions.
    if (os.getenv("KG_ADAPTIVE_MENTION_CAPS", "1") or "1").strip() != "0":
        try:
            # Scale doc cap with size while keeping a reasonable upper bound.
            # Roughly: allow ~1 mention per 60 chars, capped.
            adaptive_doc_cap = int(min(20000, max(max_mentions_per_doc, len(text) // 60)))
            max_mentions_per_doc = adaptive_doc_cap
        except Exception:
            pass

    spans_cache: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}

    for cluster_id, items in clusters.items():
        if _cluster_is_junk(items):
            continue

        # Only label a limited number of clusters with the LLM.
        if disable_llm:
            meta = {"is_entity": True, "label": "ENTITY", "canonical": _normalize_surface(items[0] if items else ""), "description": None}
        else:
            sig = DiskJSONCache.hash_key("cluster_sig_v1", json.dumps(sorted(set(items)), ensure_ascii=False))
            if sig in llm_allowed_signatures:
                meta = llm_meta_by_cluster.get(cluster_id)
                if not isinstance(meta, dict):
                    meta = label_cluster_with_llm(items)
            else:
                meta = {"is_entity": True, "label": "ENTITY", "canonical": _normalize_surface(items[0] if items else ""), "description": None}

        is_entity = bool(meta.get("is_entity", True))
        base_canonical = meta.get("canonical") or (items[0] if items else "")
        base_canonical = _normalize_surface(acronym_map.get(base_canonical, base_canonical))
        base_label = meta.get("label") or "ENTITY"
        desc = meta.get("description")

        # If LLM says not an entity, still keep it as a DOMAIN node
        if not is_entity:
            base_label = "DOMAIN"

        for surface in items:
            # Legal identifiers must remain distinct entities.
            # Clustering can otherwise collapse them into one canonical.
            surface_norm = _normalize_surface(surface)
            if _is_legal_id_like(surface_norm):
                surface_canonical = surface_norm
                # keep these as first-class legal entities; downstream node IDs will be deterministic.
                surface_label = "ARTICLE" if surface_norm.startswith("article ") else "PROVISION"
                surface_desc = None
            else:
                # Default behavior collapses cluster items to a single canonical.
                # For information-rich graphs, you can keep each surface as its own canonical.
                if canon_mode in {"surface", "per_surface", "high_recall"}:
                    surface_canonical = surface_norm
                else:
                    surface_canonical = base_canonical
                surface_label = base_label
                surface_desc = desc

            cache_key = (surface, max_mentions_per_surface)
            spans = spans_cache.get(cache_key)
            if spans is None:
                spans = find_spans(text, surface, max_spans=max_mentions_per_surface, word_set=doc_word_set)
                spans_cache[cache_key] = spans
            if not spans:
                continue
            for start, end in spans:
                if max_mentions_per_doc > 0 and len(all_entities) >= max_mentions_per_doc:
                    break
                all_entities.append(
                    Entity(
                        text=text[start:end],
                        label=surface_label,
                        start=start,
                        end=end,
                        source=("embed_cluster_llm" if (not disable_llm and meta.get("description") is not None) else "embed_cluster"),
                        canonical=surface_canonical,
                        description=surface_desc,
                        context=text[max(0, start-80):min(len(text), end+80)],
                        doc_id=doc_id,
                    )
                )
            if max_mentions_per_doc > 0 and len(all_entities) >= max_mentions_per_doc:
                break
        if max_mentions_per_doc > 0 and len(all_entities) >= max_mentions_per_doc:
            break

    # --- Ensure minimum graph density per document ---
    MIN_ENTITIES_PER_DOC = 10
    if len(all_entities) < MIN_ENTITIES_PER_DOC:
        doc = nlp(text)
        freq: Dict[str, int] = {}
        for chunk in doc.noun_chunks:
            norm = _normalize_surface(chunk.text)
            if not norm or norm.isdigit() or len(norm) <= 1:
                continue
            freq[norm] = freq.get(norm, 0) + 1

        existing_canons = { _normalize_surface(e.canonical or e.text) for e in all_entities }

        # Sort candidates by frequency descending
        for norm, _count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
            if len(all_entities) >= MIN_ENTITIES_PER_DOC:
                break
            if norm in existing_canons:
                continue

            spans = find_spans(text, norm, max_spans=1, word_set=doc_word_set)
            if not spans:
                continue
            start, end = spans[0]
            all_entities.append(
                Entity(
                    text=text[start:end],
                    label="DOMAIN",
                    start=start,
                    end=end,
                    source="auto_domain_promotion",
                    canonical=norm,
                    description=None,
                    context=text[max(0, start-80):min(len(text), end+80)],
                    doc_id=doc_id,
                )
            )

    # Persist for fast reruns
    try:
        _ENTITIES_CACHE.set(ent_cache_key, [asdict(e) for e in all_entities])
    except Exception:
        pass

    return all_entities





def build_entity_catalog(
    all_entities_per_doc: Dict[str, List[Entity]]
) -> Dict[str, EntityCatalogEntry]:
    catalog: Dict[str, EntityCatalogEntry] = {}

    for doc_id, ents in all_entities_per_doc.items():
        for e in ents:
            canon = e.canonical or _normalize_surface(e.text)
            if canon not in catalog:
                catalog[canon] = EntityCatalogEntry(
                    canonical=canon,
                    label=e.label,
                )
            entry = catalog[canon]
            entry.aliases.add(e.text)
            if e.description:
                entry.descriptions.add(e.description)
            entry.sources.add(doc_id)

    return catalog


def merge_similar_catalog_entries(catalog: Dict[str, EntityCatalogEntry], similarity_threshold: float = 0.965) -> Dict[str, EntityCatalogEntry]:
    """Merge catalog entries that are semantically close to reduce fragmentation.

    Enhancements for legal corpora:
    - Avoid merging legal identifiers like "article 3" vs "article 13".
    - Avoid merging ENTITY <-> DOMAIN automatically.
    - Use stricter thresholds for short and digit-heavy canonicals.
    """
    if not catalog:
        return catalog

    # High-recall mode: keep more distinct entities by skipping merges.
    if (os.getenv("KG_DISABLE_ENTITY_MERGE", "0") or "0").strip() == "1":
        return catalog

    legal_id_re = re.compile(
        r"^(article|section|clause|chapter|part|schedule|rule|regulation)\s+\d+[a-z]?(?:\([0-9a-z]+\))*$",
        flags=re.IGNORECASE,
    )

    # Expand legal-id protection to non-numeric constitution structure (e.g., "part iii", schedules).
    part_re = re.compile(r"^part\s+[ivxlcdm]{1,7}$", flags=re.IGNORECASE)
    sched_re = re.compile(r"^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+schedule$", flags=re.IGNORECASE)

    def _is_legal_id(name: str) -> bool:
        n = _normalize_surface(name)
        return bool(legal_id_re.match(n) or part_re.match(n) or sched_re.match(n))

    def _needs_strict(name: str) -> bool:
        nrm = _normalize_surface(name)
        if _is_legal_id(nrm):
            return True
        if any(ch.isdigit() for ch in nrm):
            return True
        if len(nrm) <= 5:
            return True
        return False

    names = list(catalog.keys())
    embeddings = get_embeddings_with_cache(names)
    merged: Dict[str, EntityCatalogEntry] = {}
    used: Set[int] = set()

    for i, name in enumerate(names):
        if i in used:
            continue
        base_entry = catalog[name]
        base_vec = embeddings[i]
        merged_aliases = set(base_entry.aliases)
        merged_desc = set(base_entry.descriptions)
        merged_sources = set(base_entry.sources)
        merged_label = base_entry.label

        base_norm = _normalize_surface(name)
        base_is_legal = _is_legal_id(base_norm)
        base_strict = _needs_strict(base_norm)

        for j in range(i + 1, len(names)):
            if j in used:
                continue
            other_name = names[j]
            other = catalog[other_name]
            other_norm = _normalize_surface(other_name)

            # Never merge DOMAIN with non-DOMAIN automatically.
            if (base_entry.label == "DOMAIN") != (other.label == "DOMAIN"):
                continue

            # Never merge across *legal types*; this is the root-cause fix for catastrophic collapse.
            if str(base_entry.label) != str(other.label):
                continue

            # If either looks like a legal id: only merge exact normalized match.
            if base_is_legal or _is_legal_id(other_norm):
                if base_norm != other_norm:
                    continue
                sim_thr = 1.0
            else:
                # Default should be strict for constitutional corpora.
                # Less aggressive default merging improves entity counts.
                default_thr = float(os.getenv("KG_ENTITY_MERGE_THRESHOLD", "0.99") or 0.99)
                sim_thr = 0.985 if (base_strict or _needs_strict(other_norm)) else max(float(similarity_threshold), default_thr)

            sim = float(np.dot(base_vec, embeddings[j]) / ((np.linalg.norm(base_vec) + 1e-9) * (np.linalg.norm(embeddings[j]) + 1e-9)))
            if sim >= sim_thr:
                merged_aliases |= other.aliases
                merged_desc |= other.descriptions
                merged_sources |= other.sources
                used.add(j)

        merged[name] = EntityCatalogEntry(
            canonical=name,
            label=merged_label,
            aliases=merged_aliases,
            descriptions=merged_desc,
            sources=merged_sources,
        )

    return merged


def add_entity_nodes(graph: KnowledgeGraph, catalog: Dict[str, EntityCatalogEntry]):
    for canon, entry in catalog.items():
        node_id = _canonical_to_node_id(canon)
        node_label = "DOMAIN" if entry.label == "DOMAIN" else "ENTITY"
        properties = {
            "canonical": entry.canonical,
            "type": entry.label,
            "aliases": sorted(entry.aliases),
            "descriptions": list(entry.descriptions),
            "sources": list(entry.sources),
        }
        # Mark confidence lower for DOMAIN nodes that often come from promotion or non-entity clusters
        if node_label == "DOMAIN":
            properties["confidence"] = "low"
        else:
            properties["confidence"] = "high"

        graph.add_node(
            KGNode(
                id=node_id,
                label=node_label,
                properties=properties,
            )
        )


def add_mention_and_cooccurrence_edges(
    graph: KnowledgeGraph,
    all_entities_per_doc: Dict[str, List[Entity]],
    sent_index: Dict[str, SentenceInfo],
):
    edge_counter = 0
    entities_per_sentence: Dict[str, List[Tuple[Entity, str]]] = {}

    # Build a fast per-document sentence range index to avoid O(E*S) scans.
    # doc_id -> list of (start_char, end_char, sent_id) sorted by start_char
    doc_sent_ranges: Dict[str, List[Tuple[int, int, str]]] = {}
    doc_sent_starts: Dict[str, List[int]] = {}
    for sid, s in sent_index.items():
        doc = str(s.doc_id)
        doc_sent_ranges.setdefault(doc, []).append((int(s.start_char), int(s.end_char), sid))
    for doc, ranges in doc_sent_ranges.items():
        ranges.sort(key=lambda x: x[0])
        doc_sent_starts[doc] = [r[0] for r in ranges]

    def _sent_ids_for_span(doc_id: str, start: int, end: int) -> List[str]:
        ranges = doc_sent_ranges.get(str(doc_id))
        if not ranges:
            return []
        starts = doc_sent_starts.get(str(doc_id)) or []
        # Locate insertion point by entity start, then scan locally.
        i = bisect_right(starts, int(start)) - 1
        i = max(0, i)
        out: List[str] = []
        # Scan from one sentence before in case entity starts near boundary.
        j = max(0, i - 1)
        e_end = int(end)
        e_start = int(start)
        while j < len(ranges):
            s_start, s_end, sid = ranges[j]
            if s_start > e_end:
                break
            if (e_start < s_end) and (e_end > s_start):
                out.append(sid)
            j += 1
        return out

    # MENTION_IN edges
    for doc_id, ents in all_entities_per_doc.items():
        for e in ents:
            ent_node_id = _entity_to_node_id(e)

            # Optimized sentence lookup
            sent_ids = _sent_ids_for_span(str(doc_id), int(e.start), int(e.end))
            for sid in sent_ids:
                edge_id = f"e:{edge_counter}"; edge_counter += 1
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=ent_node_id,
                        target=sid,
                        type="MENTION_IN",
                        properties={
                            "surface": e.text,
                            "doc_id": doc_id,
                            "char_start": e.start,
                            "char_end": e.end,
                        },
                    )
                )
                entities_per_sentence.setdefault(sid, []).append((e, ent_node_id))

    # CO_OCCURS_WITH edges (dedup + weighted)
    # Aggregate globally (one edge per pair) with count/weight and a small evidence list.
    pair_agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    max_evidence = int(os.getenv("KG_MAX_COOCCUR_EVIDENCE", "6"))

    for sid, ent_list in entities_per_sentence.items():
        # Deduplicate within a sentence by node id
        uniq_nodes = list({node_id for (_e, node_id) in ent_list if node_id and node_id != "ent:unknown"})
        if len(uniq_nodes) < 2:
            continue

        # Sentence contribution: normalize so very dense sentences don't dominate.
        base_inc = 1.0 / max(1, (len(uniq_nodes) - 1))

        for i in range(len(uniq_nodes)):
            for j in range(i + 1, len(uniq_nodes)):
                a, b = uniq_nodes[i], uniq_nodes[j]
                if a == b:
                    continue
                node1, node2 = (a, b) if a < b else (b, a)

                # Downweight generic hubs (by canonical string)
                canon1 = (node1.split(":", 1)[1] if ":" in node1 else node1).replace("_", " ")
                canon2 = (node2.split(":", 1)[1] if ":" in node2 else node2).replace("_", " ")
                w = base_inc
                if canon1 in _GENERIC_ENTITY_STOPWORDS:
                    w *= 0.25
                if canon2 in _GENERIC_ENTITY_STOPWORDS:
                    w *= 0.25
                if canon1 in _GENERIC_ENTITY_STOPWORDS and canon2 in _GENERIC_ENTITY_STOPWORDS:
                    w *= 0.1

                key = (node1, node2)
                agg = pair_agg.get(key)
                if agg is None:
                    agg = {"count": 0, "weight": 0.0, "sentences": []}
                    pair_agg[key] = agg
                agg["count"] += 1
                agg["weight"] = float(agg["weight"]) + float(w)
                if max_evidence > 0 and len(agg["sentences"]) < max_evidence:
                    agg["sentences"].append(sid)

    for (node1, node2), agg in pair_agg.items():
        edge_id = f"e:{edge_counter}"; edge_counter += 1
        graph.add_edge(
            KGEdge(
                id=edge_id,
                source=node1,
                target=node2,
                type="CO_OCCURS_WITH",
                properties={
                    "count": int(agg["count"]),
                    "weight": float(agg["weight"]),
                    "evidence_sentence_ids": list(agg["sentences"]),
                },
            )
        )

    # STRUCTURE: ARTICLE -> PART edges (semantic backbone)
    # Deterministic: attach each article mention to the most recent PART heading.
    # This enables semantic communities and legal navigation.
    seen_struct: Set[Tuple[str, str, str]] = set()  # (doc_id, art_id, part_id)
    for doc_id, ents in all_entities_per_doc.items():
        parts: List[Tuple[int, str]] = []
        articles: List[Tuple[int, str]] = []

        for e in ents:
            canon = _normalize_surface(e.canonical or e.text)
            if not canon:
                continue
            nid = _canonical_to_node_id(canon)
            if e.label == "PART" or nid.startswith("part:"):
                parts.append((int(e.start), nid))
            if e.label in {"ARTICLE", "PROVISION"} and canon.startswith("article "):
                articles.append((int(e.start), nid))

        if not parts or not articles:
            continue
        parts.sort(key=lambda x: x[0])
        articles.sort(key=lambda x: x[0])

        pi = 0
        current_part: Optional[str] = None
        for a_start, a_id in articles:
            while pi < len(parts) and parts[pi][0] <= a_start:
                current_part = parts[pi][1]
                pi += 1
            if not current_part:
                continue
            if a_id not in graph.nodes or current_part not in graph.nodes:
                continue
            key = (str(doc_id), a_id, current_part)
            if key in seen_struct:
                continue
            seen_struct.add(key)
            edge_id = f"e:{edge_counter}"; edge_counter += 1
            graph.add_edge(
                KGEdge(
                    id=edge_id,
                    source=a_id,
                    target=current_part,
                    type="PART_OF",
                    properties={"doc_id": doc_id, "source": "structure"},
                )
            )

    # STRUCTURE: CONTAINS edges for explicit hierarchy navigation
    # doc:{doc_id} -> part:* -> art:* and doc:{doc_id} -> art:* (direct)
    # These are deterministic and do not depend on communities.
    seen_contains: Set[Tuple[str, str, str]] = set()  # (doc_id, src, tgt)
    for doc_id, ents in all_entities_per_doc.items():
        doc_node_id = f"doc:{doc_id}"
        if doc_node_id not in graph.nodes:
            continue

        part_positions: List[Tuple[int, str]] = []
        art_positions: List[Tuple[int, str]] = []
        for e in ents:
            canon = _normalize_surface(e.canonical or e.text)
            if not canon:
                continue
            nid = _canonical_to_node_id(canon)
            if nid not in graph.nodes:
                continue
            if e.label == "PART" or nid.startswith("part:"):
                part_positions.append((int(e.start), nid))
            if (e.label == "ARTICLE" or nid.startswith("art:")) and canon.startswith("article "):
                art_positions.append((int(e.start), nid))

        if part_positions:
            part_positions.sort(key=lambda x: x[0])
            for _pos, pid in part_positions:
                key = (str(doc_id), doc_node_id, pid)
                if key in seen_contains:
                    continue
                seen_contains.add(key)
                edge_id = f"e:contains:{hashlib.sha256(('doc||'+str(doc_id)+'||'+pid).encode('utf-8')).hexdigest()[:16]}"
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=doc_node_id,
                        target=pid,
                        type="CONTAINS",
                        properties={"doc_id": doc_id, "source": "structure"},
                    )
                )

        if art_positions:
            art_positions.sort(key=lambda x: x[0])
            for _pos, aid in art_positions:
                key = (str(doc_id), doc_node_id, aid)
                if key in seen_contains:
                    continue
                seen_contains.add(key)
                edge_id = f"e:contains:{hashlib.sha256(('doc||'+str(doc_id)+'||'+aid).encode('utf-8')).hexdigest()[:16]}"
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=doc_node_id,
                        target=aid,
                        type="CONTAINS",
                        properties={"doc_id": doc_id, "source": "structure"},
                    )
                )

    # part:* -> art:* via existing ARTICLE->PART edges
    # Emit CONTAINS in the reverse direction for navigation.
    seen_part_contains: Set[Tuple[str, str]] = set()
    for e in graph.edges:
        if e.type != "PART_OF":
            continue
        if not str(e.source).startswith("art:"):
            continue
        if not str(e.target).startswith("part:"):
            continue
        pid = str(e.target)
        aid = str(e.source)
        key2 = (pid, aid)
        if key2 in seen_part_contains:
            continue
        seen_part_contains.add(key2)
        edge_id = f"e:contains:{hashlib.sha256(('part||'+pid+'||'+aid).encode('utf-8')).hexdigest()[:16]}"
        graph.add_edge(
            KGEdge(
                id=edge_id,
                source=pid,
                target=aid,
                type="CONTAINS",
                properties={"source": "structure"},
            )
        )


REL_EXTRACT_SYSTEM_PROMPT = """
You are an expert in LEGAL and FINANCE relation extraction for building a high-fidelity knowledge graph.

Goal:
Identify ALL semantic relationships between the provided entities in the given text.

You will receive:
- A text context (one or more sentences).
- A list of entities found in that text, with their canonical names and types.

Task:
- Extract relations as (head, relation, tail) grounded in the text.
- Be thorough: Capture both explicit statements and clear logical implications.
- For insurance/legal documents, prioritize relations like:
    DEFINES, PROVIDES_FOR, EMPOWERS, REQUIRES, PROHIBITS, LIMITS,
    AMENDS, SUBJECT_TO, NOTWITHSTANDING, EXCEPTS, APPLIES_TO,
    BALANCES_WITH, CONSIDERS_VIEWS_OF, PROCEDURE_FOR, INTERPRETS,
    OVERRIDES, SAVES_LAWS_FROM_INVALIDATION,
    REQUIRES_RECOMMENDATION_FROM, REQUIRES_CONSULTATION_WITH.
- For finance/regulatory context, use:
    OWNS, OWNED_BY, SUBSIDIARY_OF, PARENT_OF,
    ACQUIRED, ACQUIRED_BY, MERGED_WITH,
    INVESTS_IN, FUNDED_BY,
    ISSUES, ISSUED_BY,
    GUARANTEED_BY, INSURED_BY,
    SECURED_BY, COLLATERAL_FOR,
    RATED_BY, REGULATED_BY, COMPLIES_WITH,
    LISTED_ON, TRADED_ON, HAS_EXPOSURE_TO.
- Use RELATED_TO as a safe fallback for any meaningful connection that doesn't fit a specific type.

Rules:
- "head" and "tail" MUST be canonical names exactly from the provided list.
- Relation labels MUST be UPPERCASE_WITH_UNDERSCORES.
- Do NOT hallucinate entities not in the list.
- Include a confidence score 0..1 based on the strength of textual evidence.

Return ONLY a JSON list:
[
    {"head": "The Insurer", "relation": "INDEMNIFIES", "tail": "Hospitalization Costs", "confidence": 0.95},
    ...
]
"""

ONESHOT_KG_PROMPT = """
You are an expert Ontologist specializing in LEGAL, INSURANCE, and FINANCE domains.

Task:
Perform a deep, exhaustive extraction of all ENTITIES, CONCEPTS, and RELATIONSHIPS from the provided document chunk.
Your goal is HIGH RECALL. Document the granular structural relationship of all provisions, definitions, and obligations.

1. ENTITIES/CONCEPTS:
   - Canonical name: Formal standard name (e.g., "The Insurer", "Section 149(2)").
   - Label: Precise type (POLICY_TYPE, COVERAGE, EXCLUSION, CLAIM_PROCEDURE, REGULATORY_BODY, LEGAL_ACT, PROVISION, CONDITION, etc.).
   - Description: 2-3 sentences explaining the concept's exact meaning.
   - Mention: The exact phrase/word used in the text.

2. RELATIONSHIPS:
   - Head/Tail: Canonical names of entities extracted above.
   - Relation: UPPERCASE_WITH_UNDERSCORES.
   - Confidence: 0..1.
   - Evidence Snippet: Exact text phrase supporting the connection.

Priority Relation Types:
- DEFINES, PROVIDES_FOR, EMPOWERS, REQUIRES, PROHIBITS, LIMITS, AMENDS, SUBJECT_TO, NOTWITHSTANDING, EXCEPTS, APPLIES_TO, BALANCES_WITH, CONSIDERS_VIEWS_OF, PROCEDURE_FOR, INTERPRETS, OVERRIDES, OWNS, ISSUES, GUARANTEED_BY, REGULATED_BY, COMPLIES_WITH.

Return ONLY a JSON object:
{
  "entities": [
    {"canonical": "...", "label": "...", "description": "...", "mention": "..."},
    ...
  ],
  "relations": [
    {"head": "...", "relation": "...", "tail": "...", "confidence": 0.95, "evidence_snippet": "..."},
    ...
  ]
}
"""


def _load_relation_schema() -> Dict[str, Any]:
    """Load relation schema/ontology (cached).

    Primary source is `relation_schema.json` (or `KG_RELATION_SCHEMA_PATH`).
    `KG_RELATION_SCHEMA_JSON` can override/extend.
    """

    global _RELATION_SCHEMA_CACHE
    if _RELATION_SCHEMA_CACHE is not None:
        return _RELATION_SCHEMA_CACHE

    with _RELATION_SCHEMA_LOCK:
        if _RELATION_SCHEMA_CACHE is not None:
            return _RELATION_SCHEMA_CACHE

        try:
            from graph_maker.relation_schema import load_relation_schema

            schema = load_relation_schema()
        except Exception:
            schema = {}

        # Backward compatible defaults (used if schema file missing).
        schema.setdefault("synonyms", {})
        if not isinstance(schema.get("synonyms"), dict):
            schema["synonyms"] = {}
        schema.setdefault("max_label_len", 48)
        schema.setdefault("min_confidence", 0.35)
        schema.setdefault("allowed_rel_regex", r"^[A-Z][A-Z0-9_]{1,47}$")
        schema.setdefault("allowed_types", [])
        schema.setdefault("proposal_path", "edge_type_proposals.jsonl")

        # Ensure RELATED_TO is always possible as a safe fallback.
        allowed_types = schema.get("allowed_types")
        if isinstance(allowed_types, list) and allowed_types:
            if "RELATED_TO" not in set(str(x).upper() for x in allowed_types):
                allowed_types.append("RELATED_TO")

        _RELATION_SCHEMA_CACHE = schema
        return _RELATION_SCHEMA_CACHE


def _schema_allowed_types(schema: Dict[str, Any]) -> Set[str]:
    allowed = schema.get("allowed_types")
    if not isinstance(allowed, list) or not allowed:
        return set()
    return {str(x).strip().upper() for x in allowed if str(x).strip()}




def _rule_based_constitution_relations(sentence_text: str, ent_list: List[Tuple[Entity, str]]) -> List[Dict[str, Any]]:
    """Deterministic constitutional relation extraction.

    This plugs the exact gap you observed: procedural safeguards and override/shield
    relations should exist even when LLM extraction is sparse.
    """
    t = (sentence_text or "")
    tl = t.lower()
    if not t or len(ent_list) < 1:
        return []

    # Collect canonicals present in the sentence (best-effort); we may also emit
    # relations to globally-known nodes if they're not in ent_list for this sentence.
    canon_by_label: Dict[str, Set[str]] = {}
    for e, _nid in ent_list:
        canon = _normalize_surface(e.canonical or e.text)
        if not canon:
            continue
        canon_by_label.setdefault(str(e.label), set()).add(canon)

    # Identify article canonicals (normalized like "article 3", "article 31a")
    article_canons: List[str] = []
    for lbl in ("ARTICLE", "PROVISION"):
        for c in sorted(canon_by_label.get(lbl, set())):
            if c.startswith("article "):
                article_canons.append(c)

    # Helper to pick a head article for Article 3 safeguards when multiple articles appear.
    def _pick_article(prefer: Optional[str] = None) -> Optional[str]:
        if prefer:
            for c in article_canons:
                if c == prefer:
                    return c
        return article_canons[0] if article_canons else None

    out: List[Dict[str, Any]] = []

    # --- Article 3 procedural safeguards ---
    head = _pick_article(prefer="article 3")
    if not head:
        # Fallback: some PDFs use numbered headings like "3." instead of "Article 3".
        if ("article 3" in tl) or bool(re.search(r"(?m)^\s*3\s*(?:[\.\-\u2013\u2014]{1,2})\s+", t)):
            head = "article 3"
    if head:
        # Recommendation of the President (robust to missing 'the' / punctuation / possessive)
        if (
            bool(re.search(r"recommendation\s+of\s+(?:the\s+)?president", tl))
            or bool(re.search(r"president(?:'s)?\s+recommendation", tl))
            or ("recommendation" in tl and "president" in tl)
        ):
            out.append({
                "head": head,
                "relation": "REQUIRES_RECOMMENDATION_FROM",
                "tail": "president",
                "confidence": 0.92,
            })

        # Consultation / views of affected state legislature
        has_legislature = "legislature" in tl
        has_views = ("views" in tl) or bool(re.search(r"express(?:ing)?\s+its\s+views", tl))
        has_refer = bool(re.search(r"\b(refer|referred|referring)\b", tl))
        if (has_legislature and has_views) or (has_legislature and has_refer):
            tail = "affected state legislature" if "affected" in tl else "state legislature"
            out.append({
                "head": head,
                "relation": "REQUIRES_CONSULTATION_WITH",
                "tail": tail,
                "confidence": 0.88,
            })
            out.append({
                "head": head,
                "relation": "CONSIDERS_VIEWS_OF",
                "tail": tail,
                "confidence": 0.82,
            })

        if "first schedule" in tl:
            out.append({
                "head": head,
                "relation": "AMENDS",
                "tail": "first schedule",
                "confidence": 0.84,
            })
        if "fourth schedule" in tl:
            out.append({
                "head": head,
                "relation": "AMENDS",
                "tail": "fourth schedule",
                "confidence": 0.84,
            })

    # --- Articles 31A/31B/31C vs Article 13 override/shield ---
    has_void_shield = bool(re.search(r"shall\s+not\s+be\s+deemed\s+to\s+be\s+void", tl)) or ("shall not be void" in tl)
    if has_void_shield and "article 13" in tl:
        for c in article_canons:
            if c in {"article 31a", "article 31b", "article 31c"}:
                out.append({
                    "head": c,
                    "relation": "OVERRIDES",
                    "tail": "article 13",
                    "confidence": 0.9,
                })
                out.append({
                    "head": c,
                    "relation": "SAVES_LAWS_FROM_INVALIDATION",
                    "tail": "article 13",
                    "confidence": 0.88,
                })

    return out


def _rule_based_finance_legal_relations(sentence_text: str, ent_list: List[Tuple[Entity, str]]) -> List[Dict[str, Any]]:
    """Deterministic finance/legal mechanics relations.

    Goal: ensure GraphRAG has *mechanistic* edges even when LLM extraction is off or sparse.
    We only emit relations between entities already present in the sentence entity list.
    """
    t = (sentence_text or "")
    tl = t.lower()
    if not t or len(ent_list) < 2:
        return []

    # Build canonical list in sentence order (best-effort).
    canons: List[str] = []
    for e, _nid in ent_list:
        c = _normalize_surface(e.canonical or e.text)
        if c and c not in canons:
            canons.append(c)

    if len(canons) < 2:
        return []

    def _looks_like_statute(c: str) -> bool:
        return bool(
            re.search(r"\b(act|acts|statute|regulation|rule|rules|code|law|laws|directive|byelaw|bye\-law)\b", c)
            or re.match(r"^(section|sec|s|article|clause|chapter|part|schedule|regulation|rule)\s+\d+", c)
        )

    def _looks_like_exclusion(c: str) -> bool:
        return bool(re.search(r"\b(exclusion|exclusions|exception|exceptions|excluded|not covered|shall not apply)\b", c))

    def _looks_like_obligation(c: str) -> bool:
        return bool(re.search(r"\b(liability|liable|obligation|obligations|duty|indemnif(y|ies|ication))\b", c))

    def _first(pred) -> Optional[str]:
        for c in canons:
            if pred(c):
                return c
        return None

    statute = _first(_looks_like_statute)
    exclusion = _first(_looks_like_exclusion)
    obligation = _first(_looks_like_obligation)

    out: List[Dict[str, Any]] = []

    # Override / notwithstanding mechanics.
    if "notwithstanding" in tl or "override" in tl or "prevail" in tl:
        # Prefer statute overrides exclusion/obligation.
        if statute and exclusion and statute != exclusion:
            out.append({"head": statute, "relation": "OVERRIDES", "tail": exclusion, "confidence": 0.82})
            out.append({"head": statute, "relation": "NOTWITHSTANDING", "tail": exclusion, "confidence": 0.78})
        elif statute and obligation and statute != obligation:
            out.append({"head": statute, "relation": "OVERRIDES", "tail": obligation, "confidence": 0.78})
        else:
            # Fallback: connect the first two entities with NOTWITHSTANDING.
            out.append({"head": canons[0], "relation": "NOTWITHSTANDING", "tail": canons[1], "confidence": 0.7})

    # Subject-to / compliance / governed-by.
    if "subject to" in tl or "in accordance with" in tl or "pursuant to" in tl or "as per" in tl:
        if obligation and statute and obligation != statute:
            out.append({"head": obligation, "relation": "SUBJECT_TO", "tail": statute, "confidence": 0.76})
        elif exclusion and statute and exclusion != statute:
            out.append({"head": exclusion, "relation": "SUBJECT_TO", "tail": statute, "confidence": 0.72})
        elif statute and len(canons) >= 2:
            other = canons[1] if canons[0] == statute and len(canons) > 1 else canons[0]
            if other and other != statute:
                out.append({"head": other, "relation": "SUBJECT_TO", "tail": statute, "confidence": 0.68})

    if "comply" in tl or "compliance" in tl:
        if obligation and statute and obligation != statute:
            out.append({"head": obligation, "relation": "COMPLIES_WITH", "tail": statute, "confidence": 0.72})
        elif len(canons) >= 2:
            out.append({"head": canons[0], "relation": "COMPLIES_WITH", "tail": canons[1], "confidence": 0.62})

    # Recovery / reimbursement / recourse.
    if (
        "right of recovery" in tl
        or "recovery" in tl
        or re.search(r"\brecover\b", tl)
        or "reimburse" in tl
        or "recourse" in tl
        or "indemnif" in tl
    ):
        insurer = _first(lambda c: "insurer" in c or "insurance" in c)
        insured = _first(lambda c: "insured" in c)
        third_party = _first(lambda c: "third party" in c or "third-party" in c)
        if insurer and insured and insurer != insured:
            out.append({"head": insurer, "relation": "HAS_RIGHT_OF_RECOVERY", "tail": insured, "confidence": 0.8})
            out.append({"head": insurer, "relation": "RECOVERS_FROM", "tail": insured, "confidence": 0.76})
        elif insured and third_party and insured != third_party:
            out.append({"head": insured, "relation": "LIABLE_FOR", "tail": third_party, "confidence": 0.68})
        else:
            # Fallback: connect first two with RECOVERS_FROM.
            out.append({"head": canons[0], "relation": "RECOVERS_FROM", "tail": canons[1], "confidence": 0.62})

    return out


def _normalize_relation_label(label: str, schema: Dict[str, Any]) -> Optional[str]:
    if not label:
        return None
    s = str(label).strip().upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_]", "", s)
    syn = schema.get("synonyms") if isinstance(schema.get("synonyms"), dict) else {}
    if isinstance(syn, dict):
        s = syn.get(s.replace("_", ""), s)  # tolerate no-underscore form
        s = syn.get(s, s)

    max_len = int(schema.get("max_label_len", 48))
    if max_len > 0:
        s = s[:max_len]

    rx = schema.get("allowed_rel_regex")
    if rx:
        try:
            if not re.match(rx, s):
                return None
        except Exception:
            pass
    return s


def extract_oneshot_kg_from_doc(doc_id: str, text: str) -> List[Entity]:
    """Extract entities and relations in one shot for a document."""
    digest = _text_digest(text)
    cache_key = DiskJSONCache.hash_key("oneshot_v1", doc_id, digest)
    cached = _ONESHOT_RELATION_CACHE.get(cache_key)
    
    if cached and isinstance(cached, dict):
        entities_data = cached.get("entities", [])
        # CRITICAL: Always ensure 'latest' mapping exists even on cache hit for Phase 5 retrieval
        doc_latest_key = DiskJSONCache.hash_key("oneshot_doc_latest", doc_id)
        _ONESHOT_RELATION_CACHE.set(doc_latest_key, cached)
    else:
        # 1. Gliding Window Logic
        window_size = int(os.getenv("KG_ONESHOT_WINDOW_SIZE", "8000"))
        overlap = int(os.getenv("KG_ONESHOT_WINDOW_OVERLAP", "2000"))
        
        windows = []
        if len(text) <= window_size:
            windows.append(text)
        else:
            step = window_size - overlap
            for i in range(0, len(text), step):
                chunk = text[i:i + window_size]
                if len(chunk) < 500 and windows: # Skip tiny trailing chunks if possible
                    continue
                windows.append(chunk)
                if i + window_size >= len(text):
                    break
        
        global_entities = []
        global_relations = []
        seen_entity_canons = {} # canonical -> full object
        seen_relation_keys = set() # (head, rel, tail)
        
        logging.warning(f"One-shot Gliding Window starting for {doc_id} with {len(windows)} windows (Size: {window_size}, Overlap: {overlap})")
        
        def _process_window(idx, window_text):
            prompt = f"{ONESHOT_KG_PROMPT}\n\nDOCUMENT TEXT (WINDOW {idx+1}):\n{window_text}"
            
            try:
                from utils.genai_compat import generate_text as unified_generate_text
                raw = unified_generate_text(model=None, prompt=prompt, temperature=0.1, purpose="RELATION")
                data = _parse_json_safely(raw, default={"entities": [], "relations": []})
                win_entities = data.get("entities", [])
                win_relations = data.get("relations", [])
            except Exception as e:
                logging.error(f"One-shot Window {idx+1} KG Extraction failed: {e}")
                win_entities = []
                win_relations = []
            
            return win_entities, win_relations

        
        window_workers = int(os.getenv("KG_ONESHOT_WINDOW_WORKERS", "0") or 0)
        if window_workers <= 0:
            # Lowered from 10 to 4 to better respect free-tier RPM limits
            window_workers = min(4, (os.cpu_count() or 2))

        if len(windows) <= 1 or window_workers <= 1:
            for idx, window_text in enumerate(windows):
                logging.warning(f"  -> Processing Window {idx+1}/{len(windows)} ({len(window_text)} chars)...")
                win_entities, win_relations = _process_window(idx, window_text)
                
                # Merge logic
                for ent in win_entities:
                    canon = str(ent.get("canonical", "")).strip()
                    if not canon: continue
                    if canon not in seen_entity_canons:
                        seen_entity_canons[canon] = ent
                        global_entities.append(ent)
                    else:
                        existing = seen_entity_canons[canon]
                        if len(str(ent.get("description", ""))) > len(str(existing.get("description", ""))):
                            existing["description"] = ent["description"]
                
                for rel in win_relations:
                    head = str(rel.get("head", "")).strip()
                    tail = str(rel.get("tail", "")).strip()
                    label = str(rel.get("relation", "")).strip()
                    if not head or not tail or not label: continue
                    
                    rel_key = (head, label, tail)
                    if rel_key not in seen_relation_keys:
                        global_relations.append(rel)
                        seen_relation_keys.add(rel_key)
        else:
            logging.warning(f"  -> Processing {len(windows)} windows in parallel (workers={window_workers})...")
            with ThreadPoolExecutor(max_workers=window_workers) as ex:
                futs = {ex.submit(_process_window, i, t): i for i, t in enumerate(windows)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    win_entities, win_relations = fut.result()
                    
                    # Merge logic (needs lock or single-threaded post-process, but here we process as-completed)
                    # Note: seen_entity_canons and lists are being modified; for safety in as_completed loop
                    # we should be careful, but since it's a single consumer thread here it's fine.
                    for ent in win_entities:
                        canon = str(ent.get("canonical", "")).strip()
                        if not canon: continue
                        if canon not in seen_entity_canons:
                            seen_entity_canons[canon] = ent
                            global_entities.append(ent)
                        else:
                            existing = seen_entity_canons[canon]
                            if len(str(ent.get("description", ""))) > len(str(existing.get("description", ""))):
                                existing["description"] = ent["description"]
                    
                    for rel in win_relations:
                        head = str(rel.get("head", "")).strip()
                        tail = str(rel.get("tail", "")).strip()
                        label = str(rel.get("relation", "")).strip()
                        if not head or not tail or not label: continue
                        
                        rel_key = (head, label, tail)
                        if rel_key not in seen_relation_keys:
                            global_relations.append(rel)
                            seen_relation_keys.add(rel_key)

        entities_data = global_entities
        relations_data = global_relations

        cached = {
            "entities": entities_data,
            "relations": relations_data,
            "digest": digest,
            "window_count": len(windows)
        }
        # Store results
        v2_cache_key = DiskJSONCache.hash_key("oneshot_v2", doc_id, digest)
        _ONESHOT_RELATION_CACHE.set(v2_cache_key, cached)
        doc_latest_key = DiskJSONCache.hash_key("oneshot_doc_latest", doc_id)
        _ONESHOT_RELATION_CACHE.set(doc_latest_key, cached)

        msg = f"One-shot Gliding Window finished. Total: {len(entities_data)} entities, {len(relations_data)} relations across {len(windows)} windows."
        logging.warning(msg)

    # Process entities and find spans
    all_entities: List[Entity] = []
    # Build a simple token set for fast pre-filtering
    doc_word_set = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    
    max_mentions_per_surface = int(os.getenv("KG_MAX_MENTIONS_PER_SURFACE", "8"))
    
    for ent in entities_data:
        canon = ent.get("canonical")
        mention = ent.get("mention")
        label = ent.get("label", "ENTITY")
        desc = ent.get("description")
        if not canon and not mention:
            continue
            
        # Try both mention (if provided) and canonical for span finding.
        # Use fallback to canon if mention is missing or not found.
        spans = []
        if mention:
            spans = find_spans(text, mention, max_spans=max_mentions_per_surface, word_set=doc_word_set)
        
        if not spans and canon:
            spans = find_spans(text, canon, max_spans=max_mentions_per_surface, word_set=doc_word_set)
            
        if not spans:
            logging.debug(f"One-shot: Could not find span for '{canon}' (mention: '{mention}')")
            continue
            
        for start, end in spans:
            all_entities.append(Entity(
                text=text[start:end],
                label=label,
                start=start,
                end=end,
                source="oneshot",
                canonical=canon,
                description=desc,
                context=text[max(0, start-80):min(len(text), end+80)],
                doc_id=doc_id
            ))
            
    logging.info(f"One-shot successfully mapped {len(all_entities)} entity instances to text for {doc_id}")
    return all_entities


def _dedupe_entities_for_llm(ent_list: List[Tuple[Entity, str]]) -> List[Dict[str, str]]:
    """Unique entities by canonical string for LLM payload."""
    seen: Set[str] = set()
    out: List[Dict[str, str]] = []
    for e, _node_id in ent_list:
        canon = _canon_for_payload(e)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        out.append({
            "canonical": canon,
            "type": e.label,
            "surface": e.text,
        })
    return out


def extract_relations_for_context(
    text: str,
    entities: List[Tuple[Entity, str]],  # (entity, ent_node_id)
    context_id: str = "batch"
) -> List[Dict]:
    """Extract semantic relations from a block of text using the LLM.
    
    Supports batching multiple sentences to reduce API call count and improve context.
    """
    # Allow disabling LLM relation extraction entirely (keeps rule-based relations).
    if (os.getenv("KG_DISABLE_LLM_RELATIONS", "0") or "0").strip() == "1":
        return []

    if not text or not text.strip():
        return []

    schema = _load_relation_schema()
    allowed_types = _schema_allowed_types(schema)
    ent_payload = _dedupe_entities_for_llm(entities)
    if len(ent_payload) < 2:
        return []

    allowed_canons = {_normalize_surface(e["canonical"]) for e in ent_payload if e.get("canonical")}

    user_payload = {
        "context": text,
        "entities": ent_payload,
        "allowed_relation_label_format": "UPPERCASE_WITH_UNDERSCORES",
        "min_confidence": float(schema.get("min_confidence", 0.35)),
    }

    prompt = f"{REL_EXTRACT_SYSTEM_PROMPT}\n\nInput:\n{json.dumps(user_payload, ensure_ascii=False)}"

    # Cache hit check
    cache_key = DiskJSONCache.hash_key("rel_v1", _text_digest(text), _text_digest(json.dumps(ent_payload)))
    if _SEMANTIC_REL_CACHE is not None:
        cached = _SEMANTIC_REL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        raw = genai_generate_text(None, prompt, temperature=0.1, purpose="RELATION")
    except Exception as exc:
        logging.warning("extract_relations_for_context: LLM call failed: %s", exc)
        raw = ""

    data = _parse_json_safely(raw, default=[])
    if not isinstance(data, list):
        return []
    
    cleaned: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("head", "relation", "tail")):
            continue

        head = _normalize_surface(str(item.get("head", "")))
        tail = _normalize_surface(str(item.get("tail", "")))
        raw_rel = str(item.get("relation", "") or "")
        rel = _normalize_relation_label(raw_rel, schema)
        if not head or not tail or head == tail or rel is None:
            continue

        if allowed_types and rel not in allowed_types:
            rel = "RELATED_TO"
            if allowed_types and rel not in allowed_types:
                continue

        if head not in allowed_canons or tail not in allowed_canons:
            continue

        conf = item.get("confidence", None)
        try:
            conf_f = float(conf) if conf is not None else 0.6
        except Exception:
            conf_f = 0.6
        if conf_f < float(schema.get("min_confidence", 0.35)):
            continue

        cleaned.append({
            "head": head,
            "relation": rel,
            "tail": tail,
            "confidence": conf_f,
        })
    
    if _SEMANTIC_REL_CACHE is not None:
        _SEMANTIC_REL_CACHE.set(cache_key, cleaned)

    return cleaned

def add_semantic_relation_edges(
    graph: KnowledgeGraph,
    all_entities_per_doc: Dict[str, List[Entity]],
    sent_index: Dict[str, SentenceInfo],
):
    strategy = (os.getenv("KG_EXTRACTION_STRATEGY", "cluster") or "cluster").strip().lower()
    if strategy == "oneshot":
        return _add_semantic_relation_edges_oneshot(graph, sent_index)

    # Build per-sentence entity lists from existing MENTION_IN edges (Phase 4), which is
    # far faster than scanning all sentences for every extracted entity.
    ent_node_to_min_entity: Dict[str, Entity] = {}
    for node_id, node in graph.nodes.items():
        # Include DOMAIN nodes too; they often represent key finance/legal concepts.
        if node.label not in {"ENTITY", "DOMAIN"}:
            continue
        canon = str((node.properties or {}).get("canonical") or "")
        etype = str((node.properties or {}).get("type") or "ENTITY")
        if not canon:
            continue
        ent_node_to_min_entity[node_id] = Entity(
            text=canon,
            label=etype,
            start=0,
            end=0,
            source="kg",
            canonical=canon,
            description=None,
            context=None,
            doc_id=None,
        )

    entities_per_sentence: Dict[str, List[Tuple[Entity, str]]] = {}
    for edge in graph.edges:
        if edge.type != "MENTION_IN":
            continue
        ent_id = edge.source
        sid = edge.target
        if sid not in sent_index:
            continue
        eobj = ent_node_to_min_entity.get(ent_id)
        if not eobj:
            continue
        entities_per_sentence.setdefault(sid, []).append((eobj, ent_id))

    # For each sentence, extract relations using a small cross-sentence context window.
    window_back = int(os.getenv("KG_RELATION_CONTEXT_BACK_SENTENCES", "1"))
    rel_workers = int(os.getenv("KG_RELATION_WORKERS", "0") or 0)
    if rel_workers <= 0:
        rel_workers = min(12, (os.cpu_count() or 4))

    # Dedupe by (source, type, target, evidence)
    def _edge_id_for(src: str, rtype: str, tgt: str, sid: str) -> str:
        h = hashlib.sha256(f"{src}||{rtype}||{tgt}||{sid}".encode("utf-8")).hexdigest()[:16]
        return f"e:rel:{h}"

    def _context_text_for_sid(sid: str) -> str:
        sentence = sent_index[sid]
        context_text = sentence.text
        if window_back > 0:
            try:
                parts = sid.split(":")
                doc_id = parts[1]
                idx = int(parts[2])
                prev_texts = []
                for j in range(max(0, idx - window_back), idx):
                    psid = f"sent:{doc_id}:{j}"
                    if psid in sent_index:
                        prev_texts.append(sent_index[psid].text)
                if prev_texts:
                    context_text = " ".join(prev_texts + [sentence.text])
            except Exception:
                pass
        return context_text

    def _extract_rules_only(sid: str, ent_list: List[Tuple[Entity, str]]) -> List[KGEdge]:
        if len(ent_list) < 2:
            return []
        context_text = _context_text_for_sid(sid)
        edges_out: List[KGEdge] = []

        # 1) Deterministic constitutional relations
        try:
            rule_rels = _rule_based_constitution_relations(context_text, ent_list)
        except Exception:
            rule_rels = []
        for rr in rule_rels:
            head_id = _canonical_to_node_id(rr.get("head", ""))
            tail_id = _canonical_to_node_id(rr.get("tail", ""))
            rtype = str(rr.get("relation") or "").strip().upper()
            if head_id in graph.nodes and tail_id in graph.nodes and rtype:
                edges_out.append(KGEdge(
                    id=_edge_id_for(head_id, rtype, tail_id, sid),
                    source=head_id, target=tail_id, type=rtype,
                    properties={"sentence_id": sid, "confidence": float(rr.get("confidence", 0.8)), "source": "rule"}
                ))

        # 1b) Deterministic finance/legal mechanics
        try:
            mech_rels = _rule_based_finance_legal_relations(context_text, ent_list)
        except Exception:
            mech_rels = []
        for rr in mech_rels:
            head_id = _canonical_to_node_id(rr.get("head", ""))
            tail_id = _canonical_to_node_id(rr.get("tail", ""))
            rtype = str(rr.get("relation") or "").strip().upper()
            if head_id in graph.nodes and tail_id in graph.nodes and rtype:
                edges_out.append(KGEdge(
                    id=_edge_id_for(head_id, rtype, tail_id, sid),
                    source=head_id, target=tail_id, type=rtype,
                    properties={"sentence_id": sid, "confidence": float(rr.get("confidence", 0.78)), "source": "rule"}
                ))
        return edges_out

    def _extract_llm_for_batch(batch_items: List[Tuple[str, List[Tuple[Entity, str]]]]) -> List[KGEdge]:
        if not batch_items:
            return []
        
        # Combine text and unique entities
        texts = []
        batch_entities: List[Tuple[Entity, str]] = []
        seen_entity_ids = set()
        for sid, elist in batch_items:
            texts.append(sent_index[sid].text)
            for e, nid in elist:
                if nid not in seen_entity_ids:
                    batch_entities.append((e, nid))
                    seen_entity_ids.add(nid)
        
        full_text = " ".join(texts)
        primary_sid = batch_items[0][0]
        
        rels = extract_relations_for_context(full_text, batch_entities, context_id=primary_sid)
        
        edges_out = []
        for r in rels:
            head_id = _canonical_to_node_id(r["head"])
            tail_id = _canonical_to_node_id(r["tail"])
            if head_id in graph.nodes and tail_id in graph.nodes:
                edges_out.append(KGEdge(
                    id=_edge_id_for(head_id, r["relation"], tail_id, primary_sid),
                    source=head_id, target=tail_id, type=r["relation"],
                    properties={"sentence_id": primary_sid, "confidence": r["confidence"], "source": "llm"}
                ))
        return edges_out

    # Parallel extraction
    created_edges: List[KGEdge] = []
    seen_rel_edges: Set[Tuple[str, str, str, str]] = set()
    
    # Sort items by SID to maintain document order for batching
    items = sorted(entities_per_sentence.items(), key=lambda x: x[0])

    # 1. Rule-based extraction (sequential, very fast)
    for sid, ent_list in items:
        for edge in _extract_rules_only(sid, ent_list):
            key = (edge.source, edge.type, edge.target, str(edge.properties.get("sentence_id")))
            if key not in seen_rel_edges:
                seen_rel_edges.add(key)
                graph.add_edge(edge)
                created_edges.append(edge)

    # 2. LLM-based extraction (batched & parallel)
    # Refined Optimization: Batch sentences first, then only skip batches with < 2 unique entities total.
    # This preserves cross-sentence relationships while still being much faster than processing every sentence.
    llm_batch_size = int(os.getenv("KG_RELATION_BATCH_SIZE", "10") or 10)
    
    batches = []
    current_batch = []
    current_batch_entities = set()
    
    for sid, elist in items:
        current_batch.append((sid, elist))
        for _, eid in elist:
            current_batch_entities.add(eid)
            
        if len(current_batch) >= llm_batch_size:
            if len(current_batch_entities) >= 2:
                batches.append(current_batch)
            current_batch = []
            current_batch_entities = set()
            
    if current_batch and len(current_batch_entities) >= 2:
        batches.append(current_batch)

    if verbose:
        total_skipped_batches = (len(items) // llm_batch_size + 1) - len(batches)
        print(f"[Phase 5] Quality Optimization: Grouped {len(items)} sentences into {len(batches)} multi-entity batches.")
        if total_skipped_batches > 0:
            print(f"[Phase 5] Quality Optimization: Skipped {total_skipped_batches} empty/low-signal batches.")

    def _worker_with_stagger(batch, index):
        if index > 0:
            time.sleep(random.uniform(0.1, 0.8) * min(index, 3))
        return _extract_llm_for_batch(batch)

    if rel_workers <= 1:
        for batch in batches:
            for edge in _extract_llm_for_batch(batch):
                key = (edge.source, edge.type, edge.target, str(edge.properties.get("sentence_id")))
                if key not in seen_rel_edges:
                    seen_rel_edges.add(key)
                    graph.add_edge(edge)
                    created_edges.append(edge)
    else:
        with ThreadPoolExecutor(max_workers=rel_workers) as ex:
            futs = {ex.submit(_worker_with_stagger, b, i): b for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                edges = fut.result() or []
                for edge in edges:
                    key = (edge.source, edge.type, edge.target, str(edge.properties.get("sentence_id")))
                    if key not in seen_rel_edges:
                        seen_rel_edges.add(key)
                        graph.add_edge(edge)
                        created_edges.append(edge)

    return created_edges


def _add_semantic_relation_edges_oneshot(graph: KnowledgeGraph, sent_index: Dict[str, SentenceInfo]) -> List[KGEdge]:
    """Helper to load pre-extracted relations from oneshot cache and add to graph."""
    created_edges: List[KGEdge] = []
    seen_rel_edges: Set[Tuple[str, str, str, str]] = set()
    
    # 1. Load entities and their aliases for better ID lookup (resilience to catalog merging)
    canon_to_id: Dict[str, str] = {}
    for nid, node in graph.nodes.items():
        if node.label in {"ENTITY", "DOMAIN", "ARTICLE", "PROVISION"}:
            props = node.properties or {}
            # Index canonical
            canon = props.get("canonical")
            if canon:
                canon_to_id[_normalize_surface(str(canon))] = nid
            # Index aliases (crucial for hits after catalog merges)
            aliases = props.get("aliases", [])
            if isinstance(aliases, (list, set, tuple)):
                for al in aliases:
                    canon_to_id[_normalize_surface(str(al))] = nid
    
    print(f"DEBUG: Relationship lookup table built with {len(canon_to_id)} surface-to-ID entries.")

    # 2. Iterate docs to find oneshot data
    doc_ids = set()
    for sid in sent_index:
        parts = sid.split(":")
        if len(parts) >= 2:
            doc_ids.add(parts[1])
            
    print(f"DEBUG: Scanned sentence index, doc_ids found: {doc_ids}")

    schema = _load_relation_schema()
    total_new_edges = 0
    
    for doc_id in doc_ids:
        # We need to find the digest for this doc_id to hit the cache
        # This is slightly tricky since we don't have the full text here easily.
        # But we can find all sentences for this doc and reconstruct or just hope 
        # that Phase 2 already populated the cache.
        # Alternatively, we could have extract_oneshot_kg_from_doc save a mapping 
        # of doc_id -> digest.
        
        # Heuristic: try to find the cache entry by doc_id alone if DiskJSONCache supports it
        # or use a doc_id specific cache.
        
        # Since DiskJSONCache is a simple mapping, let's use a secondary cache for doc_id -> latest_data
        cache_key = DiskJSONCache.hash_key("oneshot_doc_latest", doc_id)
        data = _ONESHOT_RELATION_CACHE.get(cache_key)
        
        if not data or not isinstance(data, dict):
            logging.warning(f"  -> No One-Shot data found in cache for {doc_id} in Phase 5.")
            print(f"DEBUG: No One-Shot data found for {doc_id} (used cache_key: {cache_key})")
            continue
            
        relations = data.get("relations", [])
        logging.info(f"  -> Found {len(relations)} raw relations in One-Shot cache for {doc_id}")
        print(f"DEBUG: {doc_id} - Found {len(relations)} raw relations in cache.")
        
        mapped_count = 0
        skipped_missing_node = 0
        skipped_bad_label = 0
        
        for rel_item in relations:
            head_raw = str(rel_item.get("head", ""))
            tail_raw = str(rel_item.get("tail", ""))
            
            head_norm = _normalize_surface(head_raw)
            tail_norm = _normalize_surface(tail_raw)
            rel_label = _normalize_relation_label(str(rel_item.get("relation", "")), schema)
            conf = rel_item.get("confidence", 0.6)
            evidence = rel_item.get("evidence_snippet", "")
            
            src_id = canon_to_id.get(head_norm)
            tgt_id = canon_to_id.get(tail_norm)
            
            # 2.5 Fuzzy Fallback Lookup
            lookup_keys = list(canon_to_id.keys())
            if not src_id and head_norm:
                matches = difflib.get_close_matches(head_norm, lookup_keys, n=1, cutoff=0.92)
                if matches:
                    src_id = canon_to_id[matches[0]]
                    
            if not tgt_id and tail_norm:
                matches = difflib.get_close_matches(tail_norm, lookup_keys, n=1, cutoff=0.92)
                if matches:
                    tgt_id = canon_to_id[matches[0]]

            # 3. Dynamic Synthesis: If head/tail still missing, create a virtual entity node
            if not src_id and head_raw:
                src_id = _canonical_to_node_id(head_raw)
                if src_id not in graph.nodes:
                    graph.add_node(KGNode(
                        id=src_id, label="ENTITY",
                        properties={
                            "canonical": head_raw, 
                            "type": "CONCEPT", 
                            "source": "oneshot_synthesis",
                            "confidence": "medium",
                            "description": f"Synthesized from a discovery relation in {doc_id}."
                        }
                    ))
                    # Register in lookup table too
                    canon_to_id[head_norm] = src_id
                    
            if not tgt_id and tail_raw:
                tgt_id = _canonical_to_node_id(tail_raw)
                if tgt_id not in graph.nodes:
                    graph.add_node(KGNode(
                        id=tgt_id, label="ENTITY",
                        properties={
                            "canonical": tail_raw, 
                            "type": "CONCEPT", 
                            "source": "oneshot_synthesis",
                            "confidence": "medium",
                            "description": f"Synthesized from a discovery relation in {doc_id}."
                        }
                    ))
                    # Register in lookup table too
                    canon_to_id[tail_norm] = tgt_id

            if not src_id or not tgt_id:
                skipped_missing_node += 1
                continue
            if not rel_label:
                skipped_bad_label += 1
                continue
                
            # Find best sentence_id for evidence
            best_sid = None
            if evidence:
                # Search evidence in sent_index
                ev_norm = evidence.lower()
                for sid, sinfo in sent_index.items():
                    if doc_id in sid and ev_norm in sinfo.text.lower():
                        best_sid = sid
                        break
            
            if not best_sid:
                # Fallback to any sentence from this doc
                for sid in sent_index:
                    if doc_id in sid:
                        best_sid = sid
                        break
                        
            if not best_sid:
                continue
                
            edge_id = f"e:rel:{hashlib.sha256(f'{src_id}||{rel_label}||{tgt_id}||{best_sid}'.encode('utf-8')).hexdigest()[:16]}"
            key = (src_id, rel_label, tgt_id, best_sid)
            if key not in seen_rel_edges:
                seen_rel_edges.add(key)
                edge = KGEdge(
                    id=edge_id,
                    source=src_id,
                    target=tgt_id,
                    type=rel_label,
                    properties={
                        "confidence": conf,
                        "sentence_id": best_sid,
                        "evidence": evidence,
                        "source": "oneshot"
                    }
                )
                graph.add_edge(edge)
                created_edges.append(edge)
                mapped_count += 1
                total_new_edges += 1
                
        status_msg = f"  -> {doc_id}: Mapped {mapped_count} rels. (Skipped: {skipped_missing_node} untracked nodes, {skipped_bad_label} bad labels)"
        logging.info(status_msg)
        print(f"DEBUG: {status_msg}")
                
    print(f"DEBUG: _add_semantic_relation_edges_oneshot finished. Added {total_new_edges} edges.")
    return created_edges
