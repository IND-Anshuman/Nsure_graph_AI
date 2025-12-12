from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from functools import lru_cache
from dotenv import load_dotenv

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Importing coreferee registers its pipeline component with spaCy
import coreferee  # type: ignore  # noqa: F401

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class BaseSchema(BaseModel):
    """Base Pydantic model with relaxed extra-handling."""

    class Config:
        extra = "ignore"


class EntityMention(BaseSchema):
    """A single surface mention of an entity in the text."""

    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    is_pronoun: Optional[bool] = None
    sentence_index: Optional[int] = None


class Entity(BaseSchema):
    """Canonical entity with aggregated mentions."""

    id: str
    name: str
    type: str = Field(..., description="Coarse type label, e.g. PERSON, ORG, LOC, PRODUCT, EVENT, etc.")
    mentions: List[EntityMention] = Field(default_factory=list)


class Relation(BaseSchema):
    """Relation between two entities in subject–predicate–object form."""

    source_entity_id: str
    relation_type: str
    target_entity_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None


class EventArgument(BaseSchema):
    """A semantic role argument for an event."""

    role: str
    entity_id: str


class Event(BaseSchema):
    """Event with participants and optional time."""

    id: str
    type: str
    trigger: str
    arguments: List[EventArgument] = Field(default_factory=list)
    time: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None


class Intent(BaseSchema):
    """Local discourse intent of the text chunk."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ExtractionResult(BaseSchema):
    """Top-level structured output."""

    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    intent: Intent
    span_metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Model + client initialization
# ---------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

DEFAULT_SPACY_MODEL = os.getenv("ENTITY_AGENT_SPACY_MODEL", "en_core_web_trf")
FALLBACK_SPACY_MODEL = os.getenv("ENTITY_AGENT_SPACY_FALLBACK_MODEL", "en_core_web_sm")
OPENAI_MODEL = os.getenv("ENTITY_AGENT_OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """
    Lazily load spaCy pipeline with coreferee coreference component.

    Tries a transformer model first, falls back to a small model if needed.
    """
    logger.info("Loading spaCy model '%s' (fallback '%s')", DEFAULT_SPACY_MODEL, FALLBACK_SPACY_MODEL)
    try:
        nlp = spacy.load(DEFAULT_SPACY_MODEL)
    except OSError:
        logger.warning("Failed to load '%s', falling back to '%s'", DEFAULT_SPACY_MODEL, FALLBACK_SPACY_MODEL)
        nlp = spacy.load(FALLBACK_SPACY_MODEL)

    # Attach coreferee if available
    try:
        if "coreferee" not in nlp.pipe_names:
            logger.info("Adding 'coreferee' component to spaCy pipeline")
            nlp.add_pipe("coreferee")
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.exception("Failed to add 'coreferee' to spaCy pipeline: %s", exc)

    return nlp


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Lazily initialize the OpenAI client using environment variable OPENAI_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    logger.info("Initializing OpenAI client for Entity & Semantic Extraction Agent")
    client = OpenAI(api_key=api_key)
    return client


# ---------------------------------------------------------------------------
# Local NLP helpers (NER + coreference)
# ---------------------------------------------------------------------------


def _build_token_sentence_index(doc: Doc) -> Dict[int, int]:
    """
    Map token index -> sentence index for the document.
    """
    mapping: Dict[int, int] = {}
    for sent_idx, sent in enumerate(doc.sents):
        for token in sent:
            mapping[token.i] = sent_idx
    return mapping


def _extract_preliminary_entities(doc: Doc, token_to_sent: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Extract basic NER entities from spaCy Doc into a JSON-like representation that
    will be given to the LLM for normalization and consolidation.
    """
    prelim_entities: List[Dict[str, Any]] = []
    for idx, ent in enumerate(doc.ents):
        eid = f"E{idx}"
        prelim_entities.append(
            {
                "id": eid,
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "sentence_index": token_to_sent.get(ent.start),
            }
        )

    logger.info("Extracted %d preliminary NER entities", len(prelim_entities))
    return prelim_entities


def _build_token_to_entity_map(doc: Doc, prelim_entities: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Build a mapping from token index to one or more preliminary entity IDs
    based on spaCy NER spans.
    """
    token_to_entities: Dict[int, List[str]] = {}
    span_by_id: Dict[str, Tuple[int, int]] = {}

    for ent in prelim_entities:
        span_by_id[ent["id"]] = (ent["start_char"], ent["end_char"])

    for ent in prelim_entities:
        eid = ent["id"]
        start_char, end_char = ent["start_char"], ent["end_char"]
        if start_char is None or end_char is None:
            continue
        # assign all tokens whose char span overlaps the entity char span
        for token in doc:
            t_start, t_end = token.idx, token.idx + len(token.text)
            if not (t_end <= start_char or t_start >= end_char):  # overlap
                token_to_entities.setdefault(token.i, []).append(eid)

    return token_to_entities


def _extract_coref_clusters(
    doc: Doc, token_to_sent: Dict[int, int], token_to_entities: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
    """
    Extract coreference clusters using coreferee, providing text spans and links
    back to preliminary entity IDs where possible.
    """
    clusters: List[Dict[str, Any]] = []

    try:
        coref_chains = getattr(doc._, "coref_chains", None)
    except AttributeError:
        logger.warning("Document has no 'coref_chains' attribute; coreference disabled.")
        return clusters

    if not coref_chains:
        logger.info("No coreference chains detected.")
        return clusters

    for chain_idx, chain in enumerate(coref_chains):
        mentions: List[Dict[str, Any]] = []
        for mention in chain:
            # Each mention is a list of token indices
            indices: List[int] = list(mention)
            if not indices:
                continue
            tokens = [doc[i] for i in indices]
            span = doc[tokens[0].i : tokens[-1].i + 1]

            linked_entity_ids = set()
            for t in tokens:
                for eid in token_to_entities.get(t.i, []):
                    linked_entity_ids.add(eid)

            mentions.append(
                {
                    "text": span.text,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "is_pronoun": all(t.pos_ == "PRON" for t in tokens),
                    "sentence_index": token_to_sent.get(tokens[0].i),
                    "linked_entity_ids": sorted(linked_entity_ids),
                }
            )

        clusters.append({"id": f"C{chain_idx}", "mentions": mentions})

    logger.info("Extracted %d coreference clusters", len(clusters))
    return clusters


def _build_coref_resolved_text(doc: Doc) -> Optional[str]:
    """
    Build a coreference-resolved version of the text using coreferee's resolve method.

    This is primarily for giving the LLM extra context and doesn't have to be
    perfectly grammatical.
    """
    try:
        coref_chains = getattr(doc._, "coref_chains", None)
    except AttributeError:
        return None

    if not coref_chains:
        return None

    resolved_tokens: List[str] = []
    for token in doc:
        try:
            replacements = coref_chains.resolve(token)
        except Exception:  # pragma: no cover - defensive
            replacements = None

        if replacements:
            resolved_tokens.append(" ".join(t.text for t in replacements))
        else:
            resolved_tokens.append(token.text)

    # Simple join, only used for LLM prompt
    resolved_text = " ".join(resolved_tokens)
    return resolved_text.strip()


# ---------------------------------------------------------------------------
# LLM prompt + parsing
# ---------------------------------------------------------------------------


def _build_llm_prompt_payload(
    text: str,
    prelim_entities: List[Dict[str, Any]],
    coref_clusters: List[Dict[str, Any]],
    coref_resolved_text: Optional[str],
) -> str:
    """
    Build the user message content sent to the LLM with all the local NLP annotations.
    """
    entities_json = json.dumps(prelim_entities, ensure_ascii=False)
    clusters_json = json.dumps(coref_clusters, ensure_ascii=False)
    resolved = coref_resolved_text if coref_resolved_text is not None else "N/A"

    prompt = f"""
You are an Entity & Semantic Extraction Agent for a knowledge graph.

Your job is to read a text chunk and its pre-computed NLP annotations, then
return a concise JSON object describing:
- entities
- relations
- events
- local intent

### INPUT TEXT

```text
{text}

COREFERENCE-RESOLVED TEXT (approximate, may be noisy)
{resolved}

PRELIMINARY NER ENTITIES (from spaCy)

These entities were extracted by a classical NER system and should guide your
normalization and ID assignment. You SHOULD reuse these IDs where appropriate.
{entities_json}
Each preliminary entity has:

id: stable ID (e.g. "E0")

text: surface span

label: NER label like PERSON, ORG, GPE, PRODUCT, EVENT, etc.

start_char, end_char: character offsets in the original text

sentence_index: index of the sentence containing the entity

COREFERENCE CLUSTERS (from coreferee)

These clusters group pronouns and noun phrases that refer to the same entity.
{clusters_json}

Each cluster has:

id: cluster ID (e.g. "C0")

mentions: list of mentions, each with:

text

start_char, end_char

is_pronoun (true/false)

sentence_index

linked_entity_ids: preliminary entity IDs that overlap this mention, if any

Use these clusters to:

Link pronouns like "he", "she", "they", "it", "the company" back to the correct entity.

Merge duplicate mentions into a single canonical entity.

REQUIRED OUTPUT

Return a SINGLE JSON OBJECT with the following top-level keys:

"entities": list of entities

"relations": list of relations

"events": list of events

"intent": intent object

entities

Each entity object MUST look like:

{{
"id": "E0",
"name": "OpenAI",
"type": "ORG",
"mentions": [
{{
"text": "OpenAI",
"start_char": 10,
"end_char": 16,
"is_pronoun": false,
"sentence_index": 0
}}
]
}}

Rules:

Reuse preliminary entity IDs ("id") whenever they represent the same real-world entity.

If you need to introduce NEW entities, give them IDs like "E_NEW1", "E_NEW2", etc.

"type" should be a coarse label like PERSON, ORG, LOC, PRODUCT, EVENT, POLICY, LAW, etc.

"mentions" should aggregate all important surface forms and pronoun mentions
(using the coreference clusters).

relations

Each relation represents a triple (subject, predicate, object):

{{
"source_entity_id": "E1",
"relation_type": "FOUNDED_BY",
"target_entity_id": "E2",
"confidence": 0.92,
"evidence": "OpenAI was founded by Elon Musk and others."
}}

Rules:

Always reference existing entity IDs in source_entity_id and target_entity_id.

"relation_type" should be a compact label like FOUNDED_BY, ACQUIRED, PART_OF, LOCATED_IN, EMPLOYS, etc.

"confidence" MUST be between 0 and 1.

events

Each event describes who did what to whom:

{{
"id": "EV1",
"type": "ACQUISITION",
"trigger": "acquired",
"arguments": [
{{"role": "BUYER", "entity_id": "E3"}},
{{"role": "TARGET", "entity_id": "E4"}}
],
"time": "2023-11-30",
"confidence": 0.88,
"evidence": "Company A acquired Company B in late 2023."
}}

Rules:

"type" is a coarse label like PURCHASE, MERGER, ANNOUNCEMENT, LAUNCH, INCIDENT, LAWSUIT, etc.

"trigger" is the main verb or phrase that evokes the event.

"arguments" is a list of role/entity pairs. Roles might be AGENT, PATIENT, BUYER, SELLER, TARGET, LOCATION, etc.

All "entity_id" fields MUST reference existing entity IDs.

"time" may be a date string, phrase ("last week"), or null.

"confidence" MUST be between 0 and 1.

intent

The "intent" object MUST look like:

{{
"label": "INFORM",
"confidence": 0.9,
"rationale": "The text neutrally describes factual information without trying to persuade."
}}

Allowed labels (choose ONE):

INFORM

PROMOTE

WARN

REQUEST

NEGOTIATE

CRITICIZE

PRAISE

OTHER

IMPORTANT JSON RULES

Your response MUST be valid JSON.

Do NOT include any commentary, markdown, or explanation outside the JSON.

Do NOT include trailing commas.

Do NOT include code fences.

Use double quotes for all JSON keys and string values.

Return ONLY the final JSON object.
"""
    return "\n".join(line.rstrip() for line in prompt.splitlines())

def _call_llm_for_extraction(
    text: str,
    prelim_entities: List[Dict[str, Any]],
    coref_clusters: List[Dict[str, Any]],
    coref_resolved_text: Optional[str],
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions API to get structured extraction JSON.
    """
    client = get_openai_client()
    user_content = _build_llm_prompt_payload(text, prelim_entities, coref_clusters, coref_resolved_text)
    logger.info("Calling OpenAI model '%s' for semantic extraction", OPENAI_MODEL)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise information extraction engine that ALWAYS outputs strict JSON.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
    # Fallback: try to extract JSON substring
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.error("Failed to parse JSON from LLM response: %s", content)
            raise
        substring = content[start : end + 1]
        parsed = json.loads(substring)

    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object.")

    return parsed

def _build_result_from_llm_json(
    raw: Dict[str, Any],
    span_metadata: Dict[str, Any],
) -> ExtractionResult:
    """
    Convert the raw JSON dict from the LLM into a validated ExtractionResult.
    """
    entities_raw = raw.get("entities", []) or []
    relations_raw = raw.get("relations", []) or []
    events_raw = raw.get("events", []) or []
    intent_raw = raw.get("intent", {}) or {}

    # TODO: Add implementation to build and return ExtractionResult
    entities: List[Entity] = []
    relations: List[Relation] = []
    events: List[Event] = []

    # Entities
    for e in entities_raw:
        try:
            entities.append(Entity(**e))
        except ValidationError as exc:
            logger.warning("Skipping malformed entity from LLM: %s", exc)

    # Relations
    for r in relations_raw:
        try:
            relations.append(Relation(**r))
        except ValidationError as exc:
            logger.warning("Skipping malformed relation from LLM: %s", exc)

    # Events
    for ev in events_raw:
        try:
            events.append(Event(**ev))
        except ValidationError as exc:
            logger.warning("Skipping malformed event from LLM: %s", exc)

    # Intent (fallback to OTHER if invalid)
    try:
        intent = Intent(**intent_raw)
    except ValidationError as exc:
        logger.warning("Malformed intent from LLM, using fallback: %s", exc)
        intent = Intent(
            label="OTHER",
            confidence=0.0,
            rationale="Intent could not be parsed from LLM output.",
        )

    result = ExtractionResult(
        entities=entities,
        relations=relations,
        events=events,
        intent=intent,
        span_metadata=span_metadata,
    )
    return result

def extract_entities_semantics(
    text: str,
    doc_id: str | None = None,
    chunk_id: str | None = None,
) -> Dict[str, Any]:
    """
Run the Entity & Semantic Extraction Agent on a single text chunk.

Parameters
----------
text:
    Raw text chunk to analyze.
doc_id:
    Optional identifier for the source document.
chunk_id:
    Optional identifier for this specific chunk within the document.

Returns
-------
dict
    JSON-serializable dictionary with keys:
    - entities: List[Entity]
    - relations: List[Relation]
    - events: List[Event]
    - intent: Intent
    - span_metadata: metadata including source_doc_id, chunk_id, etc.
"""
    logger.info("Starting entity & semantic extraction (doc_id=%s, chunk_id=%s)", doc_id, chunk_id)

    nlp = get_nlp()
    doc = nlp(text)

    token_to_sent = _build_token_sentence_index(doc)
    prelim_entities = _extract_preliminary_entities(doc, token_to_sent)
    token_to_entities = _build_token_to_entity_map(doc, prelim_entities)
    coref_clusters = _extract_coref_clusters(doc, token_to_sent, token_to_entities)
    coref_resolved_text = _build_coref_resolved_text(doc)

    span_metadata: Dict[str, Any] = {
        "source_doc_id": doc_id,
        "chunk_id": chunk_id,
        "text_length": len(text),
        "num_sentences": sum(1 for _ in doc.sents),
    }

    try:
        llm_raw = _call_llm_for_extraction(
            text=text,
            prelim_entities=prelim_entities,
            coref_clusters=coref_clusters,
            coref_resolved_text=coref_resolved_text,
        )
        result = _build_result_from_llm_json(llm_raw, span_metadata)
    except Exception as exc:  # pragma: no cover - network/environment dependent
        logger.exception("LLM-based extraction failed; returning fallback structure: %s", exc)
        # Fallback: empty structure with error intent
        intent = Intent(
            label="OTHER",
            confidence=0.0,
            rationale=f"LLM extraction failed: {exc}",
        )
        result = ExtractionResult(
            entities=[],
            relations=[],
            events=[],
            intent=intent,
            span_metadata=span_metadata,
        )

    logger.info(
        "Extraction complete: %d entities, %d relations, %d events, intent=%s",
        len(result.entities),
        len(result.relations),
        len(result.events),
        result.intent.label,
    )

    return result.dict()
