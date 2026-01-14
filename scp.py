"""
SCP (Symbolic Consistency Probing)
=================================

A hypergraph-based hallucination detection prototype:
- Stores knowledge as a hypergraph (relation nodes binding entities)
- Extracts claims from text as (subject, predicate, object) triples
- Probes claims against the KB using pluggable similarity backends
- Returns verdicts and (optional) proof subgraphs

This module is intentionally usable without API keys.

Optional upgrades:
- Better embeddings: install `sentence-transformers` (downloads models from HuggingFace)
- Better claim extraction: plug in an LLM function (OpenAI/Anthropic/local model) to `LLMExtractor`
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
from difflib import SequenceMatcher

# Try to import sentence-transformers; fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    EMBEDDINGS_AVAILABLE = False
    np = None
    SentenceTransformer = None


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


class Verdict(Enum):
    PASS = "PASS"
    SOFT_PASS = "SOFT_PASS"
    FAIL = "FAIL"
    CONTRADICT = "CONTRADICT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Claim:
    """A single extracted claim (subject-predicate-object triple)."""

    subject: str
    predicate: str
    obj: str
    raw: str  # Original sentence
    confidence: float = 1.0  # Extraction confidence

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.obj})"


@dataclass
class ProbeResult:
    """Result of probing a single claim against the KB."""

    claim: Claim
    verdict: Verdict
    score: float  # 0..1 similarity/confidence
    matched_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    proof_subgraph: Optional[nx.DiGraph] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": {
                "subject": self.claim.subject,
                "predicate": self.claim.predicate,
                "object": self.claim.obj,
                "raw": self.claim.raw,
            },
            "verdict": self.verdict.value,
            "score": round(self.score, 4),
            "matched_facts": self.matched_facts,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class SCPReport:
    """Full report from probing an answer."""

    original_text: str
    claims_extracted: int
    results: List[ProbeResult]
    overall_score: float  # Aggregate consistency score
    pass_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "claims_extracted": self.claims_extracted,
            "overall_score": round(self.overall_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# EMBEDDING / SIMILARITY BACKENDS (Pluggable)
# =============================================================================


class EmbeddingBackend(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embedding vectors."""

    @abstractmethod
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute similarity between two vectors."""

    def text_similarity(self, a: str, b: str) -> float:
        """Convenience method."""
        vecs = self.encode([a, b])
        return self.similarity(vecs[0], vecs[1])


class StringSimilarityBackend(EmbeddingBackend):
    """Fallback: uses pure string similarity (no ML dependencies)."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        return 0.0

    def text_similarity(self, a: str, b: str) -> float:
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    def _normalize(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r'[""\"\'`,.;:!?()\[\]{}]', "", s)
        return s


class HashingEmbeddingBackend(EmbeddingBackend):
    """
    Lightweight local "embedding" backend (no extra deps).

    It produces deterministic hashed bag-of-words vectors and cosine similarity.
    This is not as strong as real embedding models, but is far more robust than
    raw string edit distance for many paraphrases.
    """

    def __init__(self, dim: int = 512):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_one(t) for t in texts]

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        # Dense vectors for simplicity; dim is small by default
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            na += a * a
            nb += b * b
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def _encode_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            idx = self._stable_hash(tok) % self.dim
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def _tokens(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9_ ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        # mix unigrams + simple bigrams to improve phrase matching
        words = text.split(" ")
        toks = list(words)
        toks.extend([f"{a}_{b}" for a, b in zip(words, words[1:])])
        return toks

    def _stable_hash(self, s: str) -> int:
        # Python's built-in hash() is randomized per-process; use md5 for determinism.
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


class SentenceTransformerBackend(EmbeddingBackend):
    """Backend using sentence-transformers (requires extra deps + model download)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, List[float]] = {}

    def encode(self, texts: List[str]) -> List[List[float]]:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = self.model.encode(uncached, convert_to_numpy=True)
            for t, emb in zip(uncached, embeddings):
                self._cache[t] = emb.tolist()
        return [self._cache[t] for t in texts]

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =============================================================================
# CLAIM EXTRACTORS (Pluggable)
# =============================================================================


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[Claim]:
        """Extract claims from text."""


class RuleBasedExtractor(ClaimExtractor):
    """Pattern-based claim extraction (fast, no dependencies)."""

    PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("located_in", re.compile(r"^(?P<s>.+?)\s+(?:is|are|was|were)\s+(?:located|situated)\s+in\s+(?P<o>.+?)$", re.I)),
        ("founded_in", re.compile(r"^(?P<s>.+?)\s+(?:was|were)\s+founded\s+in\s+(?P<o>.+?)$", re.I)),
        ("born_in", re.compile(r"^(?P<s>.+?)\s+(?:was|were)\s+born\s+in\s+(?P<o>.+?)$", re.I)),
        ("died_in", re.compile(r"^(?P<s>.+?)\s+died\s+in\s+(?P<o>.+?)$", re.I)),
        ("discovered", re.compile(r"^(?P<s>.+?)\s+discovered\s+(?P<o>.+?)$", re.I)),
        ("invented", re.compile(r"^(?P<s>.+?)\s+invented\s+(?P<o>.+?)$", re.I)),
        ("proposed", re.compile(r"^(?P<s>.+?)\s+proposed\s+(?P<o>.+?)$", re.I)),
        ("developed", re.compile(r"^(?P<s>.+?)\s+developed\s+(?P<o>.+?)$", re.I)),
        ("created", re.compile(r"^(?P<s>.+?)\s+created\s+(?P<o>.+?)$", re.I)),
        ("wrote", re.compile(r"^(?P<s>.+?)\s+wrote\s+(?P<o>.+?)$", re.I)),
        ("published", re.compile(r"^(?P<s>.+?)\s+published\s+(?P<o>.+?)$", re.I)),
        ("capital_of", re.compile(r"^(?P<s>.+?)\s+is\s+the\s+capital\s+of\s+(?P<o>.+?)$", re.I)),
        ("part_of", re.compile(r"^(?P<s>.+?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<o>.+?)$", re.I)),
        ("member_of", re.compile(r"^(?P<s>.+?)\s+is\s+(?:a\s+)?member\s+of\s+(?P<o>.+?)$", re.I)),
        ("has", re.compile(r"^(?P<s>.+?)\s+has\s+(?P<o>.+?)$", re.I)),
        ("contains", re.compile(r"^(?P<s>.+?)\s+contains\s+(?P<o>.+?)$", re.I)),
        ("is_a", re.compile(r"^(?P<s>.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<o>.+?)$", re.I)),
        ("is", re.compile(r"^(?P<s>.+?)\s+(?:is|are)\s+(?P<o>.+?)$", re.I)),
        ("was", re.compile(r"^(?P<s>.+?)\s+(?:was|were)\s+(?P<o>.+?)$", re.I)),
    ]

    def extract(self, text: str) -> List[Claim]:
        claims: List[Claim] = []
        for sent in self._split_sentences(text):
            claim = self._extract_from_sentence(sent)
            if claim:
                claims.append(claim)
        return claims

    def _split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        # Handle common abbreviations
        text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.\s", r"\1<PERIOD> ", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.replace("<PERIOD>", ".").strip() for p in parts if p.strip()]

    def _extract_from_sentence(self, sent: str) -> Optional[Claim]:
        s = sent.strip()
        s = re.sub(r"[.!?]+$", "", s).strip()
        for pred, pattern in self.PATTERNS:
            match = pattern.match(s)
            if not match:
                continue
            subj = match.group("s").strip()
            obj = match.group("o").strip()
            if len(subj) < 2 or len(obj) < 2:
                continue
            return Claim(
                subject=subj,
                predicate=pred,
                obj=obj,
                raw=sent,
                confidence=0.7,
            )
        return None


class LLMExtractor(ClaimExtractor):
    """LLM-based claim extraction (requires you to provide an LLM function)."""

    SYSTEM_PROMPT = """You are a claim extractor. Given text, extract factual claims as subject-predicate-object triples.

Output JSON array of objects with keys: subject, predicate, object, confidence (0-1).

Rules:
- Extract only factual claims, not opinions
- Use simple, normalized predicates (e.g., "located_in", "discovered", "is_a")
- confidence = how certain the claim is stated (not whether it's true)
- If no claims found, return []
"""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.llm_fn = llm_fn
        self._fallback = RuleBasedExtractor()

    def extract(self, text: str) -> List[Claim]:
        if not self.llm_fn:
            return self._fallback.extract(text)

        prompt = f"{self.SYSTEM_PROMPT}\n\nInput: {text}\nOutput:"
        try:
            response = self.llm_fn(prompt)
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if not json_match:
                return self._fallback.extract(text)
            data = json.loads(json_match.group())
            claims: List[Claim] = []
            for item in data:
                claims.append(
                    Claim(
                        subject=item.get("subject", ""),
                        predicate=item.get("predicate", "unknown"),
                        obj=item.get("object", ""),
                        raw=text,
                        confidence=float(item.get("confidence", 0.8)),
                    )
                )
            return claims
        except Exception:
            return self._fallback.extract(text)


class HybridExtractor(ClaimExtractor):
    """Combines rule-based and LLM extraction (prefers LLM output when available)."""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = LLMExtractor(llm_fn)

    def extract(self, text: str) -> List[Claim]:
        rule_claims = self.rule_extractor.extract(text)
        if self.llm_extractor.llm_fn:
            llm_claims = self.llm_extractor.extract(text)
            return self._merge_claims(rule_claims, llm_claims)
        return rule_claims

    def _merge_claims(self, rule_claims: List[Claim], llm_claims: List[Claim]) -> List[Claim]:
        merged = list(llm_claims)
        seen = {self._signature(c) for c in llm_claims}
        for rc in rule_claims:
            sig = self._signature(rc)
            if sig not in seen:
                merged.append(rc)
                seen.add(sig)
        return merged

    def _signature(self, claim: Claim) -> str:
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.lower().strip())

        return f"{norm(claim.subject)}|{norm(claim.predicate)}|{norm(claim.obj)}"


# =============================================================================
# HYPERGRAPH KNOWLEDGE BASE
# =============================================================================


class HyperKB:
    """
    Hypergraph-style Knowledge Base using NetworkX.

    Structure:
      - Entity nodes: "ent:<normalized_name>"
      - Relation nodes: "rel:<uuid>" with predicate attributes
      - Edges: subject --subj--> relation --obj--> object
    """

    def __init__(self, embedding_backend: Optional[EmbeddingBackend] = None):
        self.g = nx.DiGraph()
        self.backend = embedding_backend or HashingEmbeddingBackend()
        self._fact_cache: Optional[List[Tuple[str, str, str, str]]] = None  # (s, p, o, rel_id)

    def _norm_text(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r'[""\"\'`,.;:!?()\[\]{}]', "", s)
        return s

    def _canon_predicate(self, p: str) -> str:
        """
        Canonicalize predicate strings so KB and extractors can meet in the middle.
        Examples: "is_capital_of" -> "capital_of"
        """
        p = self._norm_text(p).replace(" ", "_")
        p = re.sub(r"^(is|was|are|were|be)_", "", p)
        return p

    def add_entity(self, name: str, **attrs: Any) -> str:
        eid = f"ent:{self._norm_text(name)}"
        if not self.g.has_node(eid):
            self.g.add_node(eid, kind="entity", label=name, **attrs)
        self._fact_cache = None
        return eid

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        source: str = "kb",
        **metadata: Any,
    ) -> str:
        s_id = self.add_entity(subject)
        o_id = self.add_entity(obj)

        r_id = f"rel:{uuid.uuid4().hex[:12]}"
        self.g.add_node(
            r_id,
            kind="relation",
            predicate=self._canon_predicate(predicate),
            predicate_label=predicate,
            confidence=float(confidence),
            source=source,
            **metadata,
        )
        self.g.add_edge(s_id, r_id, role="subj")
        self.g.add_edge(r_id, o_id, role="obj")
        self._fact_cache = None
        return r_id

    def add_facts_bulk(self, facts: List[Tuple[str, str, str]], **kwargs: Any) -> List[str]:
        return [self.add_fact(s, p, o, **kwargs) for s, p, o in facts]

    def iter_facts(self) -> List[Tuple[str, str, str, str]]:
        if self._fact_cache is not None:
            return self._fact_cache

        facts: List[Tuple[str, str, str, str]] = []
        for r_id, attrs in self.g.nodes(data=True):
            if attrs.get("kind") != "relation":
                continue

            pred = attrs.get("predicate", attrs.get("predicate_label", ""))

            subjects = [
                u
                for u, _ in self.g.in_edges(r_id)
                if self.g.edges[u, r_id].get("role") == "subj"
            ]
            objects = [
                v
                for _, v in self.g.out_edges(r_id)
                if self.g.edges[r_id, v].get("role") == "obj"
            ]
            for s in subjects:
                for o in objects:
                    s_label = self.g.nodes[s].get("label", s)
                    o_label = self.g.nodes[o].get("label", o)
                    facts.append((s_label, pred, o_label, r_id))

        self._fact_cache = facts
        return facts

    def has_fact_exact(self, claim: Claim) -> Optional[str]:
        c_s = self._norm_text(claim.subject)
        c_p = self._canon_predicate(claim.predicate)
        c_o = self._norm_text(claim.obj)
        for s, p, o, r_id in self.iter_facts():
            if self._norm_text(s) == c_s and self._canon_predicate(p) == c_p and self._norm_text(o) == c_o:
                return r_id
        return None

    def find_soft_matches(
        self, claim: Claim, top_k: int = 3, threshold: float = 0.0
    ) -> List[Tuple[float, Tuple[str, str, str], str]]:
        matches: List[Tuple[float, Tuple[str, str, str], str]] = []

        for s, p, o, r_id in self.iter_facts():
            s_sim = self.backend.text_similarity(claim.subject, s)
            p_sim = self.backend.text_similarity(self._canon_predicate(claim.predicate), self._canon_predicate(p))
            o_sim = self.backend.text_similarity(claim.obj, o)

            score = (0.30 * s_sim) + (0.45 * p_sim) + (0.25 * o_sim)
            if score >= threshold:
                matches.append((score, (s, p, o), r_id))

        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]

    def find_contradictions(self, claim: Claim) -> List[Tuple[str, str, str, str]]:
        contradictions: List[Tuple[str, str, str, str]] = []
        c_s = self._norm_text(claim.subject)
        c_p = self._canon_predicate(claim.predicate)
        c_o = self._norm_text(claim.obj)

        for s, p, o, r_id in self.iter_facts():
            if self._norm_text(s) == c_s and self._canon_predicate(p) == c_p and self._norm_text(o) != c_o:
                contradictions.append((s, p, o, r_id))
        return contradictions

    def get_proof_subgraph(self, relation_ids: List[str]) -> nx.DiGraph:
        nodes_to_include: set[str] = set()
        for r_id in relation_ids:
            if not self.g.has_node(r_id):
                continue
            nodes_to_include.add(r_id)
            for u, _ in self.g.in_edges(r_id):
                nodes_to_include.add(u)
            for _, v in self.g.out_edges(r_id):
                nodes_to_include.add(v)
        return self.g.subgraph(nodes_to_include).copy()

    def stats(self) -> Dict[str, int]:
        entities = sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == "entity")
        relations = sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == "relation")
        return {"entities": entities, "relations": relations, "edges": self.g.number_of_edges()}


# =============================================================================
# SCP PROBER (Main Engine)
# =============================================================================


class SCPProber:
    """Extracts claims and probes them against a HyperKB."""

    def __init__(
        self,
        kb: HyperKB,
        extractor: Optional[ClaimExtractor] = None,
        *,
        soft_threshold: float = 0.75,
        contradiction_check: bool = True,
    ):
        self.kb = kb
        self.extractor = extractor or RuleBasedExtractor()
        self.soft_threshold = soft_threshold
        self.contradiction_check = contradiction_check

    def probe(self, text: str) -> SCPReport:
        claims = self.extractor.extract(text)
        results: List[ProbeResult] = []

        if not claims:
            results.append(
                ProbeResult(
                    claim=Claim("", "unknown", "", raw=text),
                    verdict=Verdict.UNKNOWN,
                    score=0.0,
                    reason="No claims could be extracted from text.",
                )
            )
        else:
            for claim in claims:
                results.append(self._probe_claim(claim))

        if results and results[0].verdict != Verdict.UNKNOWN:
            scores = [r.score for r in results]
            verdicts = [r.verdict for r in results]
            overall_score = sum(scores) / len(scores)
            pass_count = sum(1 for v in verdicts if v in (Verdict.PASS, Verdict.SOFT_PASS))
            pass_rate = pass_count / len(verdicts)
        else:
            overall_score = 0.0
            pass_rate = 0.0

        return SCPReport(
            original_text=text,
            claims_extracted=len(claims),
            results=results,
            overall_score=overall_score,
            pass_rate=pass_rate,
        )

    def _probe_claim(self, claim: Claim) -> ProbeResult:
        exact_match = self.kb.has_fact_exact(claim)
        if exact_match:
            return ProbeResult(
                claim=claim,
                verdict=Verdict.PASS,
                score=1.0,
                matched_facts=[claim.to_tuple()],
                proof_subgraph=self.kb.get_proof_subgraph([exact_match]),
                reason="Exact match found in KB.",
                metadata={"match_type": "exact", "relation_id": exact_match},
            )

        if self.contradiction_check:
            contradictions = self.kb.find_contradictions(claim)
            if contradictions:
                return ProbeResult(
                    claim=claim,
                    verdict=Verdict.CONTRADICT,
                    score=0.0,
                    matched_facts=[(s, p, o) for s, p, o, _ in contradictions],
                    reason=f"Contradicts {len(contradictions)} fact(s) in KB.",
                    metadata={"contradictions": contradictions},
                )

        soft_matches = self.kb.find_soft_matches(claim, top_k=3, threshold=self.soft_threshold)
        if soft_matches:
            best_score, best_fact, best_rel_id = soft_matches[0]
            return ProbeResult(
                claim=claim,
                verdict=Verdict.SOFT_PASS,
                score=best_score,
                matched_facts=[best_fact],
                proof_subgraph=self.kb.get_proof_subgraph([best_rel_id]),
                reason=f"Soft match (score={best_score:.3f}) above threshold={self.soft_threshold}.",
                metadata={
                    "match_type": "soft",
                    "relation_id": best_rel_id,
                    "all_matches": [(s, (f[0], f[1], f[2]), r) for s, f, r in soft_matches],
                },
            )

        closest = self.kb.find_soft_matches(claim, top_k=1, threshold=0.0)
        closest_info = closest[0] if closest else None
        return ProbeResult(
            claim=claim,
            verdict=Verdict.FAIL,
            score=closest_info[0] if closest_info else 0.0,
            matched_facts=[closest_info[1]] if closest_info else [],
            reason="No supporting fact found in KB.",
            metadata={"match_type": "none", "closest_match": closest_info if closest_info else None},
        )

    def probe_batch(self, texts: List[str]) -> List[SCPReport]:
        return [self.probe(text) for text in texts]


# =============================================================================
# UTILITIES
# =============================================================================


def pretty_print_report(report: SCPReport) -> None:
    print("=" * 80)
    print("SCP CONSISTENCY REPORT")
    print("=" * 80)
    print(
        f"Text: {report.original_text[:100]}..."
        if len(report.original_text) > 100
        else f"Text: {report.original_text}"
    )
    print(f"Claims extracted: {report.claims_extracted}")
    print(f"Overall score: {report.overall_score:.3f}")
    print(f"Pass rate: {report.pass_rate:.1%}")
    print("-" * 80)

    for i, r in enumerate(report.results, 1):
        print(f"\n[Claim {i}]")
        print(f"  Raw: {r.claim.raw}")
        print(f"  Triple: {r.claim}")
        print(f"  Verdict: {r.verdict.value} (score={r.score:.3f})")
        if r.matched_facts:
            print(f"  Matched: {r.matched_facts[0]}")
        print(f"  Reason: {r.reason}")
    print("\n" + "=" * 80)


def export_proof_to_gexf(subgraph: nx.DiGraph, path: str) -> None:
    nx.write_gexf(subgraph, path)


def export_proof_to_json(subgraph: nx.DiGraph) -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}}
            for n, d in subgraph.nodes(data=True)
        ],
        "edges": [{"source": u, "target": v, **d} for u, v, d in subgraph.edges(data=True)],
    }

