"""
SCP (Symbolic Consistency Probing) - Refactored Prototype
==========================================================
A hypergraph-based hallucination detection system that:
1. Stores knowledge as a hypergraph (relation nodes binding entities)
2. Extracts claims from LLM outputs (rule-based + LLM stub)
3. Probes claims against the KB using embeddings or string similarity
4. Returns proof subgraphs for verified claims

Dependencies:
    pip install networkx numpy sentence-transformers

Optional (for LLM extraction):
    pip install anthropic  # or openai
"""

from __future__ import annotations

import re
import uuid
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

import networkx as nx
from difflib import SequenceMatcher

# Try to import sentence-transformers; fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
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
    """Full report from probing an LLM answer."""
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
# EMBEDDING BACKENDS (Pluggable)
# =============================================================================

class EmbeddingBackend(ABC):
    """Abstract base for embedding providers."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embedding vectors."""
        pass
    
    @abstractmethod
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute similarity between two vectors."""
        pass


class StringSimilarityBackend(EmbeddingBackend):
    """Fallback: uses string similarity (no ML dependencies)."""
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        # Return dummy "embeddings" (we won't actually use them)
        return [[0.0] for _ in texts]
    
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        # This won't be called directly; we override in text_similarity
        return 0.0
    
    def text_similarity(self, a: str, b: str) -> float:
        """Direct text similarity using SequenceMatcher."""
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return SequenceMatcher(None, a_norm, b_norm).ratio()
    
    def _normalize(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r'[""\"\'`,.;:!?()\[\]{}]', "", s)
        return s


class SentenceTransformerBackend(EmbeddingBackend):
    """Production backend using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, List[float]] = {}
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        # Check cache first
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
    
    def text_similarity(self, a: str, b: str) -> float:
        vecs = self.encode([a, b])
        return self.similarity(vecs[0], vecs[1])


# =============================================================================
# CLAIM EXTRACTORS (Pluggable)
# =============================================================================

class ClaimExtractor(ABC):
    """Abstract base for claim extraction strategies."""
    
    @abstractmethod
    def extract(self, text: str) -> List[Claim]:
        """Extract claims from text."""
        pass


class RuleBasedExtractor(ClaimExtractor):
    """Pattern-based claim extraction (fast, no dependencies)."""
    
    PATTERNS: List[Tuple[str, re.Pattern]] = [
        # Specific patterns first (more specific = higher priority)
        ("located_in", re.compile(
            r"^(?P<s>.+?)\s+(?:is|are|was|were)\s+(?:located|situated)\s+in\s+(?P<o>.+?)$", re.I)),
        ("founded_in", re.compile(
            r"^(?P<s>.+?)\s+(?:was|were)\s+founded\s+in\s+(?P<o>.+?)$", re.I)),
        ("born_in", re.compile(
            r"^(?P<s>.+?)\s+(?:was|were)\s+born\s+in\s+(?P<o>.+?)$", re.I)),
        ("died_in", re.compile(
            r"^(?P<s>.+?)\s+died\s+in\s+(?P<o>.+?)$", re.I)),
        ("discovered", re.compile(
            r"^(?P<s>.+?)\s+discovered\s+(?P<o>.+?)$", re.I)),
        ("invented", re.compile(
            r"^(?P<s>.+?)\s+invented\s+(?P<o>.+?)$", re.I)),
        ("proposed", re.compile(
            r"^(?P<s>.+?)\s+proposed\s+(?P<o>.+?)$", re.I)),
        ("developed", re.compile(
            r"^(?P<s>.+?)\s+developed\s+(?P<o>.+?)$", re.I)),
        ("created", re.compile(
            r"^(?P<s>.+?)\s+created\s+(?P<o>.+?)$", re.I)),
        ("wrote", re.compile(
            r"^(?P<s>.+?)\s+wrote\s+(?P<o>.+?)$", re.I)),
        ("published", re.compile(
            r"^(?P<s>.+?)\s+published\s+(?P<o>.+?)$", re.I)),
        ("capital_of", re.compile(
            r"^(?P<s>.+?)\s+is\s+the\s+capital\s+of\s+(?P<o>.+?)$", re.I)),
        ("part_of", re.compile(
            r"^(?P<s>.+?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<o>.+?)$", re.I)),
        ("member_of", re.compile(
            r"^(?P<s>.+?)\s+is\s+(?:a\s+)?member\s+of\s+(?P<o>.+?)$", re.I)),
        ("has", re.compile(
            r"^(?P<s>.+?)\s+has\s+(?P<o>.+?)$", re.I)),
        ("contains", re.compile(
            r"^(?P<s>.+?)\s+contains\s+(?P<o>.+?)$", re.I)),
        # Generic patterns last
        ("is_a", re.compile(
            r"^(?P<s>.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<o>.+?)$", re.I)),
        ("is", re.compile(
            r"^(?P<s>.+?)\s+(?:is|are)\s+(?P<o>.+?)$", re.I)),
        ("was", re.compile(
            r"^(?P<s>.+?)\s+(?:was|were)\s+(?P<o>.+?)$", re.I)),
    ]
    
    def extract(self, text: str) -> List[Claim]:
        claims = []
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
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.\s', r'\1<PERIOD> ', text)
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.replace('<PERIOD>', '.').strip() for p in parts if p.strip()]
    
    def _extract_from_sentence(self, sent: str) -> Optional[Claim]:
        # Clean the sentence
        s = sent.strip()
        s = re.sub(r'[.!?]+$', '', s).strip()
        
        for pred, pattern in self.PATTERNS:
            match = pattern.match(s)
            if match:
                subj = match.group('s').strip()
                obj = match.group('o').strip()
                # Skip if subject or object is too short
                if len(subj) < 2 or len(obj) < 2:
                    continue
                return Claim(
                    subject=subj,
                    predicate=pred,
                    obj=obj,
                    raw=sent,
                    confidence=0.7  # Rule-based = moderate confidence
                )
        return None


class LLMExtractor(ClaimExtractor):
    """LLM-based claim extraction (higher quality, requires API)."""
    
    SYSTEM_PROMPT = """You are a claim extractor. Given text, extract factual claims as subject-predicate-object triples.

Output JSON array of objects with keys: subject, predicate, object, confidence (0-1).

Rules:
- Extract only factual claims, not opinions
- Use simple, normalized predicates (e.g., "located_in", "discovered", "is_a")
- confidence = how certain the claim is stated (not whether it's true)
- If no claims found, return []

Example:
Input: "The Eiffel Tower is located in Paris. It was built in 1889."
Output: [
  {"subject": "The Eiffel Tower", "predicate": "located_in", "object": "Paris", "confidence": 0.95},
  {"subject": "The Eiffel Tower", "predicate": "built_in", "object": "1889", "confidence": 0.95}
]"""
    
    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            llm_fn: Function that takes a prompt and returns LLM response.
                   If None, falls back to rule-based extraction.
        """
        self.llm_fn = llm_fn
        self._fallback = RuleBasedExtractor()
    
    def extract(self, text: str) -> List[Claim]:
        if not self.llm_fn:
            return self._fallback.extract(text)
        
        prompt = f"{self.SYSTEM_PROMPT}\n\nInput: {text}\nOutput:"
        
        try:
            response = self.llm_fn(prompt)
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return self._fallback.extract(text)
            
            data = json.loads(json_match.group())
            claims = []
            for item in data:
                claims.append(Claim(
                    subject=item.get('subject', ''),
                    predicate=item.get('predicate', 'unknown'),
                    obj=item.get('object', ''),
                    raw=text,
                    confidence=float(item.get('confidence', 0.8))
                ))
            return claims
        except Exception as e:
            # Fall back to rule-based on any error
            print(f"LLM extraction failed: {e}, falling back to rules")
            return self._fallback.extract(text)


class HybridExtractor(ClaimExtractor):
    """Combines rule-based and LLM extraction for best coverage."""
    
    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = LLMExtractor(llm_fn)
    
    def extract(self, text: str) -> List[Claim]:
        # Get rule-based claims
        rule_claims = self.rule_extractor.extract(text)
        
        # If LLM available, also get LLM claims
        if self.llm_extractor.llm_fn:
            llm_claims = self.llm_extractor.extract(text)
            # Merge, preferring LLM claims (higher quality)
            return self._merge_claims(rule_claims, llm_claims)
        
        return rule_claims
    
    def _merge_claims(self, rule_claims: List[Claim], llm_claims: List[Claim]) -> List[Claim]:
        """Merge claims, deduplicating by approximate match."""
        merged = list(llm_claims)
        seen_sigs = {self._signature(c) for c in llm_claims}
        
        for rc in rule_claims:
            sig = self._signature(rc)
            if sig not in seen_sigs:
                merged.append(rc)
                seen_sigs.add(sig)
        
        return merged
    
    def _signature(self, claim: Claim) -> str:
        """Create a normalized signature for deduplication."""
        def norm(s):
            return re.sub(r'\s+', ' ', s.lower().strip())
        return f"{norm(claim.subject)}|{norm(claim.predicate)}|{norm(claim.obj)}"


# =============================================================================
# HYPERGRAPH KNOWLEDGE BASE
# =============================================================================

class HyperKB:
    """
    Hypergraph-style Knowledge Base using NetworkX.
    
    Structure:
        - Entity nodes: "ent:<normalized_name>"
        - Relation nodes: "rel:<uuid>" with predicate attribute
        - Edges: subject --subj--> relation --obj--> object
    
    This "edge-as-node" pattern allows n-ary relations and rich metadata.
    """
    
    def __init__(self, embedding_backend: Optional[EmbeddingBackend] = None):
        self.g = nx.DiGraph()
        self.backend = embedding_backend or StringSimilarityBackend()
        self._fact_cache: Optional[List[Tuple[str, str, str, str]]] = None  # (s, p, o, rel_id)
    
    def _norm(self, s: str) -> str:
        """Normalize string for matching."""
        s = s.strip().lower()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[""\"\'`,.;:!?()\[\]{}]', '', s)
        return s
    
    def add_entity(self, name: str, **attrs) -> str:
        """Add or get an entity node."""
        eid = f"ent:{self._norm(name)}"
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
        **metadata
    ) -> str:
        """
        Add a fact to the KB.
        
        Returns:
            The relation node ID.
        """
        s_id = self.add_entity(subject)
        o_id = self.add_entity(obj)
        
        r_id = f"rel:{uuid.uuid4().hex[:12]}"
        self.g.add_node(
            r_id,
            kind="relation",
            predicate=self._norm(predicate),
            predicate_label=predicate,
            confidence=float(confidence),
            source=source,
            **metadata
        )
        self.g.add_edge(s_id, r_id, role="subj")
        self.g.add_edge(r_id, o_id, role="obj")
        
        self._fact_cache = None
        return r_id
    
    def add_facts_bulk(self, facts: List[Tuple[str, str, str]], **kwargs) -> List[str]:
        """Add multiple facts efficiently."""
        return [self.add_fact(s, p, o, **kwargs) for s, p, o in facts]
    
    def iter_facts(self) -> List[Tuple[str, str, str, str]]:
        """Iterate all facts as (subject, predicate, object, relation_id) tuples."""
        if self._fact_cache is not None:
            return self._fact_cache
        
        facts = []
        for r_id, attrs in self.g.nodes(data=True):
            if attrs.get("kind") != "relation":
                continue
            
            pred = attrs.get("predicate_label", attrs.get("predicate", ""))
            
            subjects = [
                u for u, _ in self.g.in_edges(r_id)
                if self.g.edges[u, r_id].get("role") == "subj"
            ]
            objects = [
                v for _, v in self.g.out_edges(r_id)
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
        """Check for exact match. Returns relation_id if found."""
        c_s, c_p, c_o = self._norm(claim.subject), self._norm(claim.predicate), self._norm(claim.obj)
        
        for s, p, o, r_id in self.iter_facts():
            if self._norm(s) == c_s and self._norm(p) == c_p and self._norm(o) == c_o:
                return r_id
        return None
    
    def find_soft_matches(
        self,
        claim: Claim,
        top_k: int = 3,
        threshold: float = 0.0,
        min_subject_sim: float = 0.5
    ) -> List[Tuple[float, Tuple[str, str, str], str, Dict[str, float]]]:
        """
        Find soft matches using embeddings or string similarity.
        
        Args:
            claim: The claim to match
            top_k: Number of top matches to return
            threshold: Minimum overall score threshold
            min_subject_sim: Minimum subject similarity required for soft match
                            (prevents false attribution hallucinations)
        
        Returns:
            List of (score, (s, p, o), relation_id, component_scores) sorted by score descending.
        """
        matches = []
        
        for s, p, o, r_id in self.iter_facts():
            # Compute component similarities
            if isinstance(self.backend, StringSimilarityBackend):
                s_sim = self.backend.text_similarity(claim.subject, s)
                p_sim = self.backend.text_similarity(claim.predicate, p)
                o_sim = self.backend.text_similarity(claim.obj, o)
            else:
                s_sim = self.backend.text_similarity(claim.subject, s)
                p_sim = self.backend.text_similarity(claim.predicate, p)
                o_sim = self.backend.text_similarity(claim.obj, o)
            
            component_scores = {"subject": s_sim, "predicate": p_sim, "object": o_sim}
            
            # Weighted combination (predicate matters most)
            score = (0.30 * s_sim) + (0.45 * p_sim) + (0.25 * o_sim)
            
            if score >= threshold:
                matches.append((score, (s, p, o), r_id, component_scores))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]
    
    def find_false_attributions(
        self,
        claim: Claim,
        min_predicate_obj_sim: float = 0.7,
        max_subject_sim: float = 0.6
    ) -> List[Tuple[str, str, str, str, Dict[str, float]]]:
        """
        Find facts where predicate+object match but subject doesn't.
        
        This catches "false attribution" hallucinations like:
        - "Edison invented the telephone" (should be Bell)
        - "Marie Curie discovered gravity" (should be Newton)
        
        Returns:
            List of (s, p, o, relation_id, component_scores) for false attributions.
        """
        false_attributions = []
        
        for s, p, o, r_id in self.iter_facts():
            if isinstance(self.backend, StringSimilarityBackend):
                s_sim = self.backend.text_similarity(claim.subject, s)
                p_sim = self.backend.text_similarity(claim.predicate, p)
                o_sim = self.backend.text_similarity(claim.obj, o)
            else:
                s_sim = self.backend.text_similarity(claim.subject, s)
                p_sim = self.backend.text_similarity(claim.predicate, p)
                o_sim = self.backend.text_similarity(claim.obj, o)
            
            # High predicate+object similarity but low subject similarity = false attribution
            pred_obj_sim = (0.6 * p_sim) + (0.4 * o_sim)
            
            if pred_obj_sim >= min_predicate_obj_sim and s_sim < max_subject_sim:
                component_scores = {"subject": s_sim, "predicate": p_sim, "object": o_sim}
                false_attributions.append((s, p, o, r_id, component_scores))
        
        return false_attributions
    
    def find_contradictions(self, claim: Claim) -> List[Tuple[str, str, str, str]]:
        """
        Find potential contradictions.
        
        Looks for facts with same subject and predicate but different object.
        """
        contradictions = []
        c_s, c_p = self._norm(claim.subject), self._norm(claim.predicate)
        c_o = self._norm(claim.obj)
        
        for s, p, o, r_id in self.iter_facts():
            if self._norm(s) == c_s and self._norm(p) == c_p and self._norm(o) != c_o:
                contradictions.append((s, p, o, r_id))
        
        return contradictions
    
    def get_proof_subgraph(self, relation_ids: List[str]) -> nx.DiGraph:
        """Extract minimal subgraph that proves the given relations."""
        nodes_to_include = set()
        
        for r_id in relation_ids:
            if not self.g.has_node(r_id):
                continue
            nodes_to_include.add(r_id)
            
            # Add subject entities
            for u, _ in self.g.in_edges(r_id):
                nodes_to_include.add(u)
            
            # Add object entities
            for _, v in self.g.out_edges(r_id):
                nodes_to_include.add(v)
        
        return self.g.subgraph(nodes_to_include).copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export KB to dictionary format."""
        return {
            "entities": [
                {"id": n, **d}
                for n, d in self.g.nodes(data=True)
                if d.get("kind") == "entity"
            ],
            "relations": [
                {"id": n, **d}
                for n, d in self.g.nodes(data=True)
                if d.get("kind") == "relation"
            ],
            "facts": [
                {"subject": s, "predicate": p, "object": o, "relation_id": r}
                for s, p, o, r in self.iter_facts()
            ]
        }
    
    def stats(self) -> Dict[str, int]:
        """Get KB statistics."""
        entities = sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == "entity")
        relations = sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == "relation")
        return {
            "entities": entities,
            "relations": relations,
            "edges": self.g.number_of_edges(),
        }


# =============================================================================
# SCP PROBER (Main Engine)
# =============================================================================

class SCPProber:
    """
    Symbolic Consistency Prober.
    
    Extracts claims from text and probes them against a knowledge base,
    returning detailed results including proof subgraphs.
    """
    
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
        """
        Probe text for consistency with KB.
        
        Args:
            text: LLM output or any text to verify.
        
        Returns:
            SCPReport with detailed results.
        """
        claims = self.extractor.extract(text)
        results = []
        
        if not claims:
            results.append(ProbeResult(
                claim=Claim("", "unknown", "", raw=text),
                verdict=Verdict.UNKNOWN,
                score=0.0,
                reason="No claims could be extracted from text.",
            ))
        else:
            for claim in claims:
                result = self._probe_claim(claim)
                results.append(result)
        
        # Calculate aggregate scores
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
        """Probe a single claim against the KB."""
        
        # 1. Check for exact match
        exact_match = self.kb.has_fact_exact(claim)
        if exact_match:
            proof = self.kb.get_proof_subgraph([exact_match])
            return ProbeResult(
                claim=claim,
                verdict=Verdict.PASS,
                score=1.0,
                matched_facts=[claim.to_tuple()],
                proof_subgraph=proof,
                reason="Exact match found in KB.",
                metadata={"match_type": "exact", "relation_id": exact_match}
            )
        
        # 2. Check for contradictions (if enabled)
        if self.contradiction_check:
            contradictions = self.kb.find_contradictions(claim)
            if contradictions:
                return ProbeResult(
                    claim=claim,
                    verdict=Verdict.CONTRADICT,
                    score=0.0,
                    matched_facts=[(s, p, o) for s, p, o, _ in contradictions],
                    reason=f"Contradicts {len(contradictions)} fact(s) in KB.",
                    metadata={"contradictions": contradictions}
                )
        
        # 3. Check for false attributions (BEFORE soft match!)
        # This catches hallucinations like "Edison invented the telephone" when Bell did
        false_attributions = self.kb.find_false_attributions(claim)
        if false_attributions:
            # Found a fact with matching predicate+object but wrong subject
            s, p, o, r_id, comp_scores = false_attributions[0]
            return ProbeResult(
                claim=claim,
                verdict=Verdict.FAIL,
                score=0.0,
                matched_facts=[(s, p, o)],
                reason=f"False attribution: KB has ({s}) --[{p}]--> ({o}), not ({claim.subject}).",
                metadata={
                    "match_type": "false_attribution",
                    "correct_subject": s,
                    "claimed_subject": claim.subject,
                    "component_scores": comp_scores,
                    "relation_id": r_id
                }
            )
        
        # 4. Check for soft matches
        soft_matches = self.kb.find_soft_matches(claim, top_k=3, threshold=self.soft_threshold)
        
        if soft_matches:
            best_score, best_fact, best_rel_id, comp_scores = soft_matches[0]
            
            # Additional check: require minimum subject similarity for soft pass
            # This prevents false attributions from slipping through
            if comp_scores["subject"] < 0.5:
                return ProbeResult(
                    claim=claim,
                    verdict=Verdict.FAIL,
                    score=best_score,
                    matched_facts=[best_fact],
                    reason=f"Subject mismatch: claim subject '{claim.subject}' doesn't match '{best_fact[0]}' (sim={comp_scores['subject']:.3f}).",
                    metadata={
                        "match_type": "subject_mismatch",
                        "component_scores": comp_scores,
                        "relation_id": best_rel_id
                    }
                )
            
            proof = self.kb.get_proof_subgraph([best_rel_id])
            
            return ProbeResult(
                claim=claim,
                verdict=Verdict.SOFT_PASS,
                score=best_score,
                matched_facts=[best_fact],
                proof_subgraph=proof,
                reason=f"Soft match (score={best_score:.3f}) above threshold={self.soft_threshold}.",
                metadata={
                    "match_type": "soft",
                    "relation_id": best_rel_id,
                    "component_scores": comp_scores,
                    "all_matches": [(s, (f[0], f[1], f[2]), r) for s, f, r, _ in soft_matches]
                }
            )
        
        # 5. No match found
        # Still get closest match for context
        closest = self.kb.find_soft_matches(claim, top_k=1, threshold=0.0)
        closest_info = closest[0] if closest else None
        
        return ProbeResult(
            claim=claim,
            verdict=Verdict.FAIL,
            score=closest_info[0] if closest_info else 0.0,
            matched_facts=[closest_info[1]] if closest_info else [],
            reason="No supporting fact found in KB.",
            metadata={
                "match_type": "none",
                "closest_match": (closest_info[0], closest_info[1], closest_info[2]) if closest_info else None
            }
        )
    
    def probe_batch(self, texts: List[str]) -> List[SCPReport]:
        """Probe multiple texts."""
        return [self.probe(text) for text in texts]


# =============================================================================
# UTILITIES
# =============================================================================

def pretty_print_report(report: SCPReport) -> None:
    """Print a formatted report to console."""
    print("=" * 80)
    print("SCP CONSISTENCY REPORT")
    print("=" * 80)
    print(f"Text: {report.original_text[:100]}..." if len(report.original_text) > 100 else f"Text: {report.original_text}")
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
    """Export proof subgraph to GEXF format (for Gephi visualization)."""
    nx.write_gexf(subgraph, path)


def export_proof_to_json(subgraph: nx.DiGraph) -> Dict[str, Any]:
    """Export proof subgraph to JSON-serializable dict."""
    return {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}}
            for n, d in subgraph.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, **d}
            for u, v, d in subgraph.edges(data=True)
        ]
    }


# =============================================================================
# DEMO / MAIN
# =============================================================================

def demo():
    """Run a demonstration of the SCP system."""
    
    print("=" * 80)
    print("SCP (Symbolic Consistency Probing) - Hallucination Detection Demo")
    print("=" * 80)
    print()
    print("Initializing SCP system...")
    
    # Choose embedding backend
    if EMBEDDINGS_AVAILABLE:
        print("✓ Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
        backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
    else:
        print("⚠ Using string similarity (install sentence-transformers for better results)")
        backend = StringSimilarityBackend()
    
    # Create KB
    kb = HyperKB(embedding_backend=backend)
    
    # Seed with facts
    print("Loading knowledge base...")
    facts = [
        ("The Eiffel Tower", "located_in", "Paris"),
        ("Paris", "is_capital_of", "France"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Albert Einstein", "born_in", "Germany"),
        ("Charles Darwin", "proposed", "the theory of evolution"),
        ("Marie Curie", "discovered", "radium"),
        ("Marie Curie", "born_in", "Poland"),
        ("Isaac Newton", "discovered", "gravity"),
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("Python", "created_by", "Guido van Rossum"),
        ("The Great Wall", "located_in", "China"),
        ("Tokyo", "is_capital_of", "Japan"),
    ]
    kb.add_facts_bulk(facts, source="seed_data", confidence=0.95)
    print(f"✓ KB loaded: {kb.stats()}")
    
    # Create prober
    prober = SCPProber(
        kb=kb,
        extractor=RuleBasedExtractor(),
        soft_threshold=0.70
    )
    
    # Test cases with descriptions
    test_cases = [
        # (text, expected_outcome, description)
        ("The Eiffel Tower is located in Paris.",
         "PASS", "Exact match - should pass"),
        
        ("Einstein discovered relativity.",
         "SOFT_PASS", "Semantic match - similar wording"),
        
        ("The Eiffel Tower is located in London.",
         "CONTRADICT", "Direct contradiction - Eiffel Tower is in Paris"),
        
        ("Albert Einstein was born in France.",
         "CONTRADICT", "Wrong country - Einstein was born in Germany"),
        
        ("Thomas Edison invented the telephone.",
         "FAIL", "FALSE ATTRIBUTION - Bell invented the telephone, not Edison"),
        
        ("Marie Curie discovered radium. She invented the telephone.",
         "MIXED", "One correct claim + one hallucination"),
    ]
    
    print("\n" + "=" * 80)
    print("HALLUCINATION DETECTION TESTS")
    print("=" * 80)
    
    for text, expected, description in test_cases:
        print(f"\n>>> Testing: {description}")
        print(f"    Input: \"{text}\"")
        print(f"    Expected: {expected}")
        report = prober.probe(text)
        pretty_print_report(report)
    
    # Demo: False attribution detection
    print("\n" + "=" * 80)
    print("FALSE ATTRIBUTION DETECTION (Key Feature)")
    print("=" * 80)
    print("""
The algorithm detects 'false attribution' hallucinations where:
- The predicate and object match a known fact
- But the subject is WRONG

Example: "Edison invented the telephone" is FALSE because
- KB knows: Bell invented the telephone
- Predicate+Object match (invented + telephone)
- But subject (Edison ≠ Bell) doesn't match → FAIL
""")
    
    # Show JSON export
    print("\n" + "=" * 80)
    print("JSON EXPORT EXAMPLE")
    print("=" * 80)
    report = prober.probe("Einstein discovered relativity.")
    print(report.to_json())


if __name__ == "__main__":
    demo()