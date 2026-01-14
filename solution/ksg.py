#!/usr/bin/env python3
"""
KnowShowGo Integration for Hallucination Detection
===================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)

Merged from: ksg_ground_truth.py + ksg_integration.py

This module provides KnowShowGo integration for:
1. Ground truth storage (verified facts)
2. Semantic memory backend
3. Provenance tracking
4. RAG retrieval context

KnowShowGo Design Principles:
- PROPOSITIONS = Claims as RDF triples (subject → predicate → object)
- PROTOTYPES = Claim categories/schemas (VerifiedProposition, RefutedProposition)
- EXEMPLARS = Verified instances that define categories
- SUB-CLAIMS = Decomposed propositions for granular verification
- FUZZY DAG = Knowledge graph with weighted edges + provenance

Usage:
    # Standalone ground truth
    gt = KSGGroundTruth()
    gt.add_verified_fact("Bell", "invented", "telephone")
    result = gt.check("Edison invented the telephone")
    
    # With REST API client
    ksg = KSGClient("http://localhost:3000")
    ksg.create_concept("Claim", {"subject": "Bell", ...})
    
    # As verified memory backend
    memory = KSGVerifiedMemory("http://localhost:3000")
    memory.store_verified_claim(...)

Requirements:
    - KnowShowGo server running (github.com/lehelkovach/knowshowgo)
    - Or use mock mode for testing
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


# =============================================================================
# ENUMS AND STATUS TYPES
# =============================================================================

class PropositionStatus(Enum):
    """Status of a proposition in KnowShowGo."""
    VERIFIED = "verified"           # Ground truth - confirmed true
    REFUTED = "refuted"             # Ground truth - confirmed false
    UNVERIFIED = "unverified"       # Pending verification
    CONTESTED = "contested"         # Multiple sources disagree


# =============================================================================
# KSG PROTOTYPES (SCHEMA DEFINITIONS)
# =============================================================================

KSG_PROTOTYPES = {
    # Core proposition prototype (RDF-style triple)
    "Proposition": {
        "name": "Proposition",
        "description": "An RDF-style triple: subject-predicate-object claim",
        "schema": {
            "subject_uuid": "string",
            "predicate_uuid": "string",
            "object_uuid": "string",
            "raw_text": "string",
            "confidence": "number",
            "status": "enum:verified,refuted,unverified,contested"
        }
    },
    
    # Verified proposition - ground truth
    "VerifiedProposition": {
        "name": "VerifiedProposition",
        "description": "A proposition confirmed as ground truth",
        "schema": {
            "verification_method": "string",
            "verification_date": "string",
            "verifier": "string",
            "source_urls": "array"
        },
        "parent": "Proposition"
    },
    
    # Refuted proposition - known false
    "RefutedProposition": {
        "name": "RefutedProposition",
        "description": "A proposition confirmed as false",
        "schema": {
            "correct_proposition_uuid": "string",
            "refutation_reason": "string",
            "refutation_date": "string"
        },
        "parent": "Proposition"
    },
    
    # Claim (for verification pipeline)
    "Claim": {
        "name": "Claim",
        "description": "A factual claim extracted from text",
        "schema": {
            "subject": "string",
            "predicate": "string",
            "object": "string",
            "raw_text": "string",
            "confidence": "number"
        }
    },
    
    # Entity prototype
    "Entity": {
        "name": "Entity",
        "description": "A named entity (person, place, thing, concept)",
        "schema": {
            "name": "string",
            "entity_type": "enum:person,place,organization,thing,concept,event",
            "aliases": "array",
            "description": "string"
        }
    },
    
    # Predicate prototype (verbs/relations)
    "Predicate": {
        "name": "Predicate",
        "description": "A relation type (verb) between entities",
        "schema": {
            "name": "string",
            "inverse_name": "string",
            "domain_types": "array",
            "range_types": "array"
        }
    },
    
    # Source prototype (provenance)
    "Source": {
        "name": "Source",
        "description": "A source of information for provenance tracking",
        "schema": {
            "source_type": "enum:document,website,database,llm,human,api",
            "url": "string",
            "title": "string",
            "trust_score": "number",
            "last_verified": "string"
        }
    },
    
    # Verification result
    "VerificationResult": {
        "name": "VerificationResult",
        "description": "Result of verifying a claim",
        "schema": {
            "status": "enum:verified,refuted,unverifiable",
            "confidence": "number",
            "method": "string",
            "timestamp": "number"
        }
    }
}

# Association types for provenance graph
KSG_ASSOCIATION_TYPES = {
    "supports": {
        "description": "Evidence supports this proposition",
        "inverse": "supported_by"
    },
    "contradicts": {
        "description": "Evidence contradicts this proposition",
        "inverse": "contradicted_by"
    },
    "derived_from": {
        "description": "Proposition derived from this source",
        "inverse": "source_of"
    },
    "verified_by": {
        "description": "Claim verified by method/source",
        "inverse": "verifies"
    },
    "sub_claim_of": {
        "description": "This is a sub-claim of a larger proposition",
        "inverse": "has_sub_claim"
    },
    "same_as": {
        "description": "These propositions are equivalent",
        "inverse": "same_as"
    },
    "exemplar_of": {
        "description": "This is an exemplar (prototype instance) of a category",
        "inverse": "has_exemplar"
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SubClaim:
    """A decomposed sub-claim from a larger proposition."""
    subject: str
    predicate: str
    obj: str
    status: PropositionStatus
    confidence: float
    ksg_uuid: Optional[str] = None


@dataclass
class GroundTruthResult:
    """Result of checking a claim against KnowShowGo ground truth."""
    original_claim: str
    status: PropositionStatus
    confidence: float
    
    matching_proposition: Optional[Dict] = None
    contradicting_proposition: Optional[Dict] = None
    sub_claims: List[SubClaim] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "original_claim": self.original_claim,
            "status": self.status.value,
            "confidence": self.confidence,
            "matching_proposition": self.matching_proposition,
            "contradicting_proposition": self.contradicting_proposition,
            "sub_claims": [
                {
                    "subject": sc.subject,
                    "predicate": sc.predicate,
                    "object": sc.obj,
                    "status": sc.status.value,
                    "confidence": sc.confidence
                }
                for sc in self.sub_claims
            ],
            "sources": self.sources,
            "reason": self.reason
        }


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================

def default_embed(text: str, dim: int = 64) -> List[float]:
    """Default embedding using hash (for testing without ML dependencies)."""
    h = hashlib.sha256(text.lower().encode()).digest()
    return [float(b) / 255.0 for b in h[:dim]]


def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Cosine similarity between embeddings."""
    dot = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = sum(a * a for a in emb1) ** 0.5
    norm2 = sum(b * b for b in emb2) ** 0.5
    return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


# =============================================================================
# KSG GROUND TRUTH (LOCAL/MOCK MODE)
# =============================================================================

class KSGGroundTruth:
    """
    KnowShowGo-backed ground truth for hallucination detection.
    
    This is a local implementation that works without the KSG server.
    Use for testing or when server is unavailable.
    
    Features:
    1. Proposition storage (claims as RDF triples)
    2. Fuzzy matching via embeddings
    3. Exemplar-based category matching
    4. Sub-claim decomposition
    5. Provenance tracking via associations
    
    Usage:
        gt = KSGGroundTruth()
        gt.add_verified_fact("Bell", "invented", "telephone")
        result = gt.check("Edison invented the telephone")
        # result.status == PropositionStatus.REFUTED
    """
    
    def __init__(self, embed_fn: Callable = None):
        self.embed_fn = embed_fn or default_embed
        self._propositions: Dict[str, Dict] = {}
        self._entities: Dict[str, Dict] = {}
        self._predicates: Dict[str, Dict] = {}
        self._associations: List[Dict] = []
        self._embeddings: Dict[str, List[float]] = {}
    
    def add_entity(self, name: str, entity_type: str = "thing", aliases: List[str] = None) -> str:
        """Add an entity to the ground truth."""
        uuid = f"entity_{len(self._entities)}"
        self._entities[uuid] = {
            "uuid": uuid,
            "name": name,
            "entity_type": entity_type,
            "aliases": aliases or []
        }
        self._embeddings[uuid] = self.embed_fn(name)
        return uuid
    
    def add_predicate(self, name: str, inverse_name: str = None) -> str:
        """Add a predicate (verb/relation) to the ground truth."""
        uuid = f"predicate_{len(self._predicates)}"
        self._predicates[uuid] = {
            "uuid": uuid,
            "name": name,
            "inverse_name": inverse_name or f"{name}_by"
        }
        self._embeddings[uuid] = self.embed_fn(name)
        return uuid
    
    def add_verified_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        sources: List[Dict] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add a verified fact (ground truth).
        
        Creates entities and predicate if they don't exist,
        then creates a VerifiedProposition linking them.
        """
        subject_uuid = self._find_or_create_entity(subject)
        predicate_uuid = self._find_or_create_predicate(predicate)
        object_uuid = self._find_or_create_entity(obj)
        
        prop_uuid = f"prop_{len(self._propositions)}"
        raw_text = f"{subject} {predicate} {obj}"
        
        self._propositions[prop_uuid] = {
            "uuid": prop_uuid,
            "prototype": "VerifiedProposition",
            "subject_uuid": subject_uuid,
            "predicate_uuid": predicate_uuid,
            "object_uuid": object_uuid,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "raw_text": raw_text,
            "status": PropositionStatus.VERIFIED.value,
            "confidence": confidence,
            "sources": sources or []
        }
        self._embeddings[prop_uuid] = self.embed_fn(raw_text)
        
        for source in (sources or []):
            self._associations.append({
                "from": prop_uuid,
                "to": source.get("uuid", "external"),
                "type": "derived_from",
                "weight": source.get("trust_score", 0.8)
            })
        
        return prop_uuid
    
    def _find_or_create_entity(self, name: str) -> str:
        """Find entity by name or create if not exists."""
        query_emb = self.embed_fn(name)
        
        for uuid, entity in self._entities.items():
            if entity["name"].lower() == name.lower():
                return uuid
            if name.lower() in [a.lower() for a in entity.get("aliases", [])]:
                return uuid
            
            score = cosine_similarity(query_emb, self._embeddings.get(uuid, []))
            if score > 0.9:
                return uuid
        
        return self.add_entity(name, "thing")
    
    def _find_or_create_predicate(self, name: str) -> str:
        """Find predicate by name or create if not exists."""
        for uuid, pred in self._predicates.items():
            if pred["name"].lower() == name.lower():
                return uuid
        return self.add_predicate(name)
    
    def check(self, claim_text: str, decompose: bool = True) -> GroundTruthResult:
        """
        Check a claim against ground truth.
        
        Returns GroundTruthResult with status, confidence, and provenance.
        """
        parts = self._extract_triple(claim_text)
        if not parts:
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.UNVERIFIED,
                confidence=0.0,
                reason="Could not extract subject-predicate-object from claim"
            )
        
        subject, predicate, obj = parts
        claim_emb = self.embed_fn(claim_text)
        
        matches = []
        contradictions = []
        
        for prop_uuid, prop in self._propositions.items():
            prop_emb = self._embeddings.get(prop_uuid, [])
            similarity = cosine_similarity(claim_emb, prop_emb)
            
            if similarity > 0.7:
                if self._is_match(subject, predicate, obj, prop):
                    matches.append((similarity, prop))
                elif self._is_contradiction(subject, predicate, obj, prop):
                    contradictions.append((similarity, prop))
        
        if contradictions:
            best = max(contradictions, key=lambda x: x[0])
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.REFUTED,
                confidence=best[0],
                contradicting_proposition=best[1],
                sources=best[1].get("sources", []),
                reason=f"Ground truth shows {best[1]['subject']} {best[1]['predicate']} {best[1]['object']}, not {subject}",
                sub_claims=self._decompose(subject, predicate, obj) if decompose else []
            )
        
        if matches:
            best = max(matches, key=lambda x: x[0])
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.VERIFIED,
                confidence=best[0],
                matching_proposition=best[1],
                sources=best[1].get("sources", []),
                reason=f"Ground truth confirms: {best[1]['raw_text']}",
                sub_claims=self._decompose(subject, predicate, obj) if decompose else []
            )
        
        return GroundTruthResult(
            original_claim=claim_text,
            status=PropositionStatus.UNVERIFIED,
            confidence=0.5,
            reason="No matching ground truth found",
            sub_claims=self._decompose(subject, predicate, obj) if decompose else []
        )
    
    def _extract_triple(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Extract subject-predicate-object from text."""
        import re
        patterns = [
            r"^(.+?)\s+(invented|discovered|created|wrote|founded|born in|located in|is)\s+(.+?)\.?$",
            r"^(.+?)\s+(is|was|are|were)\s+(.+?)\.?$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        return None
    
    def _is_match(self, subject: str, predicate: str, obj: str, prop: Dict) -> bool:
        """Check if claim matches proposition."""
        subj_sim = cosine_similarity(self.embed_fn(subject), self.embed_fn(prop["subject"]))
        pred_sim = cosine_similarity(self.embed_fn(predicate), self.embed_fn(prop["predicate"]))
        obj_sim = cosine_similarity(self.embed_fn(obj), self.embed_fn(prop["object"]))
        return subj_sim > 0.8 and pred_sim > 0.7 and obj_sim > 0.8
    
    def _is_contradiction(self, subject: str, predicate: str, obj: str, prop: Dict) -> bool:
        """Check if claim contradicts proposition."""
        pred_sim = cosine_similarity(self.embed_fn(predicate), self.embed_fn(prop["predicate"]))
        obj_sim = cosine_similarity(self.embed_fn(obj), self.embed_fn(prop["object"]))
        subj_sim = cosine_similarity(self.embed_fn(subject), self.embed_fn(prop["subject"]))
        return pred_sim > 0.7 and obj_sim > 0.8 and subj_sim < 0.5
    
    def _decompose(self, subject: str, predicate: str, obj: str) -> List[SubClaim]:
        """Decompose claim into verifiable sub-claims."""
        sub_claims = []
        
        subject_known = any(
            e["name"].lower() == subject.lower() or 
            subject.lower() in [a.lower() for a in e.get("aliases", [])]
            for e in self._entities.values()
        )
        sub_claims.append(SubClaim(
            subject=subject, predicate="is_a", obj="Entity",
            status=PropositionStatus.VERIFIED if subject_known else PropositionStatus.UNVERIFIED,
            confidence=1.0 if subject_known else 0.5
        ))
        
        predicate_known = any(p["name"].lower() == predicate.lower() for p in self._predicates.values())
        sub_claims.append(SubClaim(
            subject=predicate, predicate="is_a", obj="Predicate",
            status=PropositionStatus.VERIFIED if predicate_known else PropositionStatus.UNVERIFIED,
            confidence=1.0 if predicate_known else 0.5
        ))
        
        object_known = any(
            e["name"].lower() == obj.lower() or 
            obj.lower() in [a.lower() for a in e.get("aliases", [])]
            for e in self._entities.values()
        )
        sub_claims.append(SubClaim(
            subject=obj, predicate="is_a", obj="Entity",
            status=PropositionStatus.VERIFIED if object_known else PropositionStatus.UNVERIFIED,
            confidence=1.0 if object_known else 0.5
        ))
        
        return sub_claims
    
    def stats(self) -> Dict:
        """Get ground truth statistics."""
        return {
            "total_propositions": len(self._propositions),
            "total_entities": len(self._entities),
            "total_predicates": len(self._predicates),
            "total_associations": len(self._associations),
            "verified": sum(1 for p in self._propositions.values() 
                          if p["status"] == PropositionStatus.VERIFIED.value)
        }


# =============================================================================
# KSG REST API CLIENT
# =============================================================================

class KSGClient:
    """
    Python client for KnowShowGo REST API.
    
    Requires KnowShowGo server running at ksg_url.
    See: github.com/lehelkovach/knowshowgo
    
    Usage:
        ksg = KSGClient("http://localhost:3000")
        claim_id = ksg.create_concept("Claim", {"subject": "Einstein", ...})
        results = ksg.search("Einstein discovered", top_k=5)
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", embed_fn: Callable = None):
        self.base_url = base_url.rstrip("/")
        self.embed_fn = embed_fn or default_embed
        self._prototype_cache = {}
    
    def _request_sync(self, method: str, path: str, data: dict = None) -> dict:
        """Synchronous HTTP request."""
        import requests
        url = f"{self.base_url}{path}"
        
        try:
            if method == "GET":
                resp = requests.get(url, timeout=10)
            elif method == "POST":
                resp = requests.post(url, json=data, timeout=10)
            elif method == "PUT":
                resp = requests.put(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unknown method: {method}")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def ensure_prototypes(self):
        """Create standard prototypes if they don't exist."""
        for name, proto in KSG_PROTOTYPES.items():
            try:
                embedding = self.embed_fn(f"{proto['name']} {proto['description']}")
                result = self._request_sync("POST", "/prototypes", {
                    **proto,
                    "embedding": embedding
                })
                self._prototype_cache[name] = result.get("uuid")
            except Exception as e:
                pass  # Prototype may already exist
    
    def create_concept(self, prototype_name: str, props: dict, text_for_embedding: str = None) -> str:
        """Create a concept from a prototype."""
        proto_uuid = self._prototype_cache.get(prototype_name)
        
        if text_for_embedding is None:
            text_for_embedding = " ".join(str(v) for v in props.values())
        
        embedding = self.embed_fn(text_for_embedding)
        
        result = self._request_sync("POST", "/concepts", {
            "prototypeUuid": proto_uuid,
            "jsonObj": props,
            "embedding": embedding
        })
        
        return result.get("uuid", "")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[dict]:
        """Search for concepts by semantic similarity."""
        embedding = self.embed_fn(query)
        
        result = self._request_sync("POST", "/concepts/search", {
            "embedding": embedding,
            "topK": top_k,
            "similarityThreshold": threshold
        })
        
        return result.get("results", [])
    
    def create_association(
        self, 
        from_uuid: str, 
        to_uuid: str, 
        association_type: str,
        weight: float = 1.0,
        metadata: dict = None
    ) -> str:
        """Create an association between concepts."""
        result = self._request_sync("POST", "/associations", {
            "fromUuid": from_uuid,
            "toUuid": to_uuid,
            "type": association_type,
            "weight": weight,
            "metadata": metadata or {}
        })
        return result.get("uuid", "")
    
    def get_associations(self, concept_uuid: str, association_type: str = None) -> List[dict]:
        """Get associations for a concept."""
        path = f"/concepts/{concept_uuid}/associations"
        if association_type:
            path += f"?type={association_type}"
        return self._request_sync("GET", path)
    
    def get_history(self, concept_uuid: str) -> List[dict]:
        """Get version history for a concept (audit trail)."""
        return self._request_sync("GET", f"/concepts/{concept_uuid}/history")
    
    def health_check(self) -> bool:
        """Check if KSG server is running."""
        result = self._request_sync("GET", "/health")
        return "error" not in result


# =============================================================================
# KSG-BACKED VERIFIED MEMORY
# =============================================================================

class KSGVerifiedMemory:
    """
    Verified Memory implementation using KnowShowGo as backend.
    
    Replaces simple JSON-based VerifiedMemory with full semantic KG:
    - Fuzzy search via embeddings
    - Version history for auditing
    - Weighted associations for provenance
    - Prototype-based schema flexibility
    
    Usage:
        memory = KSGVerifiedMemory("http://localhost:3000")
        memory.store_verified_claim("Bell", "invented", "telephone", ...)
        results = memory.retrieve("who invented telephone")
    """
    
    def __init__(self, ksg_url: str = "http://localhost:3000", embed_fn: Callable = None):
        self.ksg = KSGClient(ksg_url, embed_fn)
        self._initialized = False
    
    def initialize(self):
        """Initialize KSG with verification prototypes."""
        if self._initialized:
            return
        self.ksg.ensure_prototypes()
        self._initialized = True
    
    def store_verified_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        raw_text: str,
        status: str,
        confidence: float,
        source_type: str,
        source_id: str
    ) -> dict:
        """Store a verified claim with full provenance."""
        self.initialize()
        
        claim_uuid = self.ksg.create_concept("Claim", {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "raw_text": raw_text,
            "confidence": confidence
        }, text_for_embedding=f"{subject} {predicate} {obj}")
        
        verification_uuid = self.ksg.create_concept("VerificationResult", {
            "status": status,
            "confidence": confidence,
            "method": source_type,
            "timestamp": time.time()
        })
        
        source_uuid = self.ksg.create_concept("Source", {
            "type": source_type,
            "identifier": source_id,
            "trust_score": confidence
        })
        
        self.ksg.create_association(claim_uuid, verification_uuid, "verified_by", weight=confidence)
        self.ksg.create_association(claim_uuid, source_uuid, "derived_from", weight=confidence)
        
        return {
            "claim_uuid": claim_uuid,
            "verification_uuid": verification_uuid,
            "source_uuid": source_uuid
        }
    
    def retrieve(self, query: str, top_k: int = 5, min_confidence: float = 0.5) -> List[dict]:
        """Retrieve relevant verified claims for RAG."""
        self.initialize()
        results = self.ksg.search(query, top_k=top_k, threshold=min_confidence)
        
        enriched = []
        for result in results:
            associations = self.ksg.get_associations(result.get("uuid", ""))
            enriched.append({
                "claim": result,
                "provenance": associations,
                "similarity": result.get("similarity", 0)
            })
        return enriched
    
    def get_rag_context(self, query: str, top_k: int = 5) -> str:
        """Get verified facts formatted as RAG context."""
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return ""
        
        lines = ["Verified facts from knowledge graph:"]
        for r in results:
            claim = r["claim"]
            props = claim.get("props", {})
            similarity = r.get("similarity", 0)
            lines.append(
                f"- {props.get('subject', '?')} {props.get('predicate', '?')} "
                f"{props.get('object', '?')} (similarity: {similarity:.0%})"
            )
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate KnowShowGo ground truth for hallucination detection."""
    print("=" * 70)
    print("KNOWSHOWGO GROUND TRUTH DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    gt = KSGGroundTruth()
    
    print("\nLoading ground truth...")
    gt.add_verified_fact("Alexander Graham Bell", "invented", "the telephone",
        sources=[{"url": "wikipedia.org/wiki/Telephone", "trust_score": 0.95}])
    gt.add_verified_fact("Albert Einstein", "discovered", "the theory of relativity",
        sources=[{"url": "wikipedia.org/wiki/Relativity", "trust_score": 0.95}])
    gt.add_verified_fact("Thomas Edison", "invented", "the lightbulb",
        sources=[{"url": "wikipedia.org/wiki/Lightbulb", "trust_score": 0.90}])
    gt.add_verified_fact("Marie Curie", "discovered", "radium",
        sources=[{"url": "wikipedia.org/wiki/Radium", "trust_score": 0.95}])
    
    print(f"Ground truth stats: {gt.stats()}")
    
    test_claims = [
        "Bell invented the telephone.",           # VERIFY
        "Edison invented the telephone.",         # REFUTE
        "Einstein discovered relativity.",        # VERIFY
        "Einstein invented the lightbulb.",       # REFUTE
        "Nikola Tesla invented the radio.",       # UNVERIFIED
    ]
    
    print("\n" + "=" * 70)
    print("CHECKING CLAIMS AGAINST GROUND TRUTH")
    print("=" * 70)
    
    for claim in test_claims:
        print(f"\n{'─' * 70}")
        print(f"CLAIM: {claim}")
        result = gt.check(claim)
        
        status_icon = {
            PropositionStatus.VERIFIED: "✓",
            PropositionStatus.REFUTED: "✗",
            PropositionStatus.UNVERIFIED: "?"
        }.get(result.status, "?")
        
        print(f"STATUS: {status_icon} {result.status.value.upper()}")
        print(f"CONFIDENCE: {result.confidence:.0%}")
        print(f"REASON: {result.reason}")
    
    print("\n" + "=" * 70)
    print("KSG ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
    KnowShowGo provides GROUND TRUTH for hallucination detection:
    
    1. PROPOSITIONS = Claims as RDF triples (subject → predicate → object)
    2. PROTOTYPES = Categories (VerifiedProposition, RefutedProposition)
    3. EXEMPLARS = Verified instances that define categories
    4. SUB-CLAIMS = Decomposed propositions for granular verification
    5. ASSOCIATIONS = Provenance graph with weighted edges
    6. FUZZY MATCHING = Embeddings for semantic similarity
    
    This creates a SEMANTIC MEMORY that serves as:
    - Ground truth for LLM verification
    - RAG retrieval layer
    - Audit trail for provenance
    """)


if __name__ == "__main__":
    demo()
