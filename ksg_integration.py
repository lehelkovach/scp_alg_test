"""
KnowShowGo Integration for Verified Memory System
==================================================

This module shows how to use KnowShowGo as the semantic memory backend
for the hallucination detection / verification system.

KnowShowGo provides:
- Fuzzy ontology with prototypes (claim types, entity types)
- Versioned concepts (verified facts with history)
- Weighted associations (provenance, confidence links)
- Embedding-based semantic search

Architecture:
    LLM Output → VerificationAgent → KnowShowGo Backend
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              Prototypes            Concepts             Associations
              ───────────           ────────             ────────────
              • Claim               • Verified facts     • supports
              • Entity              • Entities           • contradicts  
              • Source              • Sources            • derived_from
              • Provenance          • Provenance records • verified_by
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import json
import time

# Note: This is a Python adapter for the JS KnowShowGo API
# In production, you'd call the REST API or use a Python port


# =============================================================================
# KSG PROTOTYPES FOR VERIFICATION SYSTEM
# =============================================================================

KSG_PROTOTYPES = {
    "Claim": {
        "name": "Claim",
        "description": "A factual claim extracted from text",
        "context": "verification",
        "labels": ["claim", "fact", "statement"],
        "schema": {
            "subject": "string",
            "predicate": "string",
            "object": "string",
            "raw_text": "string",
            "confidence": "number"
        }
    },
    "VerificationResult": {
        "name": "VerificationResult",
        "description": "Result of verifying a claim",
        "context": "verification",
        "labels": ["verification", "result", "proof"],
        "schema": {
            "status": "enum:verified,refuted,unverifiable",
            "confidence": "number",
            "method": "string",
            "timestamp": "number"
        }
    },
    "Source": {
        "name": "Source",
        "description": "A source of information",
        "context": "provenance",
        "labels": ["source", "reference", "origin"],
        "schema": {
            "type": "enum:knowledge_base,llm,document,human",
            "identifier": "string",
            "trust_score": "number"
        }
    },
    "Entity": {
        "name": "Entity",
        "description": "A named entity (person, place, thing)",
        "context": "knowledge",
        "labels": ["entity", "noun", "thing"],
        "schema": {
            "name": "string",
            "type": "string",
            "aliases": "array"
        }
    }
}

KSG_ASSOCIATION_TYPES = {
    "supports": {
        "description": "Evidence supports a claim",
        "weight_meaning": "strength of support"
    },
    "contradicts": {
        "description": "Evidence contradicts a claim",
        "weight_meaning": "strength of contradiction"
    },
    "derived_from": {
        "description": "Claim derived from source",
        "weight_meaning": "derivation confidence"
    },
    "verified_by": {
        "description": "Claim verified by method/source",
        "weight_meaning": "verification confidence"
    },
    "same_as": {
        "description": "Two entities are the same",
        "weight_meaning": "confidence of identity"
    },
    "related_to": {
        "description": "General relation between concepts",
        "weight_meaning": "relation strength"
    }
}


# =============================================================================
# KSG CLIENT (Python wrapper for REST API)
# =============================================================================

class KSGClient:
    """
    Python client for KnowShowGo REST API.
    
    Usage:
        ksg = KSGClient("http://localhost:3000")
        
        # Create a claim
        claim_id = await ksg.create_concept("Claim", {
            "subject": "Einstein",
            "predicate": "discovered",
            "object": "relativity"
        })
        
        # Search for similar claims
        results = await ksg.search("Einstein discovered", top_k=5)
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", embed_fn: Callable = None):
        self.base_url = base_url.rstrip("/")
        self.embed_fn = embed_fn or self._default_embed
        self._prototype_cache = {}
    
    def _default_embed(self, text: str) -> List[float]:
        """Default embedding (hash-based, for testing)."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:64]]
    
    async def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make HTTP request to KSG API."""
        import aiohttp
        url = f"{self.base_url}{path}"
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, json=data) as resp:
                    return await resp.json()
            elif method == "PUT":
                async with session.put(url, json=data) as resp:
                    return await resp.json()
    
    # Synchronous versions for simpler usage
    def _request_sync(self, method: str, path: str, data: dict = None) -> dict:
        """Synchronous HTTP request."""
        import requests
        url = f"{self.base_url}{path}"
        
        if method == "GET":
            resp = requests.get(url)
        elif method == "POST":
            resp = requests.post(url, json=data)
        elif method == "PUT":
            resp = requests.put(url, json=data)
        
        return resp.json()
    
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
                print(f"Prototype {name} may already exist: {e}")
    
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
        
        return result.get("uuid")
    
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
        
        return result.get("uuid")
    
    def get_associations(self, concept_uuid: str, association_type: str = None) -> List[dict]:
        """Get associations for a concept."""
        path = f"/concepts/{concept_uuid}/associations"
        if association_type:
            path += f"?type={association_type}"
        
        return self._request_sync("GET", path)
    
    def get_history(self, concept_uuid: str) -> List[dict]:
        """Get version history for a concept (audit trail)."""
        return self._request_sync("GET", f"/concepts/{concept_uuid}/history")


# =============================================================================
# KSG-BACKED VERIFIED MEMORY
# =============================================================================

class KSGVerifiedMemory:
    """
    Verified Memory implementation using KnowShowGo as backend.
    
    This replaces the simple JSON-based VerifiedMemory with a full
    semantic knowledge graph that supports:
    - Fuzzy search via embeddings
    - Version history for auditing
    - Weighted associations for provenance
    - Prototype-based schema flexibility
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
        """
        Store a verified claim with full provenance.
        
        Returns dict with claim_uuid and provenance_uuid for auditing.
        """
        self.initialize()
        
        # 1. Create the Claim concept
        claim_uuid = self.ksg.create_concept("Claim", {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "raw_text": raw_text,
            "confidence": confidence
        }, text_for_embedding=f"{subject} {predicate} {obj}")
        
        # 2. Create the VerificationResult concept
        verification_uuid = self.ksg.create_concept("VerificationResult", {
            "status": status,
            "confidence": confidence,
            "method": source_type,
            "timestamp": time.time()
        })
        
        # 3. Create the Source concept
        source_uuid = self.ksg.create_concept("Source", {
            "type": source_type,
            "identifier": source_id,
            "trust_score": confidence
        })
        
        # 4. Create associations (provenance links)
        self.ksg.create_association(
            claim_uuid, verification_uuid, 
            "verified_by", 
            weight=confidence
        )
        
        self.ksg.create_association(
            claim_uuid, source_uuid,
            "derived_from",
            weight=confidence
        )
        
        return {
            "claim_uuid": claim_uuid,
            "verification_uuid": verification_uuid,
            "source_uuid": source_uuid
        }
    
    def retrieve(self, query: str, top_k: int = 5, min_confidence: float = 0.5) -> List[dict]:
        """
        Retrieve relevant verified claims for RAG.
        
        Returns claims with their provenance chain.
        """
        self.initialize()
        
        results = self.ksg.search(query, top_k=top_k, threshold=min_confidence)
        
        enriched = []
        for result in results:
            # Get provenance associations
            associations = self.ksg.get_associations(result["uuid"])
            
            enriched.append({
                "claim": result,
                "provenance": associations,
                "similarity": result.get("similarity", 0)
            })
        
        return enriched
    
    def get_audit_trail(self, claim_uuid: str) -> dict:
        """
        Get complete audit trail for a claim.
        
        Returns version history + all provenance associations.
        """
        history = self.ksg.get_history(claim_uuid)
        associations = self.ksg.get_associations(claim_uuid)
        
        return {
            "claim_uuid": claim_uuid,
            "version_history": history,
            "provenance": associations
        }
    
    def get_rag_context(self, query: str, top_k: int = 5) -> str:
        """
        Get verified facts formatted as RAG context.
        """
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
# INTEGRATION WITH VERIFICATION AGENT
# =============================================================================

def create_ksg_verifier(
    ksg_url: str = "http://localhost:3000",
    embed_fn: Callable = None,
    initial_facts: List[tuple] = None
):
    """
    Create a verification agent backed by KnowShowGo.
    
    Usage:
        verifier = create_ksg_verifier(
            ksg_url="http://localhost:3000",
            initial_facts=[
                ("Einstein", "discovered", "relativity"),
                ("Bell", "invented", "telephone"),
            ]
        )
        
        # Verify a claim
        result = verifier.verify("Edison invented the telephone")
        
        # Get RAG context
        context = verifier.get_rag_context("who invented things")
    """
    from verified_memory import VerificationAgent, HallucinationProver
    
    # Create KSG-backed memory
    memory = KSGVerifiedMemory(ksg_url, embed_fn)
    
    # Create prober with initial facts
    prover = HallucinationProver()
    if initial_facts:
        prover.add_facts_bulk(initial_facts, source="initial")
    
    # Create agent with KSG memory
    # Note: We need a thin adapter since VerificationAgent expects VerifiedMemory interface
    class KSGMemoryAdapter:
        def __init__(self, ksg_memory):
            self.ksg = ksg_memory
        
        def store(self, claim):
            return self.ksg.store_verified_claim(
                subject=claim.subject,
                predicate=claim.predicate,
                obj=claim.obj,
                raw_text=claim.text,
                status=claim.status.value,
                confidence=claim.provenance.confidence,
                source_type=claim.provenance.method,
                source_id=claim.provenance.source
            )
        
        def retrieve(self, query, top_k=5, min_confidence=0.5, status_filter=None):
            return self.ksg.retrieve(query, top_k, min_confidence)
        
        def stats(self):
            return {"backend": "knowshowgo", "url": self.ksg.ksg.base_url}
    
    adapter = KSGMemoryAdapter(memory)
    
    return VerificationAgent(memory=adapter, prover=prover)


# =============================================================================
# ARCHITECTURE DIAGRAM
# =============================================================================

ARCHITECTURE = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    VERIFICATION SYSTEM WITH KNOWSHOWGO                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────┐                                                          ║
║  │  LLM Output     │                                                          ║
║  │  "Edison..."    │                                                          ║
║  └────────┬────────┘                                                          ║
║           │                                                                   ║
║           ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                     VERIFICATION AGENT                                  │  ║
║  │                                                                         │  ║
║  │  1. Extract claims (RuleBasedExtractor / LLM)                           │  ║
║  │  2. Prove against KB (HallucinationProver)                              │  ║
║  │  3. Store in KnowShowGo                                                 │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║           │                                                                   ║
║           ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         KNOWSHOWGO                                      │  ║
║  │  ═══════════════════════════════════════════════════════════════════   │  ║
║  │                                                                         │  ║
║  │  PROTOTYPES                CONCEPTS                 ASSOCIATIONS        │  ║
║  │  ──────────                ────────                 ────────────        │  ║
║  │  ┌─────────┐              ┌─────────┐              ┌─────────────┐      │  ║
║  │  │ Claim   │──creates────▶│ claim_1 │◀──verified_by──│ result_1  │      │  ║
║  │  │ schema  │              │ Einstein│              │ KB lookup   │      │  ║
║  │  └─────────┘              │ discover│              └─────────────┘      │  ║
║  │                           │ relativ │                    │              │  ║
║  │  ┌─────────┐              └────┬────┘                    │              │  ║
║  │  │ Source  │                   │                         │              │  ║
║  │  │ schema  │──creates────▶┌────▼────┐◀───derived_from────┘              │  ║
║  │                           │ source_1│                                   │  ║
║  │                           │ KB v1.0 │                                   │  ║
║  │                           └─────────┘                                   │  ║
║  │                                                                         │  ║
║  │  FEATURES:                                                              │  ║
║  │  • Fuzzy search via embeddings                                          │  ║
║  │  • Version history on every concept                                     │  ║
║  │  • Weighted associations = confidence scores                            │  ║
║  │  • Full audit trail via associations                                    │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║           │                                                                   ║
║           ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         RAG RETRIEVAL                                   │  ║
║  │                                                                         │  ║
║  │  verifier.get_rag_context("who invented telephone")                     │  ║
║  │                                                                         │  ║
║  │  Returns:                                                               │  ║
║  │  "Verified facts from knowledge graph:                                  │  ║
║  │   - Bell invented telephone (similarity: 95%)                           │  ║
║  │   - Edison invented lightbulb (similarity: 72%)"                        │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def print_architecture():
    print(ARCHITECTURE)


if __name__ == "__main__":
    print_architecture()
    print("\nKnowShowGo Prototypes for Verification:")
    print(json.dumps(KSG_PROTOTYPES, indent=2))
