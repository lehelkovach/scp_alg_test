"""
Verified Memory System
======================

A simple Layer-2 verification and caching system for LLM outputs.

Components:
1. HallucinationProver - Tests if a claim is hallucinated
2. VerificationAgent - Verifies answers and adds provenance  
3. VerifiedMemory - Persistent cache of verified facts (RAG layer)

Usage:
    memory = VerifiedMemory("./memory_cache")
    agent = VerificationAgent(memory, llm_fn)
    
    # Verify an LLM answer
    result = agent.verify("Einstein invented the telephone")
    # Returns: verified=False, provenance="Bell invented telephone"
    
    # Query verified memory (RAG)
    facts = memory.retrieve("who invented telephone")
    # Returns cached, verified facts with provenance
"""

import json
import os
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Tuple, Any
from enum import Enum
from pathlib import Path

# Import SCP prober for hallucination detection
try:
    from test import (
        HyperKB, SCPProber, RuleBasedExtractor, Claim, Verdict,
        StringSimilarityBackend, EMBEDDINGS_AVAILABLE
    )
    if EMBEDDINGS_AVAILABLE:
        from test import SentenceTransformerBackend
    SCP_AVAILABLE = True
except ImportError:
    SCP_AVAILABLE = False
    print("Warning: SCP module not available. Install dependencies: pip install networkx numpy")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class VerificationStatus(Enum):
    VERIFIED = "verified"           # Confirmed true
    REFUTED = "refuted"             # Confirmed false
    UNVERIFIABLE = "unverifiable"   # Cannot determine
    CACHED = "cached"               # From previous verification


@dataclass
class Provenance:
    """Tracks where a verification came from."""
    method: str                     # "scp_kb", "llm_consensus", "external_source"
    source: str                     # Specific source identifier
    timestamp: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        return cls(**d)


@dataclass
class VerifiedClaim:
    """A claim with verification status and provenance."""
    text: str                       # Original claim text
    subject: str
    predicate: str  
    obj: str
    status: VerificationStatus
    provenance: Provenance
    embedding: Optional[List[float]] = None  # For semantic search
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "status": self.status.value,
            "provenance": self.provenance.to_dict(),
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedClaim":
        return cls(
            text=d["text"],
            subject=d["subject"],
            predicate=d["predicate"],
            obj=d["object"],
            status=VerificationStatus(d["status"]),
            provenance=Provenance.from_dict(d["provenance"]),
            embedding=d.get("embedding")
        )


@dataclass
class VerificationResult:
    """Result of verifying an LLM answer."""
    original_text: str
    claims: List[VerifiedClaim]
    overall_status: VerificationStatus
    confidence: float
    summary: str
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "claims": [c.to_dict() for c in self.claims],
            "overall_status": self.overall_status.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "audit_trail": self.audit_trail
        }
    
    def __str__(self) -> str:
        lines = [
            f"Verification Result",
            f"=" * 50,
            f"Text: {self.original_text[:100]}...",
            f"Status: {self.overall_status.value}",
            f"Confidence: {self.confidence:.2%}",
            f"Summary: {self.summary}",
            f"Claims verified: {len(self.claims)}",
        ]
        for i, claim in enumerate(self.claims, 1):
            lines.append(f"  [{i}] {claim.status.value}: {claim.text[:60]}...")
            lines.append(f"      Source: {claim.provenance.source}")
        return "\n".join(lines)


# =============================================================================
# HALLUCINATION PROVER (using SCP)
# =============================================================================

class HallucinationProver:
    """
    Tests if a claim is a hallucination using the SCP algorithm.
    
    This is the core "proving" component that checks claims against
    a knowledge base using semantic similarity.
    """
    
    def __init__(self, knowledge_base: Optional["HyperKB"] = None):
        if not SCP_AVAILABLE:
            raise RuntimeError("SCP module required. Run: pip install networkx numpy")
        
        # Initialize embedding backend
        if EMBEDDINGS_AVAILABLE:
            self.backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
        else:
            self.backend = StringSimilarityBackend()
        
        # Use provided KB or create new one
        self.kb = knowledge_base or HyperKB(embedding_backend=self.backend)
        
        # Create prober
        self.prober = SCPProber(
            kb=self.kb,
            extractor=RuleBasedExtractor(),
            soft_threshold=0.70
        )
    
    def prove(self, claim_text: str) -> Tuple[VerificationStatus, Provenance, Optional[Claim]]:
        """
        Prove whether a claim is true, false, or unknown.
        
        Returns:
            (status, provenance, extracted_claim)
        """
        # Probe the claim
        report = self.prober.probe(claim_text)
        
        if not report.results:
            return (
                VerificationStatus.UNVERIFIABLE,
                Provenance(
                    method="scp_kb",
                    source="no_claims_extracted",
                    timestamp=time.time(),
                    confidence=0.0,
                    details={"reason": "Could not extract claims from text"}
                ),
                None
            )
        
        result = report.results[0]
        
        # Map SCP verdict to verification status
        if result.verdict == Verdict.PASS:
            status = VerificationStatus.VERIFIED
            confidence = 1.0
        elif result.verdict == Verdict.SOFT_PASS:
            status = VerificationStatus.VERIFIED
            confidence = result.score
        elif result.verdict in [Verdict.FAIL, Verdict.CONTRADICT]:
            status = VerificationStatus.REFUTED
            confidence = 1.0 - result.score if result.score > 0 else 0.9
        else:
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.5
        
        provenance = Provenance(
            method="scp_kb",
            source="knowledge_base_lookup",
            timestamp=time.time(),
            confidence=confidence,
            details={
                "verdict": result.verdict.value,
                "score": result.score,
                "matched_facts": result.matched_facts,
                "reason": result.reason
            }
        )
        
        # Extract the claim object
        extracted_claim = Claim(
            subject=result.claim.subject,
            predicate=result.claim.predicate,
            obj=result.claim.obj,
            raw=result.claim.raw
        ) if result.claim else None
        
        return status, provenance, extracted_claim
    
    def add_fact(self, subject: str, predicate: str, obj: str, source: str = "manual"):
        """Add a verified fact to the knowledge base."""
        self.kb.add_fact(subject, predicate, obj, source=source, confidence=1.0)
    
    def add_facts_bulk(self, facts: List[Tuple[str, str, str]], source: str = "bulk"):
        """Add multiple facts to the knowledge base."""
        self.kb.add_facts_bulk(facts, source=source, confidence=1.0)


# =============================================================================
# VERIFIED MEMORY (Persistent Cache / RAG Layer)
# =============================================================================

class VerifiedMemory:
    """
    Persistent cache of verified claims with semantic search.
    
    Acts as a "Layer 2" RAG system - stores verified facts that can be
    retrieved to augment future LLM queries.
    
    Features:
    - Persistent storage (JSON files)
    - Semantic search for relevant facts
    - Provenance tracking for auditing
    - Confidence-based filtering
    """
    
    def __init__(self, cache_dir: str = "./verified_memory"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.claims_file = self.cache_dir / "verified_claims.json"
        self.index_file = self.cache_dir / "claim_index.json"
        
        # Load existing cache
        self.claims: Dict[str, VerifiedClaim] = {}
        self.index: Dict[str, List[str]] = {}  # keyword -> claim_ids
        self._load_cache()
        
        # Embedding model for semantic search
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.claims_file.exists():
            try:
                with open(self.claims_file, 'r') as f:
                    data = json.load(f)
                    self.claims = {
                        k: VerifiedClaim.from_dict(v) 
                        for k, v in data.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except:
                pass
    
    def _save_cache(self):
        """Save cache to disk."""
        with open(self.claims_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.claims.items()}, f, indent=2)
        
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _claim_id(self, claim: VerifiedClaim) -> str:
        """Generate unique ID for a claim."""
        key = f"{claim.subject}|{claim.predicate}|{claim.obj}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def _index_claim(self, claim_id: str, claim: VerifiedClaim):
        """Add claim to keyword index."""
        keywords = set()
        for text in [claim.subject, claim.predicate, claim.obj, claim.text]:
            keywords.update(text.lower().split())
        
        for keyword in keywords:
            if keyword not in self.index:
                self.index[keyword] = []
            if claim_id not in self.index[keyword]:
                self.index[keyword].append(claim_id)
    
    def store(self, claim: VerifiedClaim) -> str:
        """
        Store a verified claim in memory.
        
        Returns: claim_id
        """
        # Generate embedding for semantic search
        if self.encoder and claim.embedding is None:
            claim.embedding = self.encoder.encode([claim.text])[0].tolist()
        
        claim_id = self._claim_id(claim)
        self.claims[claim_id] = claim
        self._index_claim(claim_id, claim)
        self._save_cache()
        
        return claim_id
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        min_confidence: float = 0.5,
        status_filter: Optional[List[VerificationStatus]] = None
    ) -> List[VerifiedClaim]:
        """
        Retrieve relevant verified claims for a query.
        
        This is the RAG retrieval function - use it to augment LLM context
        with verified facts.
        """
        if status_filter is None:
            status_filter = [VerificationStatus.VERIFIED, VerificationStatus.CACHED]
        
        candidates = []
        
        # Semantic search if embeddings available
        if self.encoder:
            query_emb = self.encoder.encode([query])[0]
            
            for claim_id, claim in self.claims.items():
                if claim.status not in status_filter:
                    continue
                if claim.provenance.confidence < min_confidence:
                    continue
                
                if claim.embedding:
                    # Cosine similarity
                    import numpy as np
                    claim_emb = np.array(claim.embedding)
                    score = float(np.dot(query_emb, claim_emb) / 
                                (np.linalg.norm(query_emb) * np.linalg.norm(claim_emb)))
                    candidates.append((score, claim))
        else:
            # Keyword fallback
            query_keywords = set(query.lower().split())
            
            for claim_id, claim in self.claims.items():
                if claim.status not in status_filter:
                    continue
                if claim.provenance.confidence < min_confidence:
                    continue
                
                claim_keywords = set(claim.text.lower().split())
                overlap = len(query_keywords & claim_keywords)
                if overlap > 0:
                    score = overlap / len(query_keywords)
                    candidates.append((score, claim))
        
        # Sort by score and return top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [claim for score, claim in candidates[:top_k]]
    
    def get_audit_trail(self, claim_id: str) -> Optional[Dict]:
        """Get full provenance for auditing."""
        if claim_id not in self.claims:
            return None
        
        claim = self.claims[claim_id]
        return {
            "claim_id": claim_id,
            "claim": claim.to_dict(),
            "provenance": claim.provenance.to_dict()
        }
    
    def stats(self) -> Dict:
        """Get memory statistics."""
        status_counts = {}
        for claim in self.claims.values():
            status = claim.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_claims": len(self.claims),
            "by_status": status_counts,
            "cache_dir": str(self.cache_dir)
        }


# =============================================================================
# VERIFICATION AGENT
# =============================================================================

class VerificationAgent:
    """
    Agent that verifies LLM answers and stores results.
    
    Workflow:
    1. Extract claims from LLM answer
    2. Check each claim against KB (HallucinationProver)
    3. Optionally verify with external LLM
    4. Store verified results in memory (VerifiedMemory)
    5. Return result with provenance for auditing
    """
    
    def __init__(
        self,
        memory: VerifiedMemory,
        llm_fn: Optional[Callable[[str], str]] = None,
        prover: Optional[HallucinationProver] = None
    ):
        self.memory = memory
        self.llm_fn = llm_fn
        self.prover = prover or HallucinationProver()
    
    def verify(self, text: str, use_llm_fallback: bool = True) -> VerificationResult:
        """
        Verify an LLM answer and return detailed result.
        
        Args:
            text: The LLM output to verify
            use_llm_fallback: If True, use LLM for claims not in KB
        
        Returns:
            VerificationResult with provenance and audit trail
        """
        audit_trail = []
        verified_claims = []
        
        # Step 1: Check memory cache first
        cached = self.memory.retrieve(text, top_k=3, min_confidence=0.8)
        if cached:
            audit_trail.append({
                "step": "cache_check",
                "result": f"Found {len(cached)} cached claims"
            })
        
        # Step 2: Prove using SCP
        status, provenance, extracted_claim = self.prover.prove(text)
        
        audit_trail.append({
            "step": "scp_proof",
            "status": status.value,
            "confidence": provenance.confidence,
            "details": provenance.details
        })
        
        if extracted_claim:
            verified_claim = VerifiedClaim(
                text=text,
                subject=extracted_claim.subject,
                predicate=extracted_claim.predicate,
                obj=extracted_claim.obj,
                status=status,
                provenance=provenance
            )
            verified_claims.append(verified_claim)
            
            # Store in memory if verified
            if status == VerificationStatus.VERIFIED:
                claim_id = self.memory.store(verified_claim)
                audit_trail.append({
                    "step": "memory_store",
                    "claim_id": claim_id
                })
        
        # Step 3: LLM fallback for unverifiable claims
        if status == VerificationStatus.UNVERIFIABLE and use_llm_fallback and self.llm_fn:
            llm_result = self._verify_with_llm(text)
            audit_trail.append({
                "step": "llm_fallback",
                "result": llm_result
            })
            
            if llm_result.get("verified"):
                status = VerificationStatus.VERIFIED
                provenance = Provenance(
                    method="llm_verification",
                    source="llm_judge",
                    timestamp=time.time(),
                    confidence=llm_result.get("confidence", 0.7),
                    details=llm_result
                )
        
        # Build summary
        if status == VerificationStatus.VERIFIED:
            summary = f"Verified with {provenance.confidence:.0%} confidence via {provenance.method}"
        elif status == VerificationStatus.REFUTED:
            summary = f"Refuted: {provenance.details.get('reason', 'Contradicts known facts')}"
        else:
            summary = "Could not verify - insufficient evidence"
        
        return VerificationResult(
            original_text=text,
            claims=verified_claims,
            overall_status=status,
            confidence=provenance.confidence,
            summary=summary,
            audit_trail=audit_trail
        )
    
    def _verify_with_llm(self, claim_text: str) -> Dict:
        """Use LLM to verify a claim."""
        if not self.llm_fn:
            return {"verified": False, "reason": "No LLM available"}
        
        prompt = f"""Verify if this statement is factually correct.
        
Statement: {claim_text}

Respond with JSON:
{{"verified": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}"""
        
        try:
            response = self.llm_fn(prompt)
            return json.loads(response)
        except:
            return {"verified": False, "reason": "LLM verification failed"}
    
    def add_trusted_facts(self, facts: List[Tuple[str, str, str]], source: str = "trusted"):
        """Add trusted facts to the prover's knowledge base."""
        self.prover.add_facts_bulk(facts, source=source)
    
    def get_rag_context(self, query: str, top_k: int = 5) -> str:
        """
        Get verified facts as RAG context for an LLM query.
        
        Use this to augment LLM prompts with verified information.
        """
        facts = self.memory.retrieve(query, top_k=top_k)
        
        if not facts:
            return ""
        
        context_lines = ["Verified facts:"]
        for fact in facts:
            context_lines.append(
                f"- {fact.subject} {fact.predicate} {fact.obj} "
                f"(confidence: {fact.provenance.confidence:.0%})"
            )
        
        return "\n".join(context_lines)


# =============================================================================
# SIMPLE API
# =============================================================================

def create_verifier(
    cache_dir: str = "./verified_memory",
    initial_facts: Optional[List[Tuple[str, str, str]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None
) -> VerificationAgent:
    """
    Create a verification agent with sensible defaults.
    
    Usage:
        verifier = create_verifier(
            initial_facts=[
                ("Einstein", "discovered", "relativity"),
                ("Bell", "invented", "telephone"),
            ]
        )
        
        result = verifier.verify("Edison invented the telephone")
        print(result)  # REFUTED: Bell invented telephone
    """
    memory = VerifiedMemory(cache_dir)
    prover = HallucinationProver()
    
    if initial_facts:
        prover.add_facts_bulk(initial_facts, source="initial")
    
    return VerificationAgent(memory=memory, llm_fn=llm_fn, prover=prover)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the verification system."""
    print("=" * 60)
    print("VERIFIED MEMORY SYSTEM DEMO")
    print("=" * 60)
    
    # Create verifier with some initial facts
    verifier = create_verifier(
        cache_dir="./demo_memory",
        initial_facts=[
            ("The Eiffel Tower", "located_in", "Paris"),
            ("Albert Einstein", "discovered", "the theory of relativity"),
            ("Alexander Graham Bell", "invented", "the telephone"),
            ("Marie Curie", "discovered", "radium"),
            ("Python", "created_by", "Guido van Rossum"),
        ]
    )
    
    # Test cases
    test_claims = [
        "The Eiffel Tower is located in Paris.",      # Should VERIFY
        "Einstein discovered relativity.",             # Should VERIFY (soft match)
        "Edison invented the telephone.",              # Should REFUTE
        "The Eiffel Tower is in London.",             # Should REFUTE
        "Quantum computers use qubits.",              # Should be UNVERIFIABLE
    ]
    
    print("\nVerifying claims:")
    print("-" * 60)
    
    for claim in test_claims:
        result = verifier.verify(claim, use_llm_fallback=False)
        print(f"\nClaim: {claim}")
        print(f"Status: {result.overall_status.value}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Summary: {result.summary}")
    
    # Show memory stats
    print("\n" + "=" * 60)
    print("MEMORY STATS")
    print("=" * 60)
    print(json.dumps(verifier.memory.stats(), indent=2))
    
    # Demo RAG context
    print("\n" + "=" * 60)
    print("RAG CONTEXT EXAMPLE")
    print("=" * 60)
    context = verifier.get_rag_context("Who invented things?")
    print(context or "No verified facts found for query")
    
    # Show audit trail
    print("\n" + "=" * 60)
    print("AUDIT TRAIL EXAMPLE")
    print("=" * 60)
    result = verifier.verify("Bell invented the telephone.")
    print(json.dumps(result.audit_trail, indent=2))


if __name__ == "__main__":
    demo()
