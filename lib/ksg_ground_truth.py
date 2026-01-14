"""
KnowShowGo as Ground Truth for Hallucination Detection
=======================================================

KnowShowGo's design maps perfectly to hallucination detection:

1. PROPOSITIONS (RDF triples) = Claims to verify
   - subject_uuid → predicate_uuid → object_uuid
   - Every claim is a proposition in the semantic graph

2. PROTOTYPES = Claim categories / schemas
   - VerifiedProposition: ground truth facts
   - RefutedProposition: known false claims
   - UnverifiedProposition: claims pending verification

3. EXEMPLARS = Verified instances that define categories
   - "Einstein discovered relativity" is an exemplar of ScientificDiscovery
   - New claims are compared to exemplars via fuzzy matching

4. SUB-CLAIMS = Decomposed propositions
   - Complex claims break into sub-propositions
   - Each sub-claim verified independently
   - Provenance tracks decomposition

5. FUZZY DAG = Knowledge graph with weighted edges
   - Associations have weights (confidence)
   - Edges track provenance (where fact came from)
   - DAG structure prevents circular reasoning

Architecture:
─────────────────────────────────────────────────────────────────────────────────

                    ┌──────────────────────────────────────────┐
                    │              LLM OUTPUT                  │
                    │    "Einstein invented the telephone"     │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CLAIM EXTRACTION                                     │
│                                                                              │
│   Extract as KSG Proposition:                                                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │ Subject     │───▶│ Predicate   │───▶│ Object      │                      │
│   │ "Einstein"  │    │ "invented"  │    │ "telephone" │                      │
│   └─────────────┘    └─────────────┘    └─────────────┘                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         KNOWSHOWGO GROUND TRUTH                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   PROTOTYPES (Schemas)                                                       │
│   ────────────────────                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  VerifiedProposition (prototype)                                    │   │
│   │    schema: { subject, predicate, object, confidence, sources }      │   │
│   │    exemplars: [ Edison/lightbulb, Bell/telephone, ... ]             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   FUZZY SEARCH (Embeddings)                                                  │
│   ─────────────────────────                                                  │
│   Query: "Einstein invented telephone"                                       │
│   Matches:                                                                   │
│     • "Bell invented telephone" (similarity: 0.89) ← CONTRADICTS             │
│     • "Einstein discovered relativity" (similarity: 0.72)                    │
│                                                                              │
│   ASSOCIATIONS (Provenance Graph)                                            │
│   ───────────────────────────────                                            │
│   ┌─────────┐ supports(0.95) ┌─────────────────┐                             │
│   │ Bell    │───────────────▶│ invented        │                             │
│   │ invented│                │ telephone       │                             │
│   │ phone   │◀───────────────│ (ground truth)  │                             │
│   └─────────┘ derived_from   └─────────────────┘                             │
│                    │                                                         │
│                    ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Source: Wikipedia, USPTO Patent #174465, confidence: 0.99          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VERIFICATION RESULT                                  │
│                                                                              │
│   VERDICT: REFUTED                                                           │
│   REASON: Ground truth shows Bell invented telephone, not Einstein           │
│   PROVENANCE: Wikipedia, USPTO Patent #174465                                │
│   CONFIDENCE: 0.99                                                           │
│                                                                              │
│   SUB-CLAIMS:                                                                │
│     ✓ "Einstein" is a Person (verified)                                      │
│     ✓ "invented" is a valid predicate (verified)                             │
│     ✓ "telephone" is an Invention (verified)                                 │
│     ✗ "Einstein invented telephone" (refuted - Bell did)                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import time


# =============================================================================
# KSG PROTOTYPES FOR GROUND TRUTH
# =============================================================================

class PropositionStatus(Enum):
    """Status of a proposition in KnowShowGo."""
    VERIFIED = "verified"           # Ground truth - confirmed true
    REFUTED = "refuted"             # Ground truth - confirmed false
    UNVERIFIED = "unverified"       # Pending verification
    CONTESTED = "contested"         # Multiple sources disagree


# Prototype definitions for KnowShowGo
KSG_GROUND_TRUTH_PROTOTYPES = {
    # Core proposition prototype (RDF-style triple)
    "Proposition": {
        "name": "Proposition",
        "description": "An RDF-style triple: subject-predicate-object claim",
        "schema": {
            "subject_uuid": "string",      # UUID of subject concept
            "predicate_uuid": "string",    # UUID of predicate concept
            "object_uuid": "string",       # UUID of object concept
            "raw_text": "string",          # Original text form
            "confidence": "number",        # 0.0-1.0
            "status": "enum:verified,refuted,unverified,contested"
        },
        "parent": "Concept"
    },
    
    # Verified proposition - ground truth
    "VerifiedProposition": {
        "name": "VerifiedProposition",
        "description": "A proposition confirmed as ground truth",
        "schema": {
            "verification_method": "string",  # How it was verified
            "verification_date": "string",
            "verifier": "string",             # Who/what verified it
            "source_urls": "array"            # Supporting sources
        },
        "parent": "Proposition"
    },
    
    # Refuted proposition - known false
    "RefutedProposition": {
        "name": "RefutedProposition",
        "description": "A proposition confirmed as false",
        "schema": {
            "correct_proposition_uuid": "string",  # What IS true
            "refutation_reason": "string",
            "refutation_date": "string"
        },
        "parent": "Proposition"
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
        },
        "parent": "Concept"
    },
    
    # Predicate prototype (verbs/relations)
    "Predicate": {
        "name": "Predicate",
        "description": "A relation type (verb) between entities",
        "schema": {
            "name": "string",
            "inverse_name": "string",         # e.g., "invented" <-> "invented_by"
            "domain_types": "array",          # Valid subject types
            "range_types": "array"            # Valid object types
        },
        "parent": "Concept"
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
        },
        "parent": "Concept"
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
# GROUND TRUTH LAYER
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
    
    # What we found in ground truth
    matching_proposition: Optional[Dict] = None
    contradicting_proposition: Optional[Dict] = None
    
    # Sub-claim analysis
    sub_claims: List[SubClaim] = field(default_factory=list)
    
    # Provenance
    sources: List[Dict] = field(default_factory=list)
    
    # Explanation
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


class KSGGroundTruth:
    """
    KnowShowGo-backed ground truth for hallucination detection.
    
    This layer provides:
    1. Proposition storage (claims as RDF triples)
    2. Fuzzy matching via embeddings
    3. Exemplar-based category matching
    4. Sub-claim decomposition
    5. Provenance tracking via associations
    
    Usage:
        gt = KSGGroundTruth(ksg_client)
        
        # Add ground truth
        gt.add_verified_fact("Bell", "invented", "telephone", sources=[...])
        
        # Check a claim
        result = gt.check("Edison invented the telephone")
        # Returns: REFUTED, reason="Bell invented telephone, not Edison"
    """
    
    def __init__(self, ksg_url: str = "http://localhost:3000"):
        self.ksg_url = ksg_url
        # In production, this would be a real KSG client
        # For now, we use an in-memory simulation
        self._propositions: Dict[str, Dict] = {}
        self._entities: Dict[str, Dict] = {}
        self._predicates: Dict[str, Dict] = {}
        self._associations: List[Dict] = []
        self._embeddings: Dict[str, List[float]] = {}
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder)."""
        import hashlib
        h = hashlib.sha256(text.lower().encode()).digest()
        return [float(b) / 255.0 for b in h[:64]]
    
    def _similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Cosine similarity between embeddings."""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
    
    def add_entity(self, name: str, entity_type: str, aliases: List[str] = None) -> str:
        """Add an entity to the ground truth."""
        uuid = f"entity_{len(self._entities)}"
        self._entities[uuid] = {
            "uuid": uuid,
            "name": name,
            "entity_type": entity_type,
            "aliases": aliases or []
        }
        self._embeddings[uuid] = self._embed(name)
        return uuid
    
    def add_predicate(self, name: str, inverse_name: str = None) -> str:
        """Add a predicate (verb/relation) to the ground truth."""
        uuid = f"predicate_{len(self._predicates)}"
        self._predicates[uuid] = {
            "uuid": uuid,
            "name": name,
            "inverse_name": inverse_name or f"{name}_by"
        }
        self._embeddings[uuid] = self._embed(name)
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
        Add a verified fact (ground truth) to KnowShowGo.
        
        This creates:
        1. Subject entity (if not exists)
        2. Predicate (if not exists)
        3. Object entity (if not exists)
        4. VerifiedProposition linking them
        5. Source associations for provenance
        """
        # Get or create entities
        subject_uuid = self._find_or_create_entity(subject)
        predicate_uuid = self._find_or_create_predicate(predicate)
        object_uuid = self._find_or_create_entity(obj)
        
        # Create proposition
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
        self._embeddings[prop_uuid] = self._embed(raw_text)
        
        # Create source associations
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
        # Search by embedding similarity
        query_emb = self._embed(name)
        best_match = None
        best_score = 0.0
        
        for uuid, entity in self._entities.items():
            if entity["name"].lower() == name.lower():
                return uuid
            if name.lower() in [a.lower() for a in entity.get("aliases", [])]:
                return uuid
            
            score = self._similarity(query_emb, self._embeddings.get(uuid, []))
            if score > best_score and score > 0.9:
                best_score = score
                best_match = uuid
        
        if best_match:
            return best_match
        
        # Create new entity
        return self.add_entity(name, "thing")
    
    def _find_or_create_predicate(self, name: str) -> str:
        """Find predicate by name or create if not exists."""
        for uuid, pred in self._predicates.items():
            if pred["name"].lower() == name.lower():
                return uuid
        
        return self.add_predicate(name)
    
    def check(self, claim_text: str, decompose: bool = True) -> GroundTruthResult:
        """
        Check a claim against KnowShowGo ground truth.
        
        Process:
        1. Extract subject-predicate-object from claim
        2. Search for matching/contradicting propositions via embeddings
        3. Optionally decompose into sub-claims
        4. Return verdict with provenance
        """
        # Extract claim components (simplified - would use NLP in production)
        parts = self._extract_triple(claim_text)
        if not parts:
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.UNVERIFIED,
                confidence=0.0,
                reason="Could not extract subject-predicate-object from claim"
            )
        
        subject, predicate, obj = parts
        
        # Search for matching propositions
        claim_emb = self._embed(claim_text)
        
        matches = []
        contradictions = []
        
        for prop_uuid, prop in self._propositions.items():
            prop_emb = self._embeddings.get(prop_uuid, [])
            similarity = self._similarity(claim_emb, prop_emb)
            
            if similarity > 0.7:
                # Check if it's a match or contradiction
                if self._is_match(subject, predicate, obj, prop):
                    matches.append((similarity, prop))
                elif self._is_contradiction(subject, predicate, obj, prop):
                    contradictions.append((similarity, prop))
        
        # Determine verdict
        if contradictions:
            # Found contradicting ground truth
            best_contradiction = max(contradictions, key=lambda x: x[0])
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.REFUTED,
                confidence=best_contradiction[0],
                contradicting_proposition=best_contradiction[1],
                sources=best_contradiction[1].get("sources", []),
                reason=self._format_contradiction_reason(
                    subject, predicate, obj, best_contradiction[1]
                ),
                sub_claims=self._decompose(subject, predicate, obj) if decompose else []
            )
        
        if matches:
            # Found supporting ground truth
            best_match = max(matches, key=lambda x: x[0])
            return GroundTruthResult(
                original_claim=claim_text,
                status=PropositionStatus.VERIFIED,
                confidence=best_match[0],
                matching_proposition=best_match[1],
                sources=best_match[1].get("sources", []),
                reason=f"Ground truth confirms: {best_match[1]['raw_text']}",
                sub_claims=self._decompose(subject, predicate, obj) if decompose else []
            )
        
        # No matching ground truth
        return GroundTruthResult(
            original_claim=claim_text,
            status=PropositionStatus.UNVERIFIED,
            confidence=0.5,
            reason="No matching ground truth found",
            sub_claims=self._decompose(subject, predicate, obj) if decompose else []
        )
    
    def _extract_triple(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Extract subject-predicate-object from text (simplified)."""
        import re
        
        # Simple patterns (would use NLP in production)
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
        subj_sim = self._similarity(
            self._embed(subject),
            self._embed(prop["subject"])
        )
        pred_sim = self._similarity(
            self._embed(predicate),
            self._embed(prop["predicate"])
        )
        obj_sim = self._similarity(
            self._embed(obj),
            self._embed(prop["object"])
        )
        
        return subj_sim > 0.8 and pred_sim > 0.7 and obj_sim > 0.8
    
    def _is_contradiction(self, subject: str, predicate: str, obj: str, prop: Dict) -> bool:
        """Check if claim contradicts proposition (same predicate+object, different subject)."""
        pred_sim = self._similarity(
            self._embed(predicate),
            self._embed(prop["predicate"])
        )
        obj_sim = self._similarity(
            self._embed(obj),
            self._embed(prop["object"])
        )
        subj_sim = self._similarity(
            self._embed(subject),
            self._embed(prop["subject"])
        )
        
        # Same predicate and object, but different subject = contradiction
        return pred_sim > 0.7 and obj_sim > 0.8 and subj_sim < 0.5
    
    def _format_contradiction_reason(
        self, subject: str, predicate: str, obj: str, prop: Dict
    ) -> str:
        """Format a human-readable contradiction reason."""
        return (
            f"Ground truth shows {prop['subject']} {prop['predicate']} {prop['object']}, "
            f"not {subject}. Source: {prop.get('sources', ['Unknown'])[0] if prop.get('sources') else 'Unknown'}"
        )
    
    def _decompose(self, subject: str, predicate: str, obj: str) -> List[SubClaim]:
        """Decompose claim into verifiable sub-claims."""
        sub_claims = []
        
        # Check if subject is known entity
        subject_known = any(
            e["name"].lower() == subject.lower() or 
            subject.lower() in [a.lower() for a in e.get("aliases", [])]
            for e in self._entities.values()
        )
        sub_claims.append(SubClaim(
            subject=subject,
            predicate="is_a",
            obj="Entity",
            status=PropositionStatus.VERIFIED if subject_known else PropositionStatus.UNVERIFIED,
            confidence=1.0 if subject_known else 0.5
        ))
        
        # Check if predicate is known
        predicate_known = any(
            p["name"].lower() == predicate.lower()
            for p in self._predicates.values()
        )
        sub_claims.append(SubClaim(
            subject=predicate,
            predicate="is_a",
            obj="Predicate",
            status=PropositionStatus.VERIFIED if predicate_known else PropositionStatus.UNVERIFIED,
            confidence=1.0 if predicate_known else 0.5
        ))
        
        # Check if object is known entity
        object_known = any(
            e["name"].lower() == obj.lower() or 
            obj.lower() in [a.lower() for a in e.get("aliases", [])]
            for e in self._entities.values()
        )
        sub_claims.append(SubClaim(
            subject=obj,
            predicate="is_a",
            obj="Entity",
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
# DEMO
# =============================================================================

def demo():
    """Demonstrate KnowShowGo ground truth for hallucination detection."""
    print("=" * 70)
    print("KNOWSHOWGO GROUND TRUTH DEMO")
    print("=" * 70)
    
    # Create ground truth layer
    gt = KSGGroundTruth()
    
    # Add verified facts (ground truth)
    print("\nLoading ground truth...")
    gt.add_verified_fact(
        "Alexander Graham Bell", "invented", "the telephone",
        sources=[{"url": "https://en.wikipedia.org/wiki/Telephone", "trust_score": 0.95}]
    )
    gt.add_verified_fact(
        "Albert Einstein", "discovered", "the theory of relativity",
        sources=[{"url": "https://en.wikipedia.org/wiki/Theory_of_relativity", "trust_score": 0.95}]
    )
    gt.add_verified_fact(
        "Thomas Edison", "invented", "the lightbulb",
        sources=[{"url": "https://en.wikipedia.org/wiki/Incandescent_light_bulb", "trust_score": 0.90}]
    )
    gt.add_verified_fact(
        "Marie Curie", "discovered", "radium",
        sources=[{"url": "https://en.wikipedia.org/wiki/Radium", "trust_score": 0.95}]
    )
    
    print(f"Ground truth stats: {gt.stats()}")
    
    # Test claims
    test_claims = [
        "Bell invented the telephone.",           # Should VERIFY
        "Edison invented the telephone.",         # Should REFUTE (Bell did)
        "Einstein discovered relativity.",        # Should VERIFY
        "Einstein invented the lightbulb.",       # Should REFUTE (Edison did)
        "Nikola Tesla invented the radio.",       # Should be UNVERIFIED (not in GT)
    ]
    
    print("\n" + "=" * 70)
    print("CHECKING CLAIMS AGAINST GROUND TRUTH")
    print("=" * 70)
    
    for claim in test_claims:
        print(f"\n{'─' * 70}")
        print(f"CLAIM: {claim}")
        print(f"{'─' * 70}")
        
        result = gt.check(claim)
        
        print(f"STATUS: {result.status.value}")
        print(f"CONFIDENCE: {result.confidence:.0%}")
        print(f"REASON: {result.reason}")
        
        if result.sub_claims:
            print("\nSUB-CLAIMS:")
            for sc in result.sub_claims:
                status_icon = "✓" if sc.status == PropositionStatus.VERIFIED else "?"
                print(f"  {status_icon} {sc.subject} {sc.predicate} {sc.obj}")
        
        if result.sources:
            print("\nSOURCES:")
            for source in result.sources:
                print(f"  • {source.get('url', 'Unknown')}")
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
    KnowShowGo provides GROUND TRUTH for hallucination detection:
    
    1. PROPOSITIONS = Claims stored as RDF triples
       (subject → predicate → object)
    
    2. PROTOTYPES = Categories defining claim types
       (VerifiedProposition, RefutedProposition)
    
    3. EXEMPLARS = Verified instances that define categories
       ("Bell invented telephone" exemplifies Invention claims)
    
    4. SUB-CLAIMS = Decomposed verification
       (Check subject is Entity, predicate is valid, object is Entity)
    
    5. ASSOCIATIONS = Provenance graph
       (Claim ─derived_from→ Source with trust_score)
    
    6. FUZZY MATCHING = Embeddings for semantic similarity
       ("invented telephone" matches even with different wording)
    
    This creates a SEMANTIC MEMORY that serves as:
    - Ground truth for LLM verification
    - RAG retrieval layer
    - Audit trail for provenance
    - Human-AI aligned knowledge representation
    """)


if __name__ == "__main__":
    demo()
