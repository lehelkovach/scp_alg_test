# KnowShowGo Integration Specification

**For:** github.com/lehelkovach/knowshowgo  
**From:** Hallucination Detection System (scp_alg_test)  
**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Date:** January 14, 2026

---

## Overview

This document outlines the required API endpoints, data structures, and features that KnowShowGo needs to implement to serve as the ground truth backend for the hallucination detection system.

---

## 1. Required REST API Endpoints

### 1.1 Prototype Management

```
POST   /prototypes                 Create a new prototype (schema)
GET    /prototypes                 List all prototypes
GET    /prototypes/:uuid           Get prototype by UUID
PUT    /prototypes/:uuid           Update prototype
```

**Request Body (POST /prototypes):**
```json
{
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
  },
  "embedding": [0.1, 0.2, ...]  // 64-1536 dim vector
}
```

**Response:**
```json
{
  "uuid": "proto:abc123",
  "name": "Claim",
  "created_at": 1705276800
}
```

### 1.2 Concept Management

```
POST   /concepts                   Create a new concept
GET    /concepts/:uuid             Get concept by UUID
PUT    /concepts/:uuid             Update concept (creates new version)
GET    /concepts/:uuid/history     Get version history
DELETE /concepts/:uuid             Soft delete (mark inactive)
```

**Request Body (POST /concepts):**
```json
{
  "prototypeUuid": "proto:abc123",
  "jsonObj": {
    "subject": "Einstein",
    "predicate": "discovered",
    "object": "relativity"
  },
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "source": "manual",
    "confidence": 0.95
  }
}
```

**Response:**
```json
{
  "uuid": "concept:xyz789",
  "prototypeUuid": "proto:abc123",
  "version": 1,
  "created_at": 1705276800
}
```

### 1.3 Semantic Search

```
POST   /concepts/search            Search concepts by embedding similarity
```

**Request Body:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "topK": 10,
  "similarityThreshold": 0.7,
  "filters": {
    "prototypeUuid": "proto:abc123",  // optional
    "labels": ["verified"]             // optional
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "uuid": "concept:xyz789",
      "similarity": 0.92,
      "props": {
        "subject": "Bell",
        "predicate": "invented",
        "object": "telephone"
      }
    }
  ],
  "total": 1,
  "search_time_ms": 12
}
```

### 1.4 Association Management

```
POST   /associations               Create association between concepts
GET    /concepts/:uuid/associations  Get associations for a concept
DELETE /associations/:uuid         Remove association
```

**Request Body (POST /associations):**
```json
{
  "fromUuid": "concept:abc",
  "toUuid": "concept:xyz",
  "type": "verified_by",
  "weight": 0.95,
  "metadata": {
    "method": "knowledge_base",
    "timestamp": 1705276800
  }
}
```

**Required Association Types:**
| Type | Description | Weight Meaning |
|------|-------------|----------------|
| `supports` | Evidence supports claim | Support strength |
| `contradicts` | Evidence contradicts claim | Contradiction strength |
| `derived_from` | Claim derived from source | Derivation confidence |
| `verified_by` | Claim verified by method | Verification confidence |
| `same_as` | Entity equivalence | Identity confidence |
| `related_to` | General relation | Relation strength |

---

## 2. Required Prototypes

The hallucination detection system needs these prototypes pre-defined or auto-created:

### 2.1 Claim Prototype
```json
{
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
}
```

### 2.2 VerificationResult Prototype
```json
{
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
}
```

### 2.3 Source Prototype
```json
{
  "name": "Source",
  "description": "A source of information",
  "context": "provenance",
  "labels": ["source", "reference", "origin"],
  "schema": {
    "type": "enum:knowledge_base,llm,wikidata,document,human",
    "identifier": "string",
    "trust_score": "number"
  }
}
```

### 2.4 Entity Prototype
```json
{
  "name": "Entity",
  "description": "A named entity (person, place, thing)",
  "context": "knowledge",
  "labels": ["entity", "noun", "thing"],
  "schema": {
    "name": "string",
    "type": "string",
    "aliases": "array",
    "wikidata_id": "string"
  }
}
```

---

## 3. Required Features

### 3.1 Embedding Support

**Requirements:**
- Accept embeddings as float arrays (64-1536 dimensions)
- Store embeddings efficiently (consider quantization for scale)
- Support cosine similarity search with configurable threshold
- Return similarity scores in search results

**Embedding Interface:**
```typescript
interface EmbeddingConfig {
  dimensions: number;        // 64-1536
  similarity_metric: "cosine" | "euclidean" | "dot";
  index_type: "flat" | "hnsw" | "ivf";  // for scale
}
```

### 3.2 Version History

**Requirements:**
- Every concept update creates a new version
- Versions are immutable once created
- History endpoint returns all versions in chronological order
- Support rollback to previous version

**Version Structure:**
```json
{
  "uuid": "concept:xyz",
  "version": 3,
  "created_at": 1705276800,
  "created_by": "system",
  "changes": {
    "confidence": {"from": 0.8, "to": 0.95}
  },
  "previous_version": 2
}
```

### 3.3 Weighted Associations

**Requirements:**
- All associations have weight (0.0-1.0)
- Weights represent confidence/strength
- Support querying by weight threshold
- Aggregate weights for multi-hop queries

### 3.4 Provenance Chain

**Requirements:**
- Every claim links to verification result via `verified_by`
- Every claim links to source via `derived_from`
- Support traversing provenance chain
- Export audit trail as JSON

**Example Provenance Query:**
```
GET /concepts/:uuid/provenance?depth=3
```

**Response:**
```json
{
  "claim": { "uuid": "concept:abc", "props": {...} },
  "verified_by": { "uuid": "concept:xyz", "weight": 0.95 },
  "derived_from": { "uuid": "concept:src", "weight": 0.99 },
  "chain_confidence": 0.94  // product of weights
}
```

---

## 4. Python Client Requirements

The hallucination detection system expects this Python interface:

```python
class KSGClient:
    def __init__(self, base_url: str, embed_fn: Callable = None):
        """Initialize client with KSG server URL and embedding function."""
        
    def ensure_prototypes(self) -> None:
        """Create verification prototypes if they don't exist."""
        
    def create_concept(
        self, 
        prototype_name: str, 
        props: dict, 
        text_for_embedding: str = None
    ) -> str:
        """Create concept, returns UUID."""
        
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[dict]:
        """Search by semantic similarity."""
        
    def create_association(
        self,
        from_uuid: str,
        to_uuid: str,
        association_type: str,
        weight: float = 1.0,
        metadata: dict = None
    ) -> str:
        """Create weighted association."""
        
    def get_associations(
        self, 
        concept_uuid: str, 
        association_type: str = None
    ) -> List[dict]:
        """Get associations for concept."""
        
    def get_history(self, concept_uuid: str) -> List[dict]:
        """Get version history."""
```

---

## 5. Data Flow Integration

### 5.1 Claim Verification Flow

```
1. LLM generates text
2. Extract claims (subject, predicate, object)
3. Generate embedding for claim text
4. Search KSG for similar claims
5. If match found:
   - Check if supports or contradicts
   - Return verdict with provenance
6. If no match:
   - Mark as unverifiable OR
   - Query external source (Wikidata)
   - Store result in KSG for future
```

### 5.2 RAG Retrieval Flow

```
1. User query arrives
2. Generate embedding for query
3. Search KSG with embedding
4. Filter by verification status
5. Return top-k verified claims
6. Format as context for LLM
```

---

## 6. Suggested Implementation Priority

### Phase 1: Core (Required for Integration)
1. ✅ Prototype CRUD
2. ✅ Concept CRUD
3. ✅ Embedding storage and search
4. ✅ Association CRUD

### Phase 2: Provenance (Required for Audit)
5. Version history
6. Provenance chain queries
7. Audit trail export

### Phase 3: Scale (For Production)
8. HNSW/IVF indexing for embeddings
9. Batch operations
10. Caching layer

---

## 7. Testing Checklist

When KnowShowGo implements these features, test with:

```python
# Test script for KSG integration
from lib.ksg_integration import KSGClient, KSG_PROTOTYPES

def test_ksg_integration():
    ksg = KSGClient("http://localhost:3000")
    
    # 1. Create prototypes
    ksg.ensure_prototypes()
    
    # 2. Create a claim
    claim_uuid = ksg.create_concept("Claim", {
        "subject": "Bell",
        "predicate": "invented", 
        "object": "telephone"
    })
    assert claim_uuid.startswith("concept:")
    
    # 3. Search for claim
    results = ksg.search("who invented the telephone")
    assert len(results) > 0
    assert results[0]["similarity"] > 0.7
    
    # 4. Create association
    source_uuid = ksg.create_concept("Source", {
        "type": "knowledge_base",
        "identifier": "scp_kb_v1"
    })
    
    assoc_uuid = ksg.create_association(
        claim_uuid, source_uuid, "derived_from", 0.95
    )
    assert assoc_uuid.startswith("assoc:")
    
    # 5. Get provenance
    associations = ksg.get_associations(claim_uuid)
    assert len(associations) > 0
    
    print("✓ All KSG integration tests passed")

if __name__ == "__main__":
    test_ksg_integration()
```

---

## 8. Questions for KnowShowGo Team

1. **Embedding dimensions:** What dimension should we standardize on? (Suggest: 384 for MiniLM, 1536 for OpenAI)

2. **Batch operations:** Will there be bulk create/search endpoints?

3. **Authentication:** What auth scheme for production? (API key, JWT, OAuth?)

4. **Rate limits:** Expected limits for search operations?

5. **Webhooks:** Will there be webhooks for concept updates?

---

*This specification is based on the hallucination detection system's requirements.*
*Please update lib/ksg_integration.py when API changes.*
