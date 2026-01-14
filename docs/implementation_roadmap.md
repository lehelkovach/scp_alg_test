# Implementation Roadmap - Hallucination Detection Coverage Analysis

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Date:** January 14, 2026

---

## Current Coverage Analysis

### Benchmark Results Summary

| Algorithm | Score | True Facts | False Attr | Contradiction | Fabrication |
|-----------|-------|------------|------------|---------------|-------------|
| **SCP** | 85/100 | 78% | 67% | 100% | 100% |
| **VerifiedMemory** | 70/100 | 78% | 0% | 100% | 100% |
| **Wikidata** | 38/100 | 44% | 100% | 0% | 0% |
| **LLM-Judge** | 23/100 | 22% | 33% | 33% | 0% |
| **Self-Consistency** | 23/100 | 22% | 33% | 33% | 0% |
| **KnowShowGo** | 0/100 | N/A | N/A | N/A | N/A |

---

## Gap Analysis

### What's Working Well ✅

1. **SCP Contradiction Detection (100%)**
   - Successfully detects "Einstein born in France" vs "born in Germany"
   - Works via semantic similarity matching

2. **SCP Fabrication Detection (100%)**
   - Catches completely made-up facts like "Curie invented smartphone"
   - Returns FAIL for unknown claims

3. **Wikidata False Attribution (100%)**
   - Excellent for "who invented X" queries
   - Uses structured SPARQL with correct subject matching

4. **VerifiedMemory Contradiction + Fabrication (100%)**
   - Inherits from SCP prover

### What's NOT Working ❌

1. **SCP False Attribution (67%)**
   - **Problem**: "Tesla discovered radium" soft-matches "Curie discovered radium"
   - **Cause**: Semantic similarity on predicate+object, ignores subject mismatch
   - **Fix Needed**: Subject-aware matching

2. **VerifiedMemory False Attribution (0%)**
   - **Problem**: All false attributions return VERIFIED
   - **Cause**: Matches on predicate+object, wrong subject still passes
   - **Fix Needed**: Same as SCP - subject comparison

3. **Wikidata Contradiction/Fabrication (0%)**
   - **Problem**: Returns UNVERIFIABLE for non-invention claims
   - **Cause**: Limited predicate support (only invented/discovered)
   - **Fix Needed**: Add more Wikidata predicates

4. **LLM-Judge/Self-Consistency (22-33%)**
   - **Problem**: Mock implementation returns UNKNOWN for most
   - **Cause**: No real LLM connected
   - **Fix Needed**: Add real LLM API integration

5. **KnowShowGo (N/A)**
   - **Problem**: Server not running
   - **Cause**: External dependency not implemented
   - **Fix Needed**: Build/deploy KnowShowGo

---

## Implementation Priority (By Development Time)

### 🟢 Quick Wins (1-2 hours each)

#### 1. Fix SCP False Attribution Detection
**Current:** 67% → **Target:** 95%

**Problem:**
```python
# "Tesla discovered radium" soft-matches "Curie discovered radium"
# because predicate+object are similar
```

**Solution:**
```python
# In scp.py, update find_similar_facts to check subject
def find_similar_facts_with_subject_check(self, claim, threshold=0.7):
    matches = self.find_similar_facts(claim, threshold)
    for score, fact, rel_id in matches:
        fact_subject, fact_pred, fact_obj = fact
        # If predicate+object match but subject differs = FALSE ATTRIBUTION
        if (self._normalize(claim.predicate) == self._normalize(fact_pred) and
            self._normalize(claim.obj) == self._normalize(fact_obj) and
            self._normalize(claim.subject) != self._normalize(fact_subject)):
            return "FALSE_ATTRIBUTION", fact_subject  # Return correct subject
    return matches
```

**Effort:** 1-2 hours
**Impact:** +28% false attribution detection

---

#### 2. Expand Wikidata Predicates
**Current:** 38% → **Target:** 70%

**Problem:**
```
"Einstein was born in Germany" → UNVERIFIABLE (no born_in predicate)
"Eiffel Tower in Paris" → UNVERIFIABLE (no located_in predicate)
```

**Solution:**
```python
# In wikidata_verifier.py, add more predicates:
PREDICATE_MAPPING = {
    "invented": "P61",      # ✅ existing
    "discovered": "P61",    # ✅ existing
    "born_in": "P19",       # 🆕 place of birth
    "died_in": "P20",       # 🆕 place of death
    "located_in": "P131",   # 🆕 located in admin territory
    "capital_of": "P36",    # 🆕 capital of
    "created_by": "P170",   # 🆕 creator
    "founded_by": "P112",   # 🆕 founded by
}
```

**Effort:** 1-2 hours
**Impact:** +32% overall, enables contradiction detection

---

#### 3. Fix VerifiedMemory Subject Matching
**Current:** 0% false attr → **Target:** 80%

**Problem:**
Same as SCP - inherits the soft-match issue.

**Solution:**
Update `HallucinationProver.prove()` to use subject-aware matching.

**Effort:** 30 minutes (after SCP fix)
**Impact:** +80% false attribution detection

---

### 🟡 Medium Effort (4-8 hours each)

#### 4. Add Real LLM Integration
**Current:** 22% (mock) → **Target:** 85%+ (real)

**Problem:**
```python
def mock_llm(prompt):
    # Returns hardcoded responses
    if "Edison" in prompt and "telephone" in prompt:
        return "FALSE"
    return "UNKNOWN"  # Most things return unknown
```

**Solution:**
```python
# Add OpenAI integration
import openai

def real_llm_judge(claim: str) -> tuple:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are a fact checker. Return TRUE, FALSE, or UNKNOWN."
        }, {
            "role": "user", 
            "content": f"Is this claim true? {claim}"
        }]
    )
    answer = response.choices[0].message.content
    # Parse response...
```

**Effort:** 4-6 hours
**Impact:** LLM-Judge jumps from 22% to 85%+
**Cost:** ~$0.15/1000 claims (gpt-4o-mini)

---

#### 5. Add Hybrid Cascade (Best of All)
**Current:** N/A → **Target:** 90%+

**Architecture:**
```
Claim → SCP (10ms) → if UNKNOWN → Wikidata (200ms) → if UNKNOWN → LLM (200ms)
```

**Benefits:**
- 70% claims resolved by SCP (~10ms, $0)
- 20% by Wikidata (~200ms, $0)
- 10% by LLM (~200ms, $0.0001)
- Average: ~50ms, ~$0.00001/claim

**Effort:** 4-8 hours
**Impact:** Combines best of all approaches

---

### 🔴 Larger Projects (1-4 weeks)

#### 6. KnowShowGo Integration
**Current:** 0% (unavailable) → **Target:** 95%

**Requirements:**
1. Deploy KnowShowGo server
2. Implement REST API endpoints (see `docs/knowshowgo_integration_spec.md`)
3. Populate initial KB
4. Connect Python client

**Effort:** 2-4 weeks
**Impact:** Full cognitive architecture with provenance

---

#### 7. Sentence-Transformer Embeddings
**Current:** Hash-based → **Target:** Real semantic

**Problem:**
```python
# Hash-based embeddings have limited semantic understanding
backend = HashingEmbeddingBackend(dim=512)  # ~70% accuracy
```

**Solution:**
```python
# Real embeddings have much better semantic matching
backend = SentenceTransformerBackend("all-MiniLM-L6-v2")  # ~90% accuracy
```

**Effort:** 30 minutes (just pip install)
**Impact:** +10-15% accuracy on semantic matching

---

## Recommended Implementation Order

| Priority | Task | Time | Impact | Dependencies |
|----------|------|------|--------|--------------|
| 1 | Fix SCP subject matching | 1-2h | +28% false attr | None |
| 2 | Expand Wikidata predicates | 1-2h | +32% overall | None |
| 3 | Fix VerifiedMemory | 30min | +80% false attr | #1 |
| 4 | Install sentence-transformers | 30min | +15% accuracy | None |
| 5 | Add real LLM integration | 4-6h | +63% LLM | API key |
| 6 | Build hybrid cascade | 4-8h | 90%+ overall | #1-5 |
| 7 | KnowShowGo integration | 2-4w | Full system | KSG server |

---

## Expected Results After Implementation

### Phase 1 (Day 1): Quick Fixes
```
After fixes #1-4:
SCP:            85/100 → 95/100
Wikidata:       38/100 → 70/100
VerifiedMemory: 70/100 → 90/100
```

### Phase 2 (Week 1): LLM + Hybrid
```
After #5-6:
LLM-Judge:      23/100 → 85/100
Hybrid:         N/A    → 95/100
```

### Phase 3 (Month 1): Full System
```
After #7:
KnowShowGo:     0/100  → 95/100
Full coverage across all hallucination types
```

---

## Cost Analysis

| Approach | Latency | Cost/1000 Claims | Accuracy |
|----------|---------|------------------|----------|
| SCP only | 10ms | $0 | 85% |
| Wikidata only | 200ms | $0 | 70% |
| LLM only | 200ms | $0.15 | 85% |
| **Hybrid (recommended)** | **50ms avg** | **$0.02** | **95%** |

---

## Next Steps

1. **Today:** Implement quick fixes #1-4
2. **This Week:** Add real LLM integration #5
3. **Next Week:** Build hybrid cascade #6
4. **This Month:** KnowShowGo integration #7

---

*Generated by Claude Opus 4.5 based on benchmark analysis*
