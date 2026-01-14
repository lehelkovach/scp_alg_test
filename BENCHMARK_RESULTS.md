## Benchmark Results

*Generated: 2026-01-14 17:20:16*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Coverage | Score | Latency |
|-----------|--------|----------|----------|-------|---------|
| SCP | available | 83% | GOOD | 85/100 | 2.4ms |
| Wikidata | available | 39% | WEAK | 38/100 | 0.0ms |
| LLM-Judge | mock_only | 22% | MINIMAL | 23/100 | 0.0ms |
| Self-Consistency | mock_only | 22% | MINIMAL | 23/100 | 0.0ms |
| KnowShowGo | unavailable | N/A | MINIMAL | 0/100 | N/A |
| VerifiedMemory | available | 72% | GOOD | 70/100 | 0.3ms |

### Coverage Legend

| Level | Accuracy | Description |
|-------|----------|-------------|
| EXCELLENT | 90%+ | Comprehensive detection across all types |
| GOOD | 70-89% | Reliable for most claim types |
| MODERATE | 50-69% | Partial detection, some blind spots |
| WEAK | 30-49% | Limited reliability |
| MINIMAL | <30% | Not recommended for production |

### Detection Capabilities

| Algorithm | False Attribution | Contradictions | Extrinsic | Intrinsic |
|-----------|-------------------|----------------|-----------|-----------|
| SCP | ✓ | ✓ | ✓ | ✓ |
| Wikidata | ✓ | ✗ | ✗ | ✗ |
| LLM-Judge | ✗ | ✗ | ✗ | ✗ |
| Self-Consistency | ✗ | ✗ | ✗ | ✗ |
| KnowShowGo | ✗ | ✗ | ✗ | ✗ |
| VerifiedMemory | ✗ | ✓ | ✓ | ✓ |

### Strengths & Weaknesses

#### SCP

**Strengths:**
- Very fast (~10ms)
- Zero API calls
- Deterministic results
- Proof subgraph for auditing
- Contradiction detection

**Weaknesses:**
- Limited to facts in KB
- Requires KB maintenance
- Semantic similarity can soft-match wrong subjects

#### Wikidata

**Strengths:**
- 100M+ facts available
- No training required
- Structured provenance
- Free public API
- High accuracy for supported predicates

**Weaknesses:**
- ~200ms latency (network)
- Limited predicate support
- Cannot verify all claim types
- Rate limited

#### LLM-Judge

**Strengths:**
- Broad knowledge coverage
- Can verify complex claims
- Natural language understanding
- No KB maintenance needed

**Weaknesses:**
- LLM can also hallucinate
- Requires API key and costs $
- ~200ms latency per call
- Non-deterministic

#### Self-Consistency

**Strengths:**
- More robust than single LLM call
- Catches LLM inconsistencies
- Better accuracy through voting

**Weaknesses:**
- 3-5x more expensive
- Higher latency
- Still relies on LLM knowledge

#### KnowShowGo

**Strengths:**
- Fuzzy matching for nuanced claims
- Version history for auditing
- Weighted associations
- Community governance
- Cognitive architecture

**Weaknesses:**
- Requires KnowShowGo server
- Needs initial KB population
- External dependency

#### VerifiedMemory

**Strengths:**
- Fast cache lookups
- Provenance tracking
- Persistent storage
- LLM fallback option

**Weaknesses:**
- Cache needs warming
- Depends on underlying prover

### Test Set Breakdown

#### SCP

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| False Attribution Detection | 5 | 6 | 83% | 0.7ms |
| Contradiction Detection | 5 | 6 | 83% | 2.2ms |
| Fabrication Detection | 5 | 6 | 83% | 4.3ms |

#### Wikidata

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| False Attribution Detection | 6 | 6 | 100% | 0.0ms |
| Contradiction Detection | 0 | 6 | 0% | 0.0ms |
| Fabrication Detection | 1 | 6 | 17% | 0.0ms |

#### LLM-Judge

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| False Attribution Detection | 2 | 6 | 33% | 0.0ms |
| Contradiction Detection | 2 | 6 | 33% | 0.0ms |
| Fabrication Detection | 0 | 6 | 0% | 0.0ms |

#### Self-Consistency

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| False Attribution Detection | 2 | 6 | 33% | 0.0ms |
| Contradiction Detection | 2 | 6 | 33% | 0.0ms |
| Fabrication Detection | 0 | 6 | 0% | 0.0ms |

#### VerifiedMemory

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| False Attribution Detection | 3 | 6 | 50% | 0.2ms |
| Contradiction Detection | 5 | 6 | 83% | 0.2ms |
| Fabrication Detection | 5 | 6 | 83% | 0.4ms |

### Implementation Guidance

When tests fail, here's what to implement:

#### false_attr

**Problem:** Detect wrong subject for correct predicate/object
**Example:** "Edison invented telephone" should fail (Bell did)

**Solution:**
```python
IMPLEMENTATION NEEDED: Subject-Aware Matching
─────────────────────────────────────────────
In scp.py, update find_similar_facts() to compare subjects:

    def find_similar_facts(self, claim, threshold=0.7):
        matches = super().find_similar_facts(claim, threshold)
        for score, (subj, pred, obj), rel_id in matches:
            # If predicate+object match but subject differs = FALSE ATTRIBUTION
            if (self._normalize(claim.predicate) == self._normalize(pred) and
                self._normalize(claim.obj) == self._normalize(obj) and
                self._normalize(claim.subject) != self._normalize(subj)):
                return [(0.0, (subj, pred, obj), rel_id)]  # Return as contradiction
        return matches

Estimated time: 1-2 hours
```

#### contradiction

**Problem:** Detect claims that conflict with known facts
**Example:** "Einstein born in France" should fail (born in Germany)

**Solution:**
```python
IMPLEMENTATION NEEDED: Expand Predicate Support
───────────────────────────────────────────────
For Wikidata (wikidata_verifier.py), add more predicates:

    PREDICATE_MAPPING = {
        "invented": "P61",
        "discovered": "P61", 
        "born_in": "P19",        # ADD: place of birth
        "died_in": "P20",        # ADD: place of death
        "located_in": "P131",    # ADD: located in territory
        "capital_of": "P36",     # ADD: capital
    }

For SCP, ensure KB has the facts:
    kb.add_fact("Einstein", "born_in", "Germany")

Estimated time: 1-2 hours
```

#### fabrication

**Problem:** Detect completely made-up facts
**Example:** "Curie invented smartphone" should fail (never happened)

**Solution:**
```python
IMPLEMENTATION NEEDED: Unknown Fact Detection
─────────────────────────────────────────────
Should return FAIL/REFUTED for claims with no KB match.

For SCP: Already implemented (returns FAIL for no match)
For Wikidata: Add fallback when SPARQL returns empty:

    if not results:
        return WikidataResult(
            status=VerificationStatus.REFUTED,
            reason="No evidence found in Wikidata"
        )

Estimated time: 30 minutes
```

### Algorithm-Specific Requirements

#### SCP

```
FIX: Update SCPProber._probe_claim() in scp.py:
────────────────────────────────────────────────
Add subject comparison after finding semantic match:

    if best_match and best_score > self.soft_threshold:
        match_subj, match_pred, match_obj = best_match
        # Check if this is actually a false attribution
        if (claim.predicate == match_pred and claim.obj == match_obj 
            and claim.subject != match_subj):
            return ProbeResult(
                claim=claim,
                verdict=Verdict.CONTRADICT,
                score=0.0,
                reason=f"False attribution: {match_subj} {match_pred} {match_obj}, not {claim.subject}"
            )
```

#### Wikidata

```
FIX: Add predicates to wikidata_verifier.py:
─────────────────────────────────────────────
PREDICATE_MAPPING = {
    # Existing
    "invented": "P61",
    "discovered": "P61",
    # Add these:
    "born_in": "P19",
    "located_in": "P131", 
    "capital_of": "P36",
    "created_by": "P170",
    "founded_by": "P112",
}

Then add SPARQL templates for each.
```

#### LLM-Judge

```
FIX: Add real LLM in hallucination_strategies.py:
─────────────────────────────────────────────────
1. Set environment variable:
   export OPENAI_API_KEY=sk-...

2. Replace mock_llm with:
   
   import openai
   
   def real_llm(prompt: str) -> str:
       response = openai.ChatCompletion.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": prompt}]
       )
       return response.choices[0].message.content

3. Update LLMJudgeStrategy to use real_llm
```

#### KnowShowGo

```
FIX: Deploy KnowShowGo server:
──────────────────────────────
1. Clone: git clone https://github.com/lehelkovach/knowshowgo
2. Install: npm install
3. Run: npm start
4. Set: export KSG_URL=http://localhost:3000

See: docs/knowshowgo_integration_spec.md
```
