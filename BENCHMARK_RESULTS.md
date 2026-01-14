## Benchmark Results

*Generated: 2026-01-14 17:10:34*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Coverage | Score | Latency |
|-----------|--------|----------|----------|-------|---------|
| SCP | available | 78% | GOOD | 84/100 | 2.4ms |
| Wikidata | available | 33% | WEAK | 46/100 | 0.0ms |
| LLM-Judge | mock_only | 22% | MINIMAL | 46/100 | 0.0ms |
| Self-Consistency | mock_only | 22% | MINIMAL | 46/100 | 0.0ms |
| KnowShowGo | unavailable | N/A | MINIMAL | 30/100 | N/A |
| VerifiedMemory | available | 67% | MODERATE | 77/100 | 0.3ms |

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
| Wikidata | ✓ | ✓ | ✗ | ✓ |
| LLM-Judge | ✓ | ✓ | ✓ | ✓ |
| Self-Consistency | ✓ | ✓ | ✓ | ✓ |
| KnowShowGo | ✓ | ✓ | ✓ | ✓ |
| VerifiedMemory | ✓ | ✓ | ✓ | ✓ |

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
| Inventions & Discoveries | 5 | 6 | 83% | 0.7ms |
| Geography & Locations | 6 | 6 | 100% | 0.1ms |
| Creators & Founders | 3 | 6 | 50% | 6.6ms |

#### Wikidata

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 6 | 6 | 100% | 0.0ms |
| Geography & Locations | 0 | 6 | 0% | 0.0ms |
| Creators & Founders | 0 | 6 | 0% | 0.0ms |

#### LLM-Judge

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 2 | 6 | 33% | 0.0ms |
| Geography & Locations | 2 | 6 | 33% | 0.0ms |
| Creators & Founders | 0 | 6 | 0% | 0.0ms |

#### Self-Consistency

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 2 | 6 | 33% | 0.0ms |
| Geography & Locations | 2 | 6 | 33% | 0.0ms |
| Creators & Founders | 0 | 6 | 0% | 0.0ms |

#### VerifiedMemory

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 3 | 6 | 50% | 0.2ms |
| Geography & Locations | 5 | 6 | 83% | 0.2ms |
| Creators & Founders | 4 | 6 | 67% | 0.5ms |
