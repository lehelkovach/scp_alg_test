## Benchmark Results

*Generated: 2026-01-14 17:05:44*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Latency | Description |
|-----------|--------|----------|---------|-------------|
| SCP | available | 78% | 2.4ms | Local KB verification using semantic emb... |
| Wikidata | available | 33% | 0.0ms | Wikidata SPARQL queries (~200ms, free pu... |
| LLM-Judge | mock_only | 22% | 0.0ms | LLM verification (~200ms, 1 API call per... |
| Self-Consistency | mock_only | 22% | 0.0ms | Multiple LLM samples (~500ms, 3-5 API ca... |
| KnowShowGo | unavailable | N/A | N/A | Fuzzy ontology graph (~10ms, requires KS... |
| VerifiedMemory | available | 67% | 0.3ms | Cached KB + LLM fallback (~50ms avg)... |

### Test Set Breakdown

#### SCP

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 5 | 6 | 83% | 0.7ms |
| Geography & Locations | 6 | 6 | 100% | 0.1ms |
| Creators & Founders | 3 | 6 | 50% | 6.5ms |

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
