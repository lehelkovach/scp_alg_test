# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark suite (all algorithms)
python benchmark.py --verbose

# Run specific algorithm
python benchmark.py --algorithm scp

# Export benchmark results to markdown
python benchmark.py --export

# Run unit tests (53 tests)
pytest tests/test_scp.py -v
```

## Repository Structure

```
scp_alg_test/
├── scp.py                   # Core SCP hallucination prover
├── wikidata_verifier.py     # Wikidata API verification
├── hallucination_strategies.py  # LLM-based strategies
├── verified_memory.py       # Verification + caching layer
├── ksg_ground_truth.py      # KnowShowGo ground truth
├── ksg_integration.py       # KnowShowGo REST client
│
├── benchmark.py             # Algorithm benchmark suite
├── tests/
│   └── test_scp.py         # 53 unit tests
│
├── docs/
│   ├── opus.txt            # Design decisions
│   └── ...
│
├── requirements.txt
└── README.md
```

## Benchmark Results

*Generated: 2026-01-14*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Latency | Description |
|-----------|--------|----------|---------|-------------|
| **SCP** | ✅ available | **78%** | 2.4ms | Local KB with semantic embeddings |
| **Wikidata** | ✅ available | 33% | ~200ms | 100M+ facts via SPARQL |
| **VerifiedMemory** | ✅ available | 67% | 0.3ms | Cached KB + fallback |
| **LLM-Judge** | ⚠️ mock | 22% | ~200ms | Requires API key |
| **Self-Consistency** | ⚠️ mock | 22% | ~500ms | Requires API key |
| **KnowShowGo** | ❌ unavailable | N/A | N/A | Requires KSG server |

### Test Set Breakdown (SCP)

| Test Set | Passed | Total | Accuracy | Avg Latency |
|----------|--------|-------|----------|-------------|
| Inventions & Discoveries | 5 | 6 | 83% | 0.7ms |
| Geography & Locations | 6 | 6 | 100% | 0.1ms |
| Creators & Founders | 3 | 6 | 50% | 6.5ms |

### Algorithm Status Legend

- ✅ **available**: Fully functional, ready to use
- ⚠️ **mock_only**: Uses mock implementation (set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for real)
- ❌ **unavailable**: Missing external dependency (see reason in output)

## Algorithms

### 1. SCP (Symbolic Consistency Probing)

Local knowledge base verification using semantic embeddings.

```python
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend

kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([("Bell", "invented", "telephone")])

prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
report = prober.probe("Edison invented the telephone")
print(report.results[0].verdict)  # FAIL
```

**Verdicts:** PASS, SOFT_PASS, FAIL, CONTRADICT, UNKNOWN

### 2. Wikidata Verification

Queries Wikidata's 100M+ facts via SPARQL.

```python
from wikidata_verifier import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Bell invented the telephone")
print(result.status)  # VERIFIED
```

### 3. LLM-as-Judge

Uses an LLM to verify claims (requires API key).

```python
from hallucination_strategies import LLMJudgeStrategy

def my_llm(prompt):
    # Your LLM API call here
    pass

judge = LLMJudgeStrategy(my_llm)
result = judge.check("Edison invented the telephone")
```

### 4. Verified Memory

Cached verification with provenance tracking.

```python
from verified_memory import VerifiedMemory, HallucinationProver

memory = VerifiedMemory("./cache")
prover = HallucinationProver()

status, provenance, claim = prover.prove("Bell invented the telephone")
print(status)  # VERIFIED
```

### 5. KnowShowGo (Future)

Fuzzy ontology knowledge graph. See `docs/knowshowgo_integration_spec.md`.

## Running Tests

```bash
# Unit tests (53 tests)
pytest tests/test_scp.py -v

# Benchmark all algorithms
python benchmark.py

# Benchmark with verbose output
python benchmark.py --verbose

# Export results to markdown
python benchmark.py --export
```

## Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestClaim | 4 | Claim dataclass |
| TestRuleBasedExtractor | 8 | Claim extraction |
| TestHyperKB | 9 | Knowledge base |
| TestEmbeddingBackends | 6 | Similarity matching |
| TestSCPProber | 11 | Core prober |
| TestVerdictScenarios | 4 | Verdict types |
| TestEdgeCases | 5 | Edge cases |
| TestFalseAttribution | 4 | Attribution errors |
| TestCoreference | 2 | Pronoun handling |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Enable real LLM-as-Judge |
| `ANTHROPIC_API_KEY` | Alternative LLM API |
| `KSG_URL` | KnowShowGo server URL |

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- All commits tagged with `[Opus4.5]`

## License

MIT License
