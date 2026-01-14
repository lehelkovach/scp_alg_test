# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark suite
python benchmark.py

# Run with verbose output
python benchmark.py --verbose

# Export results to markdown
python benchmark.py --export

# Run unit tests (53 tests)
pytest tests/test_scp.py -v
```

## Benchmark Results

*Generated: 2026-01-14*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Coverage | Score | Latency |
|-----------|--------|----------|----------|-------|---------|
| **SCP** | ✅ available | **78%** | GOOD | 84/100 | 2.4ms |
| **VerifiedMemory** | ✅ available | 67% | MODERATE | 77/100 | 0.3ms |
| **Wikidata** | ✅ available | 33% | WEAK | 46/100 | ~200ms |
| **LLM-Judge** | ⚠️ mock | 22% | MINIMAL | 46/100 | ~200ms |
| **Self-Consistency** | ⚠️ mock | 22% | MINIMAL | 46/100 | ~500ms |
| **KnowShowGo** | ❌ unavailable | N/A | MINIMAL | 30/100 | N/A |

### Coverage Levels

| Level | Accuracy | Description |
|-------|----------|-------------|
| **EXCELLENT** | 90%+ | Comprehensive detection across all hallucination types |
| **GOOD** | 70-89% | Reliable detection for most claim types |
| **MODERATE** | 50-69% | Partial detection, some blind spots |
| **WEAK** | 30-49% | Limited reliability, use with caution |
| **MINIMAL** | <30% | Not recommended for production use |

### Detection Capabilities

| Algorithm | False Attribution | Contradictions | Extrinsic | Intrinsic |
|-----------|-------------------|----------------|-----------|-----------|
| SCP | ✓ | ✓ | ✓ | ✓ |
| Wikidata | ✓ | ✓ | ✗ | ✓ |
| LLM-Judge | ✓ | ✓ | ✓ | ✓ |
| Self-Consistency | ✓ | ✓ | ✓ | ✓ |
| KnowShowGo | ✓ | ✓ | ✓ | ✓ |
| VerifiedMemory | ✓ | ✓ | ✓ | ✓ |

**Hallucination Types:**
- **False Attribution**: Wrong subject (e.g., "Edison invented the telephone")
- **Contradictions**: Conflicts with known facts (e.g., "Einstein born in France")
- **Extrinsic**: Added information not in source
- **Intrinsic**: Modified facts from source

### Test Set Breakdown (SCP - Best Performer)

| Test Set | Passed | Total | Accuracy |
|----------|--------|-------|----------|
| Inventions & Discoveries | 5 | 6 | 83% |
| Geography & Locations | 6 | 6 | 100% |
| Creators & Founders | 3 | 6 | 50% |

## Repository Structure

```
scp_alg_test/
├── scp.py                   # Core SCP prover (GOOD coverage)
├── wikidata_verifier.py     # Wikidata API (WEAK coverage)
├── hallucination_strategies.py  # LLM strategies (MINIMAL when mocked)
├── verified_memory.py       # Caching layer (MODERATE coverage)
├── ksg_ground_truth.py      # KnowShowGo (unavailable)
├── ksg_integration.py       # KnowShowGo client
│
├── benchmark.py             # Benchmark suite with coverage metrics
├── tests/
│   └── test_scp.py         # 53 unit tests
│
├── docs/
│   └── ...
│
└── README.md
```

## Algorithms

### 1. SCP (Symbolic Consistency Probing) - GOOD Coverage

Local knowledge base verification using semantic embeddings.

**Strengths:**
- Very fast (~10ms)
- Zero API calls
- Deterministic results
- Proof subgraph for auditing

**Weaknesses:**
- Limited to facts in KB
- Requires KB maintenance

```python
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend

kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([("Bell", "invented", "telephone")])

prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
report = prober.probe("Edison invented the telephone")
print(report.results[0].verdict)  # FAIL
```

### 2. Wikidata - WEAK Coverage

Queries Wikidata's 100M+ facts via SPARQL.

**Strengths:**
- 100M+ facts available
- No training required
- High accuracy for supported predicates

**Weaknesses:**
- Limited predicate support
- ~200ms latency

```python
from wikidata_verifier import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Bell invented the telephone")
```

### 3. Verified Memory - MODERATE Coverage

Cached verification with provenance tracking.

**Strengths:**
- Fast cache lookups
- Provenance tracking
- Persistent storage

```python
from verified_memory import VerifiedMemory, HallucinationProver

prover = HallucinationProver()
status, provenance, claim = prover.prove("Bell invented the telephone")
```

## Running Benchmarks

```bash
# Full benchmark with all algorithms
python benchmark.py

# Verbose output (shows each claim)
python benchmark.py --verbose

# Single algorithm
python benchmark.py --algorithm scp

# Export to markdown
python benchmark.py --export
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Enable real LLM-as-Judge (improves coverage) |
| `ANTHROPIC_API_KEY` | Alternative LLM API |
| `KSG_URL` | KnowShowGo server URL |

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)

## License

MIT License
