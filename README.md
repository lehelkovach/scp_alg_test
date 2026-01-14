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

# Run with verbose output (shows each test)
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
| **SCP** | ✅ available | **83%** | GOOD | 85/100 | 2.9ms |
| **VerifiedMemory** | ✅ available | 72% | GOOD | 70/100 | 0.3ms |
| **Wikidata** | ✅ available | 39% | WEAK | 38/100 | ~200ms |
| **LLM-Judge** | ⚠️ mock | 22% | MINIMAL | 23/100 | ~200ms |
| **Self-Consistency** | ⚠️ mock | 22% | MINIMAL | 23/100 | ~500ms |
| **KnowShowGo** | ❌ unavailable | N/A | N/A | 0/100 | N/A |

### Detection Coverage (from actual tests)

Each algorithm is tested against specific hallucination types:

**SCP (Best Overall):**
```
true_fact       ███████░░░   78% (7/9)   - Correctly verifies true claims
false_attr      ██████░░░░   67% (2/3)   - Detects wrong attribution
contradiction   ██████████  100% (3/3)   - Detects contradictions
fabrication     ██████████  100% (3/3)   - Detects made-up facts
```

**VerifiedMemory:**
```
true_fact       █████████░   89% (8/9)
false_attr      ░░░░░░░░░░    0% (0/3)
contradiction   ██████████  100% (3/3)
fabrication     ██████████  100% (3/3)
```

**Wikidata:**
```
true_fact       ████░░░░░░   44% (4/9)
false_attr      ██████████  100% (3/3)   - Strong for inventions
contradiction   ░░░░░░░░░░    0% (0/3)   - Limited predicate support
fabrication     ░░░░░░░░░░    0% (0/3)
```

### Hallucination Types Tested

| Type | Description | Example |
|------|-------------|---------|
| **True Fact** | Verify correct claims pass | "Bell invented telephone" ✓ |
| **False Attribution** | Wrong subject | "Edison invented telephone" ✗ |
| **Contradiction** | Conflicts with known fact | "Einstein born in France" ✗ |
| **Fabrication** | Completely made up | "Curie invented smartphone" ✗ |

### Test Set Breakdown

| Test Set | SCP | VerifiedMemory | Wikidata |
|----------|-----|----------------|----------|
| False Attribution | 83% | 50% | 100% |
| Contradiction | 83% | 83% | 0% |
| Fabrication | 83% | 83% | 17% |

## Repository Structure

```
scp_alg_test/
├── scp.py                   # Core SCP prover (GOOD coverage)
├── wikidata_verifier.py     # Wikidata API (WEAK coverage)
├── hallucination_strategies.py  # LLM strategies
├── verified_memory.py       # Caching layer (GOOD coverage)
├── ksg_ground_truth.py      # KnowShowGo (unavailable)
│
├── benchmark.py             # Benchmark suite with coverage tests
├── tests/
│   └── test_scp.py         # 53 unit tests
│
├── docs/
│   └── ...
│
└── README.md
```

## Coverage Levels

| Level | Score | Meaning |
|-------|-------|---------|
| **EXCELLENT** | 90+ | Detects all hallucination types reliably |
| **GOOD** | 70-89 | Reliable for most claim types |
| **MODERATE** | 50-69 | Partial detection, some blind spots |
| **WEAK** | 30-49 | Limited reliability |
| **MINIMAL** | <30 | Not recommended for production |

## Algorithms

### 1. SCP (Symbolic Consistency Probing) - GOOD Coverage

```python
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend

kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([("Bell", "invented", "telephone")])

prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
report = prober.probe("Edison invented the telephone")
print(report.results[0].verdict)  # FAIL or CONTRADICT
```

**Test Results:**
- ✅ 100% contradiction detection
- ✅ 100% fabrication detection  
- ⚠️ 67% false attribution (semantic similarity limitation)

### 2. Wikidata - WEAK Coverage

```python
from wikidata_verifier import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Bell invented the telephone")
```

**Test Results:**
- ✅ 100% false attribution (for supported predicates)
- ❌ 0% contradiction detection (limited predicates)
- ❌ 0% fabrication detection

### 3. Verified Memory - GOOD Coverage

```python
from verified_memory import HallucinationProver

prover = HallucinationProver()
status, provenance, claim = prover.prove("Bell invented the telephone")
```

**Test Results:**
- ✅ 100% contradiction detection
- ✅ 100% fabrication detection
- ⚠️ 0% false attribution (needs KB expansion)

## Running Benchmarks

```bash
# Full benchmark with all algorithms
python benchmark.py

# Verbose output (shows each claim result)
python benchmark.py --verbose

# Single algorithm
python benchmark.py --algorithm scp

# Export to markdown
python benchmark.py --export
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Enable real LLM-as-Judge |
| `ANTHROPIC_API_KEY` | Alternative LLM API |
| `KSG_URL` | KnowShowGo server URL |

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)

## License

MIT License
