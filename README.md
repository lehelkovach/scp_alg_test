# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all solution comparisons
python run_tests.py --verbose

# Run unit tests (53 tests)
pytest tests/test_scp.py -v

# Run demos
python tests/demos.py           # All demos
python tests/demos.py scp       # SCP demo
python tests/demos.py wikidata  # Wikidata demo
```

## Repository Structure

```
scp_alg_test/
├── scp.py                   # Core SCP hallucination prover
├── wikidata_verifier.py     # Wikidata API verification (100M+ facts)
├── hallucination_strategies.py  # LLM-based strategies
├── verified_memory.py       # Verification + caching layer
├── ksg_ground_truth.py      # KnowShowGo ground truth
├── ksg_integration.py       # KnowShowGo REST client
│
├── tests/
│   ├── test_scp.py         # 53 unit tests
│   └── demos.py            # Demo scripts for all methods
│
├── docs/
│   ├── opus.txt            # Design decisions
│   ├── external_dependencies.txt
│   ├── knowshowgo_integration_spec.md
│   └── ...
│
├── run_tests.py            # Solution comparison runner
├── requirements.txt        # Dependencies
└── README.md
```

## Solutions Overview

| Solution | Speed | API Calls | Best For |
|----------|-------|-----------|----------|
| **SCP** | ~10ms | 0 | High-volume, known domains |
| **Wikidata** | ~200ms | 0* | General facts, no setup |
| **LLM Judge** | ~200ms | 1 | Quick checks |

*Wikidata is a free public API

## Algorithm Explanations

### 1. SCP (Symbolic Consistency Probing)

Verifies claims against a knowledge base using semantic similarity.

```python
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend

# Create KB
kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([("Bell", "invented", "telephone")])

# Create prober
prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())

# Verify claim
report = prober.probe("Edison invented the telephone")
print(report.results[0].verdict)  # FAIL or CONTRADICT
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

Uses an LLM to verify claims.

```python
from hallucination_strategies import LLMJudgeStrategy, mock_llm

judge = LLMJudgeStrategy(mock_llm)  # Replace with real LLM
result = judge.check("Edison invented the telephone")
print(result.verdict)  # FAIL
```

## Running Tests

```bash
# Unit tests (53 tests)
pytest tests/test_scp.py -v

# Solution comparison
python run_tests.py --verbose

# Specific solution
python run_tests.py --solution wikidata
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

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- All commits tagged with `[Opus4.5]`

## License

MIT License
