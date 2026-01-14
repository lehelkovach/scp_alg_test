# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python run_tests.py --verbose

# Run specific solution
python run_tests.py --solution wikidata

# Run demos
python solutions/scp/scp_prover.py
python solutions/wikidata/wikidata_prover.py
```

## Solutions Overview

| Solution | Speed | API Calls | Accuracy | Best For |
|----------|-------|-----------|----------|----------|
| **SCP** | ~10ms | 0 | Limited by KB | High-volume, known domains |
| **Wikidata** | ~200ms | 0* | High | General facts, no setup |
| **Zero-Resource** | ~10ms | 0 | Context-dependent | RAG faithfulness |
| **LLM Judge** | ~200ms | 1 | Medium | Quick checks |
| **Hybrid** | ~50ms avg | 0.1 avg | High | Production systems |
| **KnowShowGo** | ~10ms | 0 | Highest | Full cognitive architecture |
| **API Service** | ~10ms | 0 | High | REST API access |

*Wikidata is a free public API

## Repository Structure

```
scp_alg_test/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── run_tests.py                 # Main test runner [Opus4.5]
│
├── lib/                         # Core library implementations
│   ├── __init__.py
│   ├── scp.py                  # SCP hallucination prover
│   ├── wikidata_verifier.py    # Wikidata API integration
│   ├── verified_memory.py      # Verification + caching
│   ├── hallucination_strategies.py  # Strategy comparison
│   ├── ksg_ground_truth.py     # KnowShowGo architecture
│   └── ksg_integration.py      # KnowShowGo REST client
│
├── solutions/                   # Organized solution implementations
│   ├── scp/                    # Knowledge Base verification
│   │   └── scp_prover.py
│   ├── wikidata/               # External knowledge graph
│   │   └── wikidata_prover.py
│   ├── llm_strategies/         # LLM-based approaches
│   │   └── llm_judge.py
│   ├── knowshowgo/             # Cognitive architecture
│   │   └── ksg_prover.py
│   ├── hybrid/                 # Combined approach
│   │   └── hybrid_prover.py
│   ├── zero_resource/          # Context-based verification
│   │   └── faithfulness_checker.py
│   └── api_service/            # REST API service
│       └── graph_memory_service.py
│
├── tests/                       # Test files
│   ├── test_scp.py             # Unit tests
│   └── demo_scp.py             # Demo script
│
└── docs/                        # Documentation
    ├── opus.txt                # Design decisions [Opus4.5]
    ├── gemini.txt              # Previous session notes
    └── neuro_symbolic_architecture.py  # Architecture analysis
```

## Algorithm Explanations

### 1. SCP (Symbolic Consistency Probing)
Compares claims against a pre-built knowledge base using embeddings.
- **Speed:** ~10ms | **API Calls:** 0
- **Verdicts:** PASS, SOFT_PASS, FAIL, CONTRADICT

### 2. Wikidata Verification
Queries Wikidata's 100M+ facts via SPARQL.
- **Speed:** ~200ms | **API Calls:** 0 (free public API)
- **No setup required** - instant access to structured knowledge

### 3. Zero-Resource Faithfulness
Checks if LLM output is faithful to source context (RAG scenarios).
- **Speed:** ~10ms | **API Calls:** 0
- **Detects:** Extrinsic hallucinations (added info), Intrinsic (contradictions)

### 4. LLM-as-Judge
Uses an LLM to verify claims.
- **Speed:** ~200ms | **API Calls:** 1
- **Strategies:** Single judge, self-consistency, cross-model

### 5. Hybrid Cascade
Combines KB → Wikidata → LLM in order of speed/cost.
- **Speed:** ~50ms avg | **API Calls:** 0.1 avg
- **Best of all worlds** for production

### 6. KnowShowGo (Cognitive Architecture)
Fuzzy ontology knowledge graph mirroring human cognition.
- UUID tokens, weighted edges, winner-take-all emergence
- See https://github.com/lehelkovach/knowshowgo

### 7. API Service
REST API for continuous KB building and verification.
- **Endpoints:** /ingest, /verify, /query, /stats

## Usage Examples

### Basic Verification
```python
from lib import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Edison invented the telephone")
print(result.status)  # "refuted"
print(result.reason)  # "Wikidata shows Bell invented telephone"
```

### RAG Faithfulness
```python
from solutions.zero_resource import check_faithfulness

context = "Revenue increased 15%. CEO Jane Doe announced partnership."
answer = "Revenue went up 15% and stock rose 10%."  # Hallucination!

report, hallucinations = check_faithfulness(context, answer)
# hallucinations = [Claim about stock - not in context]
```

### Hybrid Verification
```python
from lib import HybridVerifier

hybrid = HybridVerifier(
    local_facts=[("Python", "created_by", "Guido van Rossum")],
    use_wikidata=True
)
result = hybrid.verify("Bell invented the telephone")
# Fast KB check → Wikidata fallback → LLM fallback
```

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- All Opus 4.5 commits tagged with `[Opus4.5]` prefix

## License

MIT License
