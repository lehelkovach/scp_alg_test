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
| **LLM Judge** | ~200ms | 1 | Medium | Quick checks |

*Wikidata is a free public API

### SCP Modes

The SCP solution includes three modes in one unified module:

| Mode | Use Case |
|------|----------|
| KB Mode | Verify against pre-built knowledge base |
| Context Mode | RAG faithfulness checking |
| API Mode | REST service for continuous verification |

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
│   │   └── scp_prover.py      # KB, Context, and API modes
│   ├── wikidata/               # External knowledge graph
│   │   └── wikidata_prover.py
│   └── llm/                    # LLM-based approaches
│       └── llm_judge.py
│
├── tests/                       # Test files
│   ├── test_scp.py             # Unit tests
│   └── demo_scp.py             # Demo script
│
└── docs/                        # Documentation
    ├── opus.txt                # Design decisions [Opus4.5]
    ├── gemini.txt              # Previous session notes
    ├── neuro_symbolic_architecture.py  # Architecture analysis
    └── ksg_architecture.py     # KnowShowGo reference
```

## Algorithm Explanations

### 1. SCP (Symbolic Consistency Probing)

Unified hallucination detection with three modes:

**KB Mode:** Verify against pre-built knowledge base
- Build KB from facts: `prover.add_facts([("Bell", "invented", "telephone")])`
- Verify claims: `prover.verify("Edison invented telephone")` → refuted

**Context Mode:** RAG faithfulness checking (zero external dependencies)
- Check LLM output against source document
- Detects: Extrinsic (added info), Intrinsic (contradictions)

**API Mode:** REST service
- `POST /ingest` - Add facts to KB
- `POST /verify` - Verify answer
- `GET /stats` - KB statistics

Speed: ~10ms | API Calls: 0

### 2. Wikidata Verification

Queries Wikidata's 100M+ facts via SPARQL.
- **Speed:** ~200ms | **API Calls:** 0 (free public API)
- **No setup required** - instant access to structured knowledge

### 3. LLM-as-Judge

Uses an LLM to verify claims.
- **Speed:** ~200ms | **API Calls:** 1
- **Strategies:** Single judge, self-consistency, cross-model

## Usage Examples

### Basic KB Verification
```python
from solutions.scp import SCPKBProver

prover = SCPKBProver()
prover.add_facts([("Bell", "invented", "telephone")])

result = prover.verify("Edison invented the telephone")
print(result["status"])  # "refuted"
```

### RAG Faithfulness
```python
from solutions.scp import check_faithfulness

context = "Revenue increased 15%. CEO Jane Doe announced partnership."
answer = "Revenue went up 15% and stock rose 10%."  # Hallucination!

report, hallucinations = check_faithfulness(context, answer)
# hallucinations = [Claim about stock - not in context]
```

### Wikidata Verification
```python
from solutions.wikidata import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Edison invented the telephone")
print(result.status)  # REFUTED
```

### API Service
```bash
uvicorn solutions.scp.scp_prover:app --port 8000

# Ingest facts
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Bell invented the telephone."}'

# Verify claims
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"answer_text": "Edison invented the telephone."}'
```

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- All Opus 4.5 commits tagged with `[Opus4.5]` prefix

## License

MIT License
