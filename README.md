# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (shows all algorithms + rankings)
cd solution && python3 benchmark.py

# Run with verbose output (shows each test + implementation guidance)
cd solution && python3 benchmark.py --verbose

# Run unit tests (53 tests)
cd solution && python3 -m pytest test_scp.py -v
```

---

## Current Status

### Algorithm Rankings (66 test cases)

| Rank | Algorithm | Accuracy | Score | Latency | Status |
|------|-----------|----------|-------|---------|--------|
| 1 | **SCP** | 67% | 69/100 | 4.1ms | ✅ MODERATE |
| 2 | **VerifiedMemory** | 64% | 66/100 | 0.6ms | ✅ MODERATE |
| 3 | Wikidata | 21% | 20/100 | 130ms | ⚠️ MINIMAL |
| 4 | LLM-Judge | 8% | 7/100 | ~200ms | 🔶 Mock |
| 5 | Self-Consistency | 8% | 7/100 | ~500ms | 🔶 Mock |
| 6 | KnowShowGo | N/A | 0/100 | N/A | ❌ Unavailable |

### Detection by Hallucination Type (SCP)

```
true_fact       ██████░░░░   60% (21/35)
false_attr      ███░░░░░░░   38% (5/13)   ← Needs fix
contradiction   ██████████  100% (9/9)    ✓
fabrication     ██████████  100% (9/9)    ✓
```

---

## Project Structure

```
scp_alg_test/
├── solution/                    # ALL CODE LIVES HERE
│   ├── scp.py                  # Core SCP algorithm (BEST: 67%)
│   ├── verified_memory.py      # Caching layer (64%)
│   ├── wikidata_verifier.py    # External KB (21%)
│   ├── hallucination_strategies.py  # LLM strategies (mock)
│   ├── ksg.py                  # KnowShowGo (unavailable)
│   ├── benchmark.py            # Main benchmark suite
│   └── test_scp.py             # 53 unit tests
│
├── docs/
│   ├── project_overview.md     # Full project documentation
│   ├── implementation_roadmap.md
│   └── ...
│
├── README.md                   # This file
└── .agent-instructions         # AI agent guidance
```

---

## Test Coverage

| Test Set | Tests | Description |
|----------|-------|-------------|
| False Attribution | 16 | "Edison invented telephone" → FAIL |
| Contradictions | 16 | "Einstein born in France" → FAIL |
| Fabrications | 16 | "Curie invented smartphone" → FAIL |
| Edge Cases | 8 | Partial matches, synonyms, negations |
| Domain Coverage | 10 | Science, tech, history, geography, arts |
| **TOTAL** | **66** | |

---

## Algorithms

### SCP (Symbolic Consistency Probing) - BEST

```python
from solution.scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend

kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([("Bell", "invented", "telephone")])

prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
report = prober.probe("Edison invented the telephone")
# report.results[0].verdict == Verdict.CONTRADICT
```

**Strengths:** Fast (~4ms), no external API, best accuracy (67%)  
**Weaknesses:** Requires populated KB, 38% false attribution detection

### Verified Memory (Caching)

```python
from solution.verified_memory import HallucinationProver

prover = HallucinationProver()
status, provenance, claim = prover.prove("Bell invented the telephone")
# status == VerificationStatus.VERIFIED
```

**Strengths:** Fast caching (0.6ms), provenance tracking  
**Weaknesses:** Depends on underlying KB quality

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/project_overview.md`](docs/project_overview.md) | **Full project documentation** - scope, goals, test coverage, status |
| [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md) | Prioritized fixes with time estimates |
| [`.agent-instructions`](.agent-instructions) | AI agent conventions and guidance |
| [`docs/knowshowgo_integration_spec.md`](docs/knowshowgo_integration_spec.md) | KnowShowGo API specification |

---

## For AI Agents

See [`.agent-instructions`](.agent-instructions) for detailed guidance.

### First Steps

1. Run `cd solution && python3 benchmark.py` to see current status
2. Run `cd solution && python3 benchmark.py --verbose` to see what needs fixing
3. Read [`docs/project_overview.md`](docs/project_overview.md) for full context
4. Check [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md) for prioritized tasks

### Priority Fixes

| Priority | Task | Effort |
|----------|------|--------|
| HIGH | Fix SCP false attribution | 1-2 hours |
| HIGH | Add Wikidata predicates | 1-2 hours |
| MEDIUM | Enable real LLM | 30 min |
| MEDIUM | Deploy KnowShowGo | 4-8 hours |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Enable real LLM-as-Judge |
| `ANTHROPIC_API_KEY` | Alternative LLM API |
| `KSG_URL` | KnowShowGo server URL |

---

## Coverage Levels

| Level | Score | Meaning |
|-------|-------|---------|
| **EXCELLENT** | 90%+ | Detects all hallucination types reliably |
| **GOOD** | 70-89% | Reliable for most claim types |
| **MODERATE** | 50-69% | Partial detection, some blind spots |
| **WEAK** | 30-49% | Limited reliability |
| **MINIMAL** | <30% | Not recommended for production |

---

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- **All commits tagged:** `[Opus4.5]`

## License

MIT License
