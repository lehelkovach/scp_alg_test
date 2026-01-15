# Hallucination Detection System - Project Overview

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Last Updated:** 2026-01-14

---

## Table of Contents

1. [Project Scope](#project-scope)
2. [Goals](#goals)
3. [Project Structure](#project-structure)
4. [Test Coverage](#test-coverage)
5. [Current Status](#current-status)
6. [Algorithm Rankings](#algorithm-rankings)
7. [For AI Agents](#for-ai-agents)

---

## Project Scope

This project implements and benchmarks **hallucination detection algorithms** for verifying Large Language Model (LLM) outputs. The system determines whether claims made by an LLM are:

- **True** - Verified against a knowledge base
- **False** - Contradicts known facts or is fabricated
- **Unverifiable** - Cannot be confirmed or denied

### What This Project Does

1. **Extracts claims** from natural language text (subject-predicate-object triples)
2. **Verifies claims** against multiple knowledge sources
3. **Tracks provenance** for audit trails
4. **Caches results** for efficient repeated queries
5. **Benchmarks algorithms** with comprehensive test suites

### What This Project Does NOT Do

- Does not train or fine-tune LLMs
- Does not generate text (only verifies it)
- Does not provide real-time API service (benchmark/library only)
- Does not include production deployment configs

---

## Goals

### Primary Goals

| Goal | Status | Notes |
|------|--------|-------|
| Detect false attribution | ⚠️ Partial | "Edison invented telephone" → FAIL |
| Detect contradictions | ✅ Working | "Einstein born in France" → FAIL |
| Detect fabrications | ✅ Working | "Curie invented smartphone" → FAIL |
| Track provenance | ✅ Working | Source, confidence, timestamp |
| Benchmark algorithms | ✅ Working | 66 test cases, 6 algorithms |

### Secondary Goals

| Goal | Status | Notes |
|------|--------|-------|
| KnowShowGo integration | ❌ Pending | Requires server deployment |
| Real LLM-as-Judge | ❌ Pending | Requires API key |
| Wikidata expansion | ⚠️ Partial | Limited predicate support |
| Production API | ❌ Not started | FastAPI service planned |

### Success Metrics

- **Accuracy**: 67% (SCP, best available)
- **Coverage Score**: 69/100 (MODERATE)
- **Latency**: <10ms for local KB verification
- **Test Coverage**: 66 test cases across 5 categories

---

## Project Structure

```
scp_alg_test/
│
├── solution/                         # ALL CODE LIVES HERE
│   ├── __init__.py                  # Package exports
│   ├── scp.py                       # Core SCP algorithm (BEST: 67%)
│   ├── wikidata_verifier.py         # Wikidata API (21%)
│   ├── hallucination_strategies.py  # LLM strategies (mock: 8%)
│   ├── verified_memory.py           # Caching layer (64%)
│   ├── ksg.py                       # KnowShowGo (unavailable)
│   ├── benchmark.py                 # Main benchmark suite
│   └── test_scp.py                  # Unit tests (53 tests)
│
├── docs/                            # Documentation
│   ├── project_overview.md          # THIS FILE
│   ├── opus.txt                     # AI decision log
│   ├── implementation_roadmap.md    # Prioritized fixes
│   ├── knowshowgo_integration_spec.md
│   └── external_dependencies.txt
│
├── README.md                        # Quick start guide
├── requirements.txt                 # Python dependencies
├── .gitignore
└── .agent-instructions              # AI agent guidance
```

### Module Responsibilities

| Module | Purpose | Accuracy | Latency |
|--------|---------|----------|---------|
| `scp.py` | Core verification using semantic embeddings | 67% | ~4ms |
| `verified_memory.py` | Cached verification with provenance | 64% | ~0.6ms |
| `wikidata_verifier.py` | External KB queries | 21% | ~130ms |
| `hallucination_strategies.py` | LLM-based strategies | 8%* | ~200ms |
| `ksg.py` | KnowShowGo semantic memory | N/A | N/A |
| `benchmark.py` | Test runner and metrics | - | - |

*Mock implementation

---

## Test Coverage

### Overview

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Test Cases** | 66 | Benchmark suite |
| **Unit Tests** | 53 | test_scp.py |
| **Test Sets** | 5 | By hallucination type |
| **Algorithms Tested** | 6 | SCP, Wikidata, LLM, etc. |

### Test Sets

#### 1. False Attribution Detection (16 tests)
Tests if algorithm can detect wrong subject attribution.

```
Example: "Edison invented the telephone" → FALSE (Bell did)
```

| Category | Tests | Examples |
|----------|-------|----------|
| Inventions | 6 | Bell/Edison/telephone, Edison/Tesla/lightbulb |
| Discoveries | 6 | Einstein/Newton/relativity, Curie/Tesla/radium |
| Founders | 4 | Gates/Jobs/Microsoft, Zuckerberg/Musk/Facebook |

#### 2. Contradiction Detection (16 tests)
Tests if algorithm can detect claims that contradict known facts.

```
Example: "Einstein was born in France" → FALSE (Germany)
```

| Category | Tests | Examples |
|----------|-------|----------|
| Birth places | 4 | Einstein/Germany, Newton/England |
| Landmarks | 6 | Eiffel Tower/Paris, Colosseum/Rome |
| Capitals | 6 | Tokyo/Japan, Paris/France, Berlin/Germany |

#### 3. Fabrication Detection (16 tests)
Tests if algorithm can detect completely made-up facts.

```
Example: "Einstein invented the internet" → FALSE
```

| Category | Tests | Examples |
|----------|-------|----------|
| Tech fabrications | 4 | Einstein/internet, Einstein/computer |
| Historical fabrications | 6 | Curie/smartphone, Napoleon/Great Wall |
| Absurd fabrications | 6 | Newton/time travel, Caesar/Antarctica |

#### 4. Edge Cases (8 tests)
Tests boundary conditions and tricky cases.

| Type | Tests | Examples |
|------|-------|----------|
| Partial matches | 2 | "Bell invented telephone" (no article) |
| Synonyms | 2 | "created" vs "invented", "found" vs "discovered" |
| Near-misses | 2 | Edison/electricity (close but wrong) |
| Negations | 2 | "Einstein did not discover gravity" |

#### 5. Domain Coverage (10 tests)
Tests across different knowledge domains.

| Domain | Tests | Examples |
|--------|-------|----------|
| Science | 2 | Darwin/evolution |
| Technology | 2 | Jobs/Apple |
| History | 2 | Washington/first president |
| Geography | 2 | Everest/tallest mountain |
| Arts | 2 | Beethoven/9th Symphony |

### Coverage by Hallucination Type

| Type | Tests | Best Algorithm | Detection Rate |
|------|-------|----------------|----------------|
| True Facts | 35 | VerifiedMemory | 71% |
| False Attribution | 13 | Wikidata | 31% |
| Contradictions | 9 | SCP | 100% |
| Fabrications | 9 | SCP | 100% |

### Unit Tests (test_scp.py)

53 unit tests covering:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestClaim | 4 | Claim dataclass |
| TestRuleBasedExtractor | 8 | Claim extraction patterns |
| TestHyperKB | 10 | Knowledge base operations |
| TestStringSimilarityBackend | 3 | String matching |
| TestHashingEmbeddingBackend | 3 | Embedding generation |
| TestSCPProber | 12 | Core verification |
| TestVerdictScenarios | 4 | Verdict logic |
| TestEdgeCases | 5 | Boundary conditions |
| TestFalseAttributionDetection | 4 | Attribution errors |
| TestCoreferenceHandling | 2 | Pronoun resolution |

---

## Current Status

### Algorithm Status

| Algorithm | Status | Accuracy | Coverage | Priority Fix |
|-----------|--------|----------|----------|--------------|
| **SCP** | ✅ Available | 67% | MODERATE | False attribution |
| **VerifiedMemory** | ✅ Available | 64% | MODERATE | False attribution |
| **Wikidata** | ✅ Available | 21% | MINIMAL | Add predicates |
| **LLM-Judge** | 🔶 Mock | 8% | MINIMAL | Add API key |
| **Self-Consistency** | 🔶 Mock | 8% | MINIMAL | Add API key |
| **KnowShowGo** | ❌ Unavailable | N/A | N/A | Deploy server |

### Detection Capabilities

```
                    SCP    Wikidata  LLM-Judge  VerifiedMem
                    ───    ────────  ─────────  ───────────
True Facts          60%    20%       8%         71%
False Attribution   38%    31%       8%         0%
Contradictions      100%   0%        0%         100%
Fabrications        100%   33%       0%         100%
```

### Known Issues

1. **SCP False Attribution** (38% detection)
   - Problem: Semantic similarity matches predicate+object but ignores subject
   - Fix: Add subject-aware matching in `find_similar_facts()`
   - Effort: 1-2 hours

2. **Wikidata Limited Predicates** (0% contradiction)
   - Problem: Only supports "invented/discovered" predicates
   - Fix: Add born_in, located_in, capital_of predicates
   - Effort: 1-2 hours

3. **LLM Strategies Mock-Only** (8% accuracy)
   - Problem: No real LLM API configured
   - Fix: Set OPENAI_API_KEY environment variable
   - Effort: 30 minutes

4. **KnowShowGo Unavailable**
   - Problem: Server not deployed
   - Fix: Deploy from github.com/lehelkovach/knowshowgo
   - Effort: 4-8 hours

---

## Algorithm Rankings

### Current Ranking (66 test cases)

| Rank | Algorithm | Accuracy | Score | Latency | Status |
|------|-----------|----------|-------|---------|--------|
| 1 | **SCP** | 67% | 69/100 | 4.1ms | ✅ MODERATE |
| 2 | **VerifiedMemory** | 64% | 66/100 | 0.6ms | ✅ MODERATE |
| 3 | Wikidata | 21% | 20/100 | 130ms | ⚠️ MINIMAL |
| 4 | LLM-Judge | 8% | 7/100 | ~200ms | 🔶 Mock |
| 5 | Self-Consistency | 8% | 7/100 | ~500ms | 🔶 Mock |
| 6 | KnowShowGo | N/A | 0/100 | N/A | ❌ Unavailable |

### Recommended Use Cases

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Fast local verification | SCP | Best accuracy, low latency |
| Repeated queries | VerifiedMemory | Caching, provenance |
| Invention/discovery claims | Wikidata | 100% for supported predicates |
| Complex reasoning | LLM-Judge (real) | Broad knowledge |
| Audit requirements | VerifiedMemory | Full provenance chain |

---

## For AI Agents

### Quick Start

```bash
# 1. Run benchmark to see current status
cd solution && python3 benchmark.py

# 2. Run with verbose output (shows what needs fixing)
cd solution && python3 benchmark.py --verbose

# 3. Run unit tests
cd solution && python3 -m pytest test_scp.py -v
```

### Priority Tasks

| Priority | Task | File | Effort |
|----------|------|------|--------|
| HIGH | Fix false attribution | scp.py | 1-2 hours |
| HIGH | Add Wikidata predicates | wikidata_verifier.py | 1-2 hours |
| MEDIUM | Enable real LLM | hallucination_strategies.py | 30 min |
| MEDIUM | Deploy KnowShowGo | External | 4-8 hours |

### Key Files to Understand

1. **`solution/scp.py`** - Core algorithm, start here
2. **`solution/benchmark.py`** - Test suite, see TEST_SETS
3. **`solution/test_scp.py`** - Unit tests, understand expected behavior
4. **`.agent-instructions`** - Detailed conventions and guidance

### Commit Format

```
[Opus4.5] <verb>: <description>

- Detail 1
- Detail 2
```

### Do NOT

- Modify test expectations without documenting why
- Add dependencies without updating requirements.txt
- Push to branches other than main or cursor/redundant-branches-check-b148
- Create production deployment configs without confirmation

---

## Appendix: Coverage Levels

| Level | Score | Meaning |
|-------|-------|---------|
| **EXCELLENT** | 90%+ | Detects all hallucination types reliably |
| **GOOD** | 70-89% | Reliable for most claim types |
| **MODERATE** | 50-69% | Partial detection, some blind spots |
| **WEAK** | 30-49% | Limited reliability |
| **MINIMAL** | <30% | Not recommended for production |

---

*Document generated by Claude Opus 4.5 for Lehel Kovach*
