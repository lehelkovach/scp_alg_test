# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Python File Format](#python-file-format)
4. [Console Output Format](#console-output-format)
5. [Module Contents](#module-contents)
6. [Benchmark Results](#benchmark-results)
7. [Hallucination Types](#hallucination-types)
8. [Algorithm Details](#algorithm-details)
9. [Running Tests](#running-tests)
10. [Environment Variables](#environment-variables)
11. [For AI Agents](#for-ai-agents)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark suite (shows all algorithms + coverage)
cd solution && python3 benchmark.py

# Run with verbose output (shows each test + what to fix)
cd solution && python3 benchmark.py --verbose

# Export results to markdown
cd solution && python3 benchmark.py --export

# Run unit tests (53 tests)
cd solution && python3 -m pytest test_scp.py -v
```

---

## Project Structure

```
scp_alg_test/
│
├── solution/                         # ALL CODE LIVES HERE
│   ├── __init__.py                  # Package exports (HyperKB, SCPProber, etc.)
│   ├── scp.py                       # Core SCP algorithm - GOOD coverage (85/100)
│   ├── wikidata_verifier.py         # Wikidata API integration - WEAK (38/100)
│   ├── hallucination_strategies.py  # LLM-based strategies (Judge, Consistency)
│   ├── verified_memory.py           # Caching layer - GOOD coverage (70/100)
│   ├── ksg.py                       # KnowShowGo integration (ground truth + memory)
│   ├── benchmark.py                 # MAIN BENCHMARK SUITE
│   └── test_scp.py                  # Unit tests (53 tests)
│
├── docs/                            # Documentation
│   ├── opus.txt                     # AI decision log (append-only)
│   ├── implementation_roadmap.md    # Prioritized fixes and time estimates
│   ├── knowshowgo_integration_spec.md  # KnowShowGo API spec
│   ├── external_dependencies.txt    # External requirements
│   ├── ksg_architecture.py          # KnowShowGo reference architecture
│   └── neuro_symbolic_architecture.py  # Neuro-symbolic design notes
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
└── .agent-instructions              # Instructions for AI coding agents
```

### What Each File Does

| File | Purpose | Status |
|------|---------|--------|
| `solution/scp.py` | Core hallucination prover using semantic embeddings | ✅ GOOD |
| `solution/wikidata_verifier.py` | Query Wikidata SPARQL for verification | ⚠️ WEAK |
| `solution/hallucination_strategies.py` | LLM-as-Judge, Self-Consistency | 🔶 Mock |
| `solution/verified_memory.py` | Cache verified facts with provenance | ✅ GOOD |
| `solution/ksg.py` | KnowShowGo integration (ground truth + memory) | ❌ Unavailable |
| `solution/benchmark.py` | Run all algorithms, show coverage + fixes | ✅ Ready |
| `solution/test_scp.py` | 53 unit tests for core algorithm | ✅ Passing |

---

## Python File Format

All Python files in this project follow this format:

```python
#!/usr/bin/env python3
"""
Module Title
============

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)

Description of what this module does.

Usage:
    from solution.module import ClassName
    obj = ClassName()
"""

# =============================================================================
# Standard library imports
# =============================================================================
import os
import sys
from typing import List, Dict, Optional

# =============================================================================
# Third-party imports
# =============================================================================
import numpy as np
import networkx as nx

# =============================================================================
# Local imports (within solution/)
# =============================================================================
from scp import HyperKB, Verdict

# Module metadata
__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


# =============================================================================
# SECTION NAME IN CAPS
# =============================================================================

class ClassName:
    """
    Brief class description.
    
    Attributes:
        attr_name (type): Description of attribute.
    
    Example:
        >>> obj = ClassName()
        >>> result = obj.method("input")
        >>> print(result)
    """
    
    def __init__(self, param: str = "default"):
        """Initialize with parameters."""
        self.param = param
    
    def method(self, input_text: str) -> str:
        """
        Method description.
        
        Args:
            input_text: What this parameter does.
        
        Returns:
            Description of return value.
        
        Raises:
            ValueError: When input is invalid.
        """
        return input_text.upper()
```

### Key Conventions

1. **Module docstring** at top with title, author, description, usage
2. **Section separators** using `# ===...` with CAPS titles
3. **Type hints** on all function signatures
4. **Docstrings** on all public classes and methods
5. **`__author__` and `__ai_assistant__`** metadata

---

## Console Output Format

### Benchmark Output Structure

```
======================================================================
HALLUCINATION DETECTION BENCHMARK SUITE
Author: Lehel Kovach | AI: Claude Opus 4.5
======================================================================

Running benchmarks with 3 test sets...

======================================================================
ALGORITHM: SCP
======================================================================
Description: Local KB verification using semantic embeddings (~10ms, 0 API calls)
Status: available - Ready with local KB

  Test Set: False Attribution Detection
  --------------------------------------------------
  Results: 5/6 (83%)
  Avg Latency: 0.7ms

  Detection Coverage (from tests):
  --------------------------------------------------
  true_fact       ███████░░░   78% (7/9)
  false_attr      ██████░░░░   67% (2/3)
  contradiction   ██████████  100% (3/3)
  fabrication     ██████████  100% (3/3)

  Overall Coverage: GOOD (score: 85/100)
```

### Verbose Output (with `--verbose`)

```
  ✓ [true_fact   ] Alexander Graham Bell invented the telep...
      Expected: TRUE, Got: TRUE (PASS)
  ✗ [false_attr  ] Nikola Tesla discovered radium....
      Expected: FALSE, Got: TRUE (SOFT_PASS)
      ⚠️  Detect wrong subject for correct predicate/object
      💡 Example: "Edison invented telephone" should fail (Bell did)
```

### Symbols Used

| Symbol | Meaning |
|--------|---------|
| ✓ | Test passed |
| ✗ | Test failed |
| ⚠️ | Warning / needs attention |
| 📋 | List or checklist |
| 🔧 | Fix needed |
| 🎯 | Priority item |
| █ | Progress bar filled |
| ░ | Progress bar empty |

### Implementation Guidance Output

```
======================================================================
🎯 PRIORITY ACTION ITEMS
======================================================================

Priority Task                                          Time       Action
--------------------------------------------------------------------------------
HIGH     Fix false attribution (8 failures)            1-2 hours  Update scp.py
HIGH     Fix contradiction detection (7 failures)      1-2 hours  Expand KB facts
MEDIUM   Deploy KnowShowGo server                      4-8 hours  See docs/
```

---

## Module Contents

### solution/scp.py - Core Algorithm

**Classes:**
- `Verdict` - Enum: PASS, SOFT_PASS, FAIL, CONTRADICT, UNKNOWN
- `Claim` - Frozen dataclass: subject, predicate, object
- `ProbeResult` - Result for single claim verification
- `SCPReport` - Full verification report with all claims
- `HyperKB` - Hypergraph knowledge base with embeddings
- `SCPProber` - Main prober class

**Key Methods:**
```python
# Create knowledge base with facts
kb = HyperKB(embedding_backend=HashingEmbeddingBackend(dim=512))
kb.add_facts_bulk([
    ("Bell", "invented", "telephone"),
    ("Einstein", "discovered", "relativity"),
])

# Verify text
prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
report = prober.probe("Edison invented the telephone.")
# report.results[0].verdict == Verdict.CONTRADICT
```

### solution/benchmark.py - Test Suite

**Key Exports:**
- `ALGORITHMS` - Dict of all algorithm classes
- `TEST_SETS` - Test data organized by hallucination type
- `IMPLEMENTATION_GUIDANCE` - Fix instructions per failure type
- `run_algorithm_benchmark()` - Run benchmarks for one algorithm

**Command Line:**
```bash
python3 benchmark.py                    # All algorithms
python3 benchmark.py --algorithm scp    # Single algorithm
python3 benchmark.py --verbose          # Detailed output
python3 benchmark.py --export           # Save to markdown
```

### solution/verified_memory.py - Caching Layer

**Classes:**
- `HallucinationProver` - Wraps SCP with standard interface
- `VerifiedMemory` - Persistent cache with provenance
- `VerificationStatus` - Enum: VERIFIED, REFUTED, UNVERIFIABLE

```python
prover = HallucinationProver()
status, provenance, claim = prover.prove("Bell invented the telephone.")
# status == VerificationStatus.VERIFIED
# provenance.source == "local_kb"
```

---

## Benchmark Results

*Generated: 2026-01-14*

### Algorithm Comparison

| Algorithm | Status | Accuracy | Coverage | Score | Latency |
|-----------|--------|----------|----------|-------|---------|
| **SCP** | ✅ available | **83%** | GOOD | 85/100 | 2.4ms |
| **VerifiedMemory** | ✅ available | 72% | GOOD | 70/100 | 0.3ms |
| **Wikidata** | ✅ available | 39% | WEAK | 38/100 | ~200ms |
| **LLM-Judge** | ⚠️ mock | 22% | MINIMAL | 23/100 | ~200ms |
| **Self-Consistency** | ⚠️ mock | 22% | MINIMAL | 23/100 | ~500ms |
| **KnowShowGo** | ❌ unavailable | N/A | N/A | 0/100 | N/A |

### Coverage Levels

| Level | Score | Meaning |
|-------|-------|---------|
| **EXCELLENT** | 90+ | Detects all hallucination types reliably |
| **GOOD** | 70-89 | Reliable for most claim types |
| **MODERATE** | 50-69 | Partial detection, some blind spots |
| **WEAK** | 30-49 | Limited reliability |
| **MINIMAL** | <30 | Not recommended for production |

---

## Hallucination Types

| Type | Description | Example | Detection Goal |
|------|-------------|---------|----------------|
| **TRUE_FACT** | Correct, verifiable | "Bell invented telephone" | Should PASS |
| **FALSE_ATTRIBUTION** | Wrong subject | "Edison invented telephone" | Should FAIL |
| **CONTRADICTION** | Conflicts with KB | "Einstein born in France" | Should FAIL |
| **FABRICATION** | Completely made up | "Curie invented smartphone" | Should FAIL |

---

## Algorithm Details

### 1. SCP (Symbolic Consistency Probing) - BEST

**How it works:**
1. Extract claims from text using rule-based patterns
2. Convert claims to semantic embeddings
3. Search knowledge base for similar facts
4. Return verdict based on match strength

**Strengths:** Fast (~3ms), no external API, good accuracy
**Weaknesses:** Requires populated KB, limited to supported patterns

### 2. Wikidata Verifier

**How it works:**
1. Parse claim into subject/predicate/object
2. Map predicate to Wikidata property ID
3. Query Wikidata SPARQL endpoint
4. Compare results

**Strengths:** Large external KB (100M+ facts)
**Weaknesses:** ~200ms latency, limited predicate support

### 3. Verified Memory (Caching)

**How it works:**
1. Check local cache for verified claim
2. If not found, verify with underlying prover
3. Store result with provenance
4. Return cached result for future queries

**Strengths:** Fast repeated queries, provenance tracking
**Weaknesses:** Depends on underlying prover quality

---

## Running Tests

### Unit Tests (53 tests)

```bash
cd solution
python3 -m pytest test_scp.py -v

# Run specific test class
python3 -m pytest test_scp.py::TestSCPProber -v

# Run with coverage
python3 -m pytest test_scp.py --cov=scp --cov-report=html
```

### Benchmark Suite

```bash
cd solution

# Full benchmark (all algorithms)
python3 benchmark.py

# Single algorithm
python3 benchmark.py --algorithm scp
python3 benchmark.py --algorithm wikidata

# Verbose (shows each test result + fix guidance)
python3 benchmark.py --verbose

# Export to markdown
python3 benchmark.py --export
```

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Enable real LLM-as-Judge | No (mock available) |
| `ANTHROPIC_API_KEY` | Alternative LLM API | No |
| `KSG_URL` | KnowShowGo server URL | No (future) |

---

## For AI Agents

See `.agent-instructions` for comprehensive guidance on:
- Project structure and conventions
- File format standards
- How to run tests and benchmarks
- Current coverage gaps and fixes
- Commit message format
- What NOT to do

### First Steps for Any Agent

1. Run `cd solution && python3 benchmark.py` to see current status
2. Run `cd solution && python3 -m pytest test_scp.py -v` to verify tests pass
3. Read `docs/implementation_roadmap.md` for prioritized work
4. Check `.agent-instructions` for detailed conventions

---

## License

MIT License

---

## Authorship

- **Author:** Lehel Kovach
- **AI Assistant:** Claude Opus 4.5 (Anthropic)
- **All commits tagged:** `[Opus4.5]`
