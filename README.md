# Hallucination Detection System

**Author:** Lehel Kovach  
**AI Assistant:** Claude Opus 4.5 (Anthropic)  
**Repository:** https://github.com/lehelkovach/scp_alg_test

A comprehensive toolkit for detecting hallucinations in Large Language Model (LLM) outputs. This repository contains multiple approaches to verification, from simple knowledge base lookups to sophisticated cognitive architectures.

## Purpose

LLMs can generate plausible-sounding but factually incorrect statements (hallucinations). This repository provides:

1. **Multiple detection algorithms** - Different approaches with different tradeoffs
2. **Production-ready solutions** - Wikidata integration provides instant access to 100M+ facts
3. **Research framework** - Architecture for building cognitive ground truth systems
4. **Benchmarking tools** - Compare accuracy and speed across solutions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python run_tests.py

# Run specific solution
python run_tests.py --solution wikidata --verbose

# Run demos
python solutions/scp/scp_prover.py
python solutions/wikidata/wikidata_prover.py
```

## Solutions Overview

| Solution | Speed | API Calls | Accuracy | Best For |
|----------|-------|-----------|----------|----------|
| **SCP** | ~10ms | 0 | Limited by KB | High-volume, known domains |
| **Wikidata** | ~200ms | 0* | High | General facts, no setup |
| **LLM Judge** | ~200ms | 1 | Medium | Quick checks, no KB |
| **Self-Consistency** | ~500ms | 3-5 | Medium | Catching inconsistency |
| **Cross-Model** | ~400ms | 2+ | High | High-stakes decisions |
| **Hybrid** | ~50ms avg | 0.1 avg | High | Production systems |
| **KnowShowGo** | ~10ms | 0 | Highest | Full cognitive architecture |

*Wikidata is a free public API

## Repository Structure

```
scp_alg_test/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_tests.py             # Main test runner (Opus 4.5)
├── opus.txt                 # Design decisions & reasoning (Opus 4.5)
│
├── solutions/               # Organized solution implementations
│   ├── scp/                # Knowledge Base verification
│   │   ├── __init__.py
│   │   └── scp_prover.py   # SCP algorithm implementation
│   │
│   ├── wikidata/           # External knowledge graph
│   │   ├── __init__.py
│   │   └── wikidata_prover.py  # Wikidata SPARQL queries
│   │
│   ├── llm_strategies/     # LLM-based approaches
│   │   ├── __init__.py
│   │   └── llm_judge.py    # Judge, self-consistency, cross-model
│   │
│   ├── knowshowgo/         # Cognitive architecture
│   │   ├── __init__.py
│   │   └── ksg_prover.py   # Fuzzy ontology ground truth
│   │
│   └── hybrid/             # Combined approach
│       ├── __init__.py
│       └── hybrid_prover.py  # KB → Wikidata → LLM cascade
│
├── docs/                    # Documentation
│   └── opus.txt            # Detailed design decisions
│
├── scp.py                  # Core SCP implementation
├── wikidata_verifier.py    # Wikidata integration
├── verified_memory.py      # Verification + caching layer
├── hallucination_strategies.py  # Strategy comparison
├── ksg_ground_truth.py     # KnowShowGo architecture
├── ksg_integration.py      # KnowShowGo REST client
└── neuro_symbolic_architecture.py  # Architecture analysis
```

## Algorithm Explanations

### 1. SCP (Symbolic Consistency Probing)

**How it works:**
1. Parse claim into subject-predicate-object triple
2. Generate embedding for the claim
3. Search knowledge base for similar facts
4. Compare and return verdict

```python
Claim: "Edison invented the telephone"
Extract: (Edison, invented, telephone)
KB Search: (Bell, invented, telephone) found with 0.89 similarity
Verdict: CONTRADICT - Bell invented it, not Edison
```

**Strengths:** Fast (~10ms), deterministic, full provenance  
**Limitations:** Limited to facts in KB

### 2. Wikidata Verification

**How it works:**
1. Parse claim to extract subject, predicate, object
2. Send SPARQL query to Wikidata endpoint
3. Compare Wikidata's answer to claimed subject
4. Return verified/refuted with source

```python
Claim: "Edison invented the telephone"
SPARQL: SELECT ?inventor WHERE { ?item rdfs:label "telephone"@en. ?item wdt:P61 ?inventor }
Result: "Alexander Graham Bell"
Verdict: REFUTED - Wikidata says Bell, not Edison
```

**Strengths:** 100M+ facts, no setup, always current  
**Limitations:** ~200ms latency, rate limited

### 3. LLM-as-Judge

**How it works:**
1. Format claim as verification prompt
2. Ask LLM: "Is this statement true? Answer TRUE/FALSE/UNCERTAIN"
3. Parse response and extract confidence
4. Return verdict

```python
Prompt: "Is this statement factually correct? 'Edison invented the telephone'"
LLM Response: "FALSE - Alexander Graham Bell invented the telephone in 1876"
Verdict: REFUTED
```

**Strengths:** No KB needed, handles novel claims  
**Limitations:** LLM can also hallucinate, costs per call

### 4. Self-Consistency

**How it works:**
1. Ask same LLM the verification question 3-5 times
2. Compare answers across samples
3. Inconsistency suggests hallucination

```python
Sample 1: "TRUE"
Sample 2: "FALSE" 
Sample 3: "TRUE"
Analysis: Inconsistent (2 TRUE, 1 FALSE)
Verdict: UNCERTAIN - LLM is not confident
```

**Strengths:** Catches uncertain hallucinations  
**Limitations:** Doesn't catch confident hallucinations, 3-5x cost

### 5. Hybrid Cascade

**How it works:**
1. Check local KB (fastest, free)
2. If unknown, check Wikidata (comprehensive, free)
3. If still unknown, ask LLM (last resort)
4. Cache results for future queries

```
Query → [KB: 10ms] → found? → return
              ↓ not found
        [Wikidata: 200ms] → found? → return + cache
              ↓ not found  
        [LLM: 200ms] → return + cache
```

**Strengths:** Best of all worlds, ~50ms average  
**Limitations:** Complexity

### 6. KnowShowGo (Cognitive Architecture)

**How it works:**
1. Store facts as propositions (subject-predicate-object nodes)
2. Link with weighted associations (confidence, provenance)
3. Fuzzy search via embeddings
4. Winner-take-all for canonical truth

```
Claim: "Edison invented telephone"
       ↓
Search: Find similar propositions
       ↓
Match: (Bell, invented, telephone) - similarity 0.89
       ↓
Associations:
  ├── derived_from → Wikipedia (trust: 0.95)
  └── verified_by → USPTO Patent (trust: 0.99)
       ↓
Verdict: REFUTED with full provenance
```

**Strengths:** Cognitive-aligned, full provenance, fuzzy → discrete  
**Limitations:** Requires KnowShowGo infrastructure

## Usage Examples

### Basic Verification

```python
from solutions.wikidata import WikidataVerifier

verifier = WikidataVerifier()
result = verifier.verify("Edison invented the telephone")

print(result.status)      # "refuted"
print(result.confidence)  # 0.90
print(result.reason)      # "Wikidata shows Bell invented telephone"
```

### Hybrid Verification

```python
from solutions.hybrid import HybridVerifier

hybrid = HybridVerifier(
    local_facts=[("Python", "created_by", "Guido van Rossum")],
    use_wikidata=True
)

# Fast path (local KB)
result = hybrid.verify("Python was created by Guido van Rossum")
# → 10ms, source: "local_kb"

# Wikidata path
result = hybrid.verify("Bell invented the telephone")
# → 200ms, source: "wikidata"
```

### Adding Custom Facts

```python
from solutions.scp import HyperKB, SCPProber, RuleBasedExtractor, StringSimilarityBackend

kb = HyperKB(embedding_backend=StringSimilarityBackend())
kb.add_fact("My Company", "founded_by", "John Doe", source="internal", confidence=1.0)

prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
result = prober.probe("John Doe founded My Company")
# → PASS
```

## Design Philosophy

See `opus.txt` for detailed reasoning behind every design decision.

Key principles:
1. **Separation of concerns:** LLM for language, Graph for facts
2. **Provenance tracking:** Every claim has an audit trail
3. **Fuzzy → Discrete:** Winner-take-all emergence for canonical truth
4. **Cognitive alignment:** Mirrors how humans organize knowledge
5. **Iterative improvement:** KB grows from verified outputs

## Contributing

This repository was developed by Lehel Kovach with assistance from Claude Opus 4.5.

All commits from Opus 4.5 are tagged with `[Opus4.5]` in the commit message or branch name.

## License

MIT License - See LICENSE file for details.

---

*Built with AI assistance from Claude Opus 4.5 (Anthropic)*
