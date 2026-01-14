# scp_alg_test

This repo contains experiments around **hallucination detection / answer verification**.

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run tests (recommended):

```bash
python3 -m unittest -q
```

## Files added for hallucination detection strategies

These modules are intentionally small and heavily commented so you can treat them as reference implementations:

- **`hd_claim_extract_rules.py`**: rule-based claim extraction (fastest, limited coverage).
- **`hd_verify_kb.py`**: verify claims against a curated KB (exact match + contradiction).
- **`hd_verify_evidence.py`**: verify claims against retrieved evidence snippets (evidence-bounded + provenance).
- **`hd_self_consistency.py`**: measure stability by sampling the *same* model multiple times (risk signal, not truth).
- **`hd_cross_model.py`**: measure agreement across multiple models (risk signal, not truth).
- **`hd_memory.py`**: a tiny “layer-2 memory” (verification cache + provenance store) suitable for RAG reuse.

Each strategy has a corresponding `unittest` module:

- `test_hd_claim_extraction.py`
- `test_hd_verify_kb.py`
- `test_hd_verify_evidence.py`
- `test_hd_consistency.py`
- `test_hd_memory.py`
