# scp_alg_test

This repo contains a prototype **hallucination detection** / **consistency probing** approach:

- **Knowledge representation**: a directed hypergraph (relation-as-node / “edge-as-node”) stored in-memory via `networkx`
- **Claim extraction**: rule-based patterns by default (optional LLM-based extractor hook)
- **Verification**: compare extracted claims against the KB via a pluggable similarity backend

### Do I need an OpenAI key (or any API key)?

- **No**, not to run the system end-to-end. By default it uses:
  - **Rule-based claim extraction** (no LLM calls)
  - **Local similarity** via `HashingEmbeddingBackend` (no downloads, no keys)
- **Yes**, only if you choose to plug in an API-backed LLM for claim extraction or generation (e.g. OpenAI/Anthropic).
- **No key** is needed for `sentence-transformers`, but it **will download a model** (internet required) unless you point it at a local model.

### Run the demo

```bash
python3 -m pip install -r requirements.txt
python3 test.py
```

### Run tests

```bash
python3 -m unittest -q
```