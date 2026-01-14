import scp
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

def check_faithfulness(context_text: str, generated_answer: str):
    """
    Checks if the generated answer is supported by the context_text.
    This detects 'Extrinsic Hallucinations' (adding info not in source)
    without using any external internet queries.
    """
    
    # 1. Setup Lightweight Backend (No API keys, purely local hashing)
    # Using HashingEmbeddingBackend is the fastest option (0ms network latency)
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)
    extractor = RuleBasedExtractor() # Regex-based, near-zero latency

    # 2. Build "Ground Truth" Graph from Context
    # We treat the source text as the absolute truth for this check
    print(f"--- Building Graph from Context ({len(context_text)} chars) ---")
    context_claims = extractor.extract(context_text)
    
    # Ingest context facts into KB
    for c in context_claims:
        kb.add_fact(c.subject, c.predicate, c.obj, confidence=1.0, source="context")

    print(f"KB Stats: {kb.stats()} - {len(context_claims)} facts indexed.")

    # 3. Probe the Answer against the Context Graph
    print(f"\n--- Probing Answer ---")
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)
    report = prober.probe(generated_answer)

    # 4. Analyze Results
    hallucinations = []
    
    for res in report.results:
        # In a RAG/Context scenario:
        # FAIL = The model said something not in the context (Extrinsic Hallucination)
        # CONTRADICT = The model contradicted the context (Intrinsic Hallucination)
        if res.verdict in [Verdict.FAIL, Verdict.CONTRADICT]:
            hallucinations.append(res)

    return report, hallucinations

# --- DEMO ---
if __name__ == "__main__":
    # Example: A RAG scenario regarding a fictional report
    context = """
    The Q3 financial report shows that Acme Corp revenue increased by 15%. 
    The CEO, Jane Doe, announced a new partnership with Beta Ltd.
    Operating costs decreased by 5% due to automation.
    """

    # 1. Faithful Answer (Should Pass)
    ans_good = "Jane Doe announced a partnership with Beta Ltd, and revenue went up 15%."

    # 2. Hallucinated Answer (Contains info NOT in context - 'Stock rose 10%')
    ans_hallucination = "Acme Corp revenue increased by 15% and their stock price rose 10%."

    # 3. Contradictory Answer (Contains wrong info - 'Revenue decreased')
    ans_contradiction = "Operating costs decreased, but revenue also decreased by 15%."

    for i, ans in enumerate([ans_good, ans_hallucination, ans_contradiction]):
        print(f"\n\nTEST CASE {i+1}:")
        print(f"Answer: {ans}")
        report, errors = check_faithfulness(context, ans)
        
        if not errors:
            print("✅ VERDICT: FAITHFUL (No hallucinations detected)")
        else:
            print(f"❌ VERDICT: HALLUCINATION DETECTED ({len(errors)} claims)")
            for e in errors:
                print(f"   - Claim: '{e.claim}' -> {e.verdict.value} (Reason: {e.reason})")
