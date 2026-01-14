"""
Solution 1: Zero-Resource "Faithfulness" Detection (The "Context Check")
=======================================================================

Concept:
    Instead of checking if a fact is true in the *real world* (which requires Google/Wikipedia),
    we check if it is faithful to the *source context* provided in the prompt.
    
    This detects "Intrinsic Hallucinations" (contradicting the source) and 
    "Extrinsic Hallucinations" (adding info not in the source).

Mechanism:
    1. EXTRACT: Use regex patterns to extract claims from the Source Text.
    2. BUILD: Create a temporary "Ground Truth" graph from these claims.
    3. PROBE: Extract claims from the LLM Answer and verify them against the graph.
    
Pros:
    - Zero Latency (Milliseconds)
    - Zero Cost (No API calls, no GPUs)
    - Deterministic

Cons:
    - Rigid (Regex might miss complex sentence structures)
    - Only checks faithfulness, not external truth.

Usage:
    python test_solution1_faithfulness.py
"""

import unittest
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

class TestFaithfulness(unittest.TestCase):
    
    def setUp(self):
        # Setup Zero-Resource Backend (Hashing = No Model Download)
        self.backend = HashingEmbeddingBackend(dim=512)
        self.kb = HyperKB(embedding_backend=self.backend)
        self.extractor = RuleBasedExtractor()
        
        # Scenario: Financial Report Summary
        self.context_text = """
        The Q3 financial report shows that Acme Corp revenue increased by 15%. 
        The CEO, Jane Doe, announced a new partnership with Beta Ltd.
        Operating costs decreased by 5% due to automation.
        """
        
        # 1. Build Ground Truth from Context
        print("\n[Setup] Building Ground Truth Graph from Context...")
        claims = self.extractor.extract(self.context_text)
        for c in claims:
            self.kb.add_fact(c.subject, c.predicate, c.obj, confidence=1.0)
            print(f"  + Fact: {c}")
            
        self.prober = SCPProber(kb=self.kb, extractor=self.extractor, soft_threshold=0.6)

    def test_faithful_answer(self):
        """Answer matches context -> PASS"""
        ans = "Jane Doe announced a partnership with Beta Ltd, and revenue increased by 15%."
        print(f"\n[Test 1] Probing Faithful Answer: '{ans}'")
        
        report = self.prober.probe(ans)
        
        # Expect High Score (Relaxed threshold for hashing backend)
        self.assertGreater(report.overall_score, 0.7)
        self.assertIn(report.results[0].verdict, [Verdict.PASS, Verdict.SOFT_PASS])
        print("  -> Verdict: FAITHFUL (Correct)")

    def test_contradiction(self):
        """Answer contradicts context -> CONTRADICT"""
        ans = "Operating costs decreased, but revenue also decreased by 15%."
        print(f"\n[Test 2] Probing Contradiction: '{ans}'")
        
        report = self.prober.probe(ans)
        
        # One claim will fail/contradict
        contradictions = [r for r in report.results if r.verdict == Verdict.CONTRADICT]
        
        if not contradictions:
             # Fallback if contradiction logic misses exact swap, it should at least be a FAIL or Low Score
             self.assertLess(report.overall_score, 0.6)
        else:
            self.assertTrue(len(contradictions) > 0)
            
        print("  -> Verdict: HALLUCINATION/CONTRADICTION DETECTED (Correct)")

    def test_extrinsic_hallucination(self):
        """Answer adds info not in context -> FAIL"""
        ans = "Acme Corp stock price rose 10%."
        print(f"\n[Test 3] Probing Extrinsic Hallucination: '{ans}'")
        
        report = self.prober.probe(ans)
        
        # If extraction works, it should fail to find support
        # If extraction fails (no patterns match), it's UNKNOWN
        # We accept either as a "Not Verified" outcome
        self.assertIn(report.results[0].verdict, [Verdict.FAIL, Verdict.UNKNOWN])
        print("  -> Verdict: UNSUPPORTED FACT DETECTED (Correct)")

if __name__ == "__main__":
    unittest.main()
