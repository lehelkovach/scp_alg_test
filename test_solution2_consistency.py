"""
Solution 2: "Internal Consistency" Detection (The "Self-Check")
==============================================================

Concept:
    If a model is hallucinating, it is often inconsistent with itself.
    If we ask the same question multiple times (or look at a long generated text),
    hallucinations often manifest as internal contradictions.

Mechanism:
    1. INGEST: Take a single long answer (or multiple samples).
    2. BUILD: Extract ALL claims and put them into a single Knowledge Graph.
    3. CHECK: Query the graph for self-contradictions (e.g., A is B vs A is not B).

Pros:
    - No Ground Truth required (works on "creative" or "unknown" topics).
    - Catches "Schizophrenic" generation (model changing its mind).

Cons:
    - Won't catch consistent lies (if model is consistently wrong).
    - Requires generating text (cost).

Usage:
    python test_solution2_consistency.py
"""

import unittest
from scp import HyperKB, RuleBasedExtractor, HashingEmbeddingBackend

class TestInternalConsistency(unittest.TestCase):
    
    def setUp(self):
        self.backend = HashingEmbeddingBackend(dim=512)
        self.kb = HyperKB(embedding_backend=self.backend)
        self.extractor = RuleBasedExtractor()

    def test_single_text_contradiction(self):
        """
        Detect if a single long text contradicts itself.
        Example: A story where a character's location changes impossibly.
        """
        text = """
        John was born in Paris.
        He grew up in London.
        Later in life, John was born in New York.
        """
        print(f"\n[Test] Checking Internal Consistency of: {text.strip()}")
        
        # 1. Extract all claims
        claims = self.extractor.extract(text)
        
        # 2. Add to KB
        print("  -> Building Graph...")
        for c in claims:
            # Normalize subject for test stability
            subj = c.subject
            if "John" in subj:
                subj = "John"
                
            self.kb.add_fact(subj, c.predicate, c.obj)
            print(f"     + ({subj}, {c.predicate}, {c.obj})")
            
        # 3. Check for contradictions
        # We iterate through the claims we just added and see if the KB now disagrees with them
        # (Self-check)
        contradictions_found = []
        for c in claims:
            # find_contradictions looks for (s, p, NOT o)
            # Use normalized subject
            subj = c.subject
            if "John" in subj:
                subj = "John"
            
            normalized_claim = type(c)(subj, c.predicate, c.obj, c.raw, c.confidence)
            con = self.kb.find_contradictions(normalized_claim)
            if con:
                contradictions_found.extend(con)
                
        print(f"  -> Found {len(contradictions_found)} internal contradictions.")
        
        # Expectation: "born in Paris" vs "born in New York"
        # Note: Depending on regex extraction, "grew up in" might be distinct.
        # But "born in" appears twice with different objects.
        
        self.assertTrue(len(contradictions_found) > 0)
        print("  -> Verdict: INCONSISTENT (Correct)")

if __name__ == "__main__":
    unittest.main()
