"""
Solution 3: GraphRAG Verification (The "Persistent Memory" Check)
================================================================

Concept:
    The "Gold Standard". Instead of testing the LLM's output against *itself* or *context*,
    we test it against a curated, persistent "System of Record" (The Graph).
    
    This uses the FastAPI Microservice approach (`graph_service.py`) but simulated here as a test.

Mechanism:
    1. LOAD: Load the persistent 'knowledge_graph.json' (The "Brain").
    2. PROBE: Check the answer against this trusted knowledge.
    
Pros:
    - High Accuracy (verified against curated facts).
    - Can verify "World Truth" (if the Graph is populated with World Truth).
    - Persistent (learns over time).

Cons:
    - Needs a populated Graph first (the "Cold Start" problem).

Usage:
    python test_solution3_graphrag.py
"""

import unittest
import os
import json
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

class TestGraphRAGVerification(unittest.TestCase):
    
    def setUp(self):
        self.db_file = "test_knowledge_graph.json"
        
        # 1. Simulate a Populated "System of Record"
        self.backend = HashingEmbeddingBackend(dim=512)
        self.kb = HyperKB(embedding_backend=self.backend)
        
        # Seed it with "World Knowledge"
        facts = [
            ("Python", "created_by", "Guido van Rossum"),
            ("Python", "released_in", "1991"),
            ("Java", "created_by", "James Gosling"),
        ]
        self.kb.add_facts_bulk(facts, source="wikipedia_dump")
        
        # Save to disk to simulate persistence
        self.kb.save_to_disk(self.db_file)
        
        # 2. Initialize Service (Load from disk)
        self.service_kb = HyperKB(embedding_backend=self.backend)
        self.service_kb.load_from_disk(self.db_file)
        self.prober = SCPProber(kb=self.service_kb, extractor=RuleBasedExtractor())

    def tearDown(self):
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

    def test_verify_against_db(self):
        """Verify an answer against the persistent DB"""
        
        # Correct Answer
        ans_good = "Python was created by Guido van Rossum."
        print(f"\n[Test 1] Verifying: '{ans_good}'")
        report = self.prober.probe(ans_good)
        
        # Hashing backend might only get SOFT_PASS if "was created by" != "created_by" exactly
        # So we accept SOFT_PASS
        self.assertIn(report.results[0].verdict, [Verdict.PASS, Verdict.SOFT_PASS])
        print("  -> Verdict: VERIFIED (Correct)")
        
        # Hallucination (Wrong Creator)
        ans_bad = "Python was created by Elon Musk."
        print(f"\n[Test 2] Verifying: '{ans_bad}'")
        report = self.prober.probe(ans_bad)
        
        # Should detect contradiction or fail to find support
        # Since "Python created_by" exists, this is a CONTRADICTION or mismatch
        self.assertIn(report.results[0].verdict, [Verdict.CONTRADICT, Verdict.FAIL])
        print("  -> Verdict: HALLUCINATION DETECTED (Correct)")

if __name__ == "__main__":
    unittest.main()
