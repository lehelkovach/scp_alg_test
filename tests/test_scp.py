"""
SCP (Symbolic Consistency Probing) Unit Tests
==============================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

Tests the core SCP hallucination detection algorithm.
"""

import sys
import os
import unittest

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lib.scp import HashingEmbeddingBackend, HyperKB, RuleBasedExtractor, SCPProber, Verdict

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


class TestSCP(unittest.TestCase):
    """Test cases for SCP hallucination detection."""
    
    def setUp(self) -> None:
        """Set up test KB with ground truth facts."""
        backend = HashingEmbeddingBackend(dim=512)
        self.kb = HyperKB(embedding_backend=backend)
        self.kb.add_facts_bulk(
            [
                ("The Eiffel Tower", "located_in", "Paris"),
                ("Albert Einstein", "discovered", "the theory of relativity"),
                ("Albert Einstein", "born_in", "Germany"),
                ("Marie Curie", "discovered", "radium"),
                ("Marie Curie", "born_in", "Poland"),
            ],
            source="seed_data",
            confidence=0.95,
        )
        self.prober = SCPProber(
            kb=self.kb,
            extractor=RuleBasedExtractor(),
            soft_threshold=0.70,
            contradiction_check=True,
        )

    def test_exact_pass(self) -> None:
        """Test exact match returns PASS verdict."""
        report = self.prober.probe("The Eiffel Tower is located in Paris.")
        self.assertEqual(report.results[0].verdict, Verdict.PASS)

    def test_soft_pass_paraphrase(self) -> None:
        """Test paraphrased claim returns SOFT_PASS verdict."""
        report = self.prober.probe("Einstein discovered relativity.")
        self.assertEqual(report.results[0].verdict, Verdict.SOFT_PASS)

    def test_contradiction(self) -> None:
        """Test contradicting claim returns CONTRADICT verdict."""
        report = self.prober.probe("Albert Einstein was born in France.")
        self.assertEqual(report.results[0].verdict, Verdict.CONTRADICT)

    def test_fail_unknown_fact(self) -> None:
        """Test hallucinated claim returns FAIL verdict."""
        report = self.prober.probe("Marie Curie invented the telephone.")
        self.assertEqual(report.results[0].verdict, Verdict.FAIL)


if __name__ == "__main__":
    unittest.main()
