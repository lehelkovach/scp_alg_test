import unittest

from scp import HashingEmbeddingBackend, HyperKB, RuleBasedExtractor, SCPProber, Verdict


class TestSCP(unittest.TestCase):
    def setUp(self) -> None:
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
        report = self.prober.probe("The Eiffel Tower is located in Paris.")
        self.assertEqual(report.results[0].verdict, Verdict.PASS)

    def test_soft_pass_paraphrase(self) -> None:
        report = self.prober.probe("Einstein discovered relativity.")
        self.assertEqual(report.results[0].verdict, Verdict.SOFT_PASS)

    def test_contradiction(self) -> None:
        report = self.prober.probe("Albert Einstein was born in France.")
        self.assertEqual(report.results[0].verdict, Verdict.CONTRADICT)

    def test_fail_unknown_fact(self) -> None:
        report = self.prober.probe("Marie Curie invented the telephone.")
        self.assertEqual(report.results[0].verdict, Verdict.FAIL)


if __name__ == "__main__":
    unittest.main()

