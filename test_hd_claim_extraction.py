import unittest

from hd_claim_extract_rules import RuleClaimExtractor


class TestRuleClaimExtractor(unittest.TestCase):
    def test_extracts_basic_patterns(self) -> None:
        x = RuleClaimExtractor()
        claims = x.extract(
            "The Eiffel Tower is located in Paris. "
            "Albert Einstein was born in Germany. "
            "Paris is the capital of France."
        )
        self.assertEqual(len(claims), 3)
        self.assertEqual((claims[0].subject, claims[0].predicate, claims[0].obj), ("The Eiffel Tower", "located_in", "Paris"))
        self.assertEqual((claims[1].subject, claims[1].predicate, claims[1].obj), ("Albert Einstein", "born_in", "Germany"))
        self.assertEqual((claims[2].subject, claims[2].predicate, claims[2].obj), ("Paris", "capital_of", "France"))

    def test_skips_empty(self) -> None:
        x = RuleClaimExtractor()
        self.assertEqual(x.extract(""), [])


if __name__ == "__main__":
    unittest.main()

