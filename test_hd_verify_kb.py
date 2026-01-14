import unittest

from hd_types import Claim, Verdict
from hd_verify_kb import InMemoryKB


class TestInMemoryKBVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = InMemoryKB()
        self.kb.add_facts(
            [
                ("Albert Einstein", "born_in", "Germany"),
                ("The Eiffel Tower", "located_in", "Paris"),
            ]
        )

    def test_entailed(self) -> None:
        r = self.kb.verify(Claim("Albert Einstein", "born_in", "Germany"))
        self.assertEqual(r.verdict, Verdict.ENTAILED)

    def test_contradicted(self) -> None:
        r = self.kb.verify(Claim("Albert Einstein", "born_in", "France"))
        self.assertEqual(r.verdict, Verdict.CONTRADICTED)

    def test_not_supported(self) -> None:
        r = self.kb.verify(Claim("Marie Curie", "born_in", "Poland"))
        self.assertEqual(r.verdict, Verdict.NOT_SUPPORTED)


if __name__ == "__main__":
    unittest.main()

