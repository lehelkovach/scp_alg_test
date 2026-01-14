import unittest

from hd_types import Claim, EvidenceSpan, Verdict
from hd_verify_evidence import EvidenceVerifier


class TestEvidenceVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.v = EvidenceVerifier()

    def test_entailed(self) -> None:
        claim = Claim("The Eiffel Tower", "located_in", "Paris")
        evidence = [
            EvidenceSpan(doc_id="doc:1", text="The Eiffel Tower is located in Paris and was built in 1889.")
        ]
        r = self.v.verify(claim, evidence)
        self.assertEqual(r.verdict, Verdict.ENTAILED)
        self.assertTrue(r.evidence)
        self.assertEqual(r.evidence[0].doc_id, "doc:1")

    def test_contradicted(self) -> None:
        claim = Claim("Albert Einstein", "born_in", "France")
        evidence = [
            EvidenceSpan(doc_id="bio", text="Albert Einstein was born in Germany.")
        ]
        r = self.v.verify(claim, evidence)
        self.assertEqual(r.verdict, Verdict.CONTRADICTED)

    def test_not_supported(self) -> None:
        claim = Claim("Albert Einstein", "born_in", "Germany")
        evidence = [
            EvidenceSpan(doc_id="misc", text="Albert Einstein is a famous physicist.")
        ]
        r = self.v.verify(claim, evidence)
        self.assertEqual(r.verdict, Verdict.NOT_SUPPORTED)


if __name__ == "__main__":
    unittest.main()

