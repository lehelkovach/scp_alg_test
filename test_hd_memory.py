import unittest

from hd_memory import ProvenanceStore, VerificationCache, cache_key, CachedVerification
from hd_types import Claim, VerificationResult, Verdict, EvidenceSpan


class TestMemory(unittest.TestCase):
    def test_cache_key_stability(self) -> None:
        claim1 = Claim("Albert Einstein", "born_in", "Germany")
        claim2 = Claim("  ALBERT  EINSTEIN ", "born_in", "GERMANY")

        k1 = cache_key(claim1, corpus_version="v1", verifier_version="ver1")
        k2 = cache_key(claim2, corpus_version="v1", verifier_version="ver1")
        self.assertEqual(k1, k2)

    def test_cache_roundtrip(self) -> None:
        claim = Claim("Albert Einstein", "born_in", "Germany")
        res = VerificationResult(
            claim=claim,
            verdict=Verdict.ENTAILED,
            confidence=1.0,
            reason="demo",
            evidence=(EvidenceSpan(doc_id="bio", text="Einstein was born in Germany."),),
        )

        key = cache_key(claim, corpus_version="v1", verifier_version="ver1")
        cache = VerificationCache()
        cache.set(key, CachedVerification(result=res, corpus_version="v1", verifier_version="ver1"))

        got = cache.get(key)
        self.assertIsNotNone(got)
        self.assertEqual(got.result.verdict, Verdict.ENTAILED)

    def test_provenance_store_lookup(self) -> None:
        store = ProvenanceStore()
        claim = Claim("Albert Einstein", "born_in", "Germany")
        res = VerificationResult(claim=claim, verdict=Verdict.ENTAILED, confidence=1.0)
        store.upsert(res)

        by_claim = store.get(claim)
        self.assertIsNotNone(by_claim)
        self.assertEqual(by_claim.verdict, Verdict.ENTAILED)

        by_subj = store.lookup_by_subject("Albert Einstein")
        self.assertEqual(len(by_subj), 1)


if __name__ == "__main__":
    unittest.main()

