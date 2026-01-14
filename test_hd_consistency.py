import unittest

from hd_cross_model import cross_model_agreement
from hd_self_consistency import SelfConsistencyChecker


class TestSelfConsistency(unittest.TestCase):
    def test_stable(self) -> None:
        def model(_prompt: str) -> str:
            return "Germany"

        checker = SelfConsistencyChecker(model, n=5)
        r = checker.check("Where was Einstein born?")
        self.assertEqual(r.agreement, 1.0)

    def test_unstable(self) -> None:
        outputs = ["Germany", "France", "Germany", "Austria", "France"]
        i = {"idx": 0}

        def model(_prompt: str) -> str:
            out = outputs[i["idx"]]
            i["idx"] += 1
            return out

        checker = SelfConsistencyChecker(model, n=5)
        r = checker.check("Where was Einstein born?")
        self.assertLess(r.agreement, 0.7)


class TestCrossModel(unittest.TestCase):
    def test_cross_model_agreement(self) -> None:
        def a(_p: str) -> str:
            return "Germany"

        def b(_p: str) -> str:
            return "Germany"

        def c(_p: str) -> str:
            return "France"

        report = cross_model_agreement([("a", a), ("b", b), ("c", c)], "Where was Einstein born?")
        self.assertAlmostEqual(report.agreement, 2 / 3)


if __name__ == "__main__":
    unittest.main()

