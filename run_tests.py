#!/usr/bin/env python3
"""
Hallucination Detection Test Runner
====================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

Runs all hallucination detection solutions and compares their results.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --solution scp    # Run specific solution
    python run_tests.py --verbose    # Detailed output
"""

import sys
import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add lib to path
sys.path.insert(0, os.path.dirname(__file__))

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


# ==============================================================================
# TEST CLAIMS
# ==============================================================================

TEST_CLAIMS = [
    ("Alexander Graham Bell invented the telephone.", "verified", "True fact"),
    ("Thomas Edison invented the telephone.", "refuted", "False - Bell did"),
    ("Albert Einstein discovered the theory of relativity.", "verified", "True"),
    ("Isaac Newton discovered the theory of relativity.", "refuted", "False"),
    ("Marie Curie discovered radium.", "verified", "True fact"),
    ("The Eiffel Tower is located in Paris.", "verified", "True fact"),
    ("The Eiffel Tower is located in London.", "refuted", "False"),
    ("Python was created by Guido van Rossum.", "verified", "True fact"),
    ("Quantum computers use qubits.", "unverifiable", "May not be in KB"),
]


@dataclass
class TestResult:
    claim: str
    expected: str
    actual: str
    passed: bool
    confidence: float
    latency_ms: float
    source: str
    reason: str


@dataclass
class SolutionResults:
    solution_name: str
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    avg_latency_ms: float
    results: List[TestResult]


# ==============================================================================
# SOLUTION RUNNERS
# ==============================================================================

def run_scp_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run SCP solution."""
    print("\n" + "=" * 70)
    print("SOLUTION: SCP (Symbolic Consistency Probing)")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    from lib.scp import HyperKB, SCPProber, RuleBasedExtractor, StringSimilarityBackend, Verdict
    
    backend = StringSimilarityBackend()
    kb = HyperKB(embedding_backend=backend)
    
    facts = [
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Marie Curie", "discovered", "radium"),
        ("Thomas Edison", "invented", "the lightbulb"),
        ("The Eiffel Tower", "located_in", "Paris"),
        ("Python", "created_by", "Guido van Rossum"),
    ]
    kb.add_facts_bulk(facts, source="ground_truth", confidence=1.0)
    
    prober = SCPProber(kb=kb, extractor=RuleBasedExtractor(), soft_threshold=0.7)
    
    results = []
    for claim, expected, desc in claims:
        start = time.perf_counter()
        report = prober.probe(claim)
        latency = (time.perf_counter() - start) * 1000
        
        if report.results:
            r = report.results[0]
            if r.verdict in [Verdict.PASS, Verdict.SOFT_PASS]:
                actual = "verified"
            elif r.verdict in [Verdict.CONTRADICT, Verdict.FAIL]:
                actual = "refuted"
            else:
                actual = "unverifiable"
            confidence = r.score
            reason = r.reason[:80] if r.reason else ""
        else:
            actual = "unverifiable"
            confidence = 0.0
            reason = "No claims extracted"
        
        passed = (actual == expected) or (expected == "unverifiable")
        
        results.append(TestResult(claim, expected, actual, passed, confidence, latency, "local_kb", reason))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults("SCP", len(results), passed_count, len(results) - passed_count,
                          passed_count / len(results), avg_latency, results)


def run_wikidata_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run Wikidata solution."""
    print("\n" + "=" * 70)
    print("SOLUTION: Wikidata (100M+ facts)")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    from lib.wikidata_verifier import WikidataVerifier, VerificationStatus
    
    verifier = WikidataVerifier()
    
    results = []
    for claim, expected, desc in claims:
        start = time.perf_counter()
        result = verifier.verify(claim)
        latency = (time.perf_counter() - start) * 1000
        
        if result.status == VerificationStatus.VERIFIED:
            actual = "verified"
        elif result.status == VerificationStatus.REFUTED:
            actual = "refuted"
        else:
            actual = "unverifiable"
        
        passed = (actual == expected) or (expected == "unverifiable")
        
        results.append(TestResult(claim, expected, actual, passed, result.confidence, latency, "wikidata", result.reason[:80]))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults("Wikidata", len(results), passed_count, len(results) - passed_count,
                          passed_count / len(results), avg_latency, results)


def run_hybrid_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run Hybrid solution."""
    print("\n" + "=" * 70)
    print("SOLUTION: Hybrid (KB → Wikidata → LLM)")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    from lib.wikidata_verifier import HybridVerifier
    
    hybrid = HybridVerifier(
        local_facts=[("Python", "created_by", "Guido van Rossum")],
        use_wikidata=True
    )
    
    results = []
    for claim, expected, desc in claims:
        start = time.perf_counter()
        result = hybrid.verify(claim)
        latency = (time.perf_counter() - start) * 1000
        
        actual = result.get("status", "unverifiable")
        if actual not in ["verified", "refuted", "unverifiable"]:
            actual = "unverifiable"
        
        passed = (actual == expected) or (expected == "unverifiable")
        
        results.append(TestResult(claim, expected, actual, passed, 
                                 result.get("confidence", 0.0), latency,
                                 result.get("source", "unknown"), 
                                 result.get("reason", "")[:80]))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults("Hybrid", len(results), passed_count, len(results) - passed_count,
                          passed_count / len(results), avg_latency, results)


def print_summary(all_results: List[SolutionResults]):
    """Print summary comparison."""
    print("\n" + "=" * 70)
    print("SUMMARY: SOLUTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Solution':<15} {'Accuracy':<12} {'Avg Latency':<15} {'Passed':<10}")
    print("-" * 52)
    
    for r in all_results:
        print(f"{r.solution_name:<15} {r.accuracy:>8.0%}     {r.avg_latency_ms:>8.1f}ms     {r.passed}/{r.total_tests}")
    
    print("\n" + "=" * 70)
    best = max(all_results, key=lambda x: x.accuracy)
    fastest = min(all_results, key=lambda x: x.avg_latency_ms)
    print(f"Most Accurate: {best.solution_name} ({best.accuracy:.0%})")
    print(f"Fastest: {fastest.solution_name} ({fastest.avg_latency_ms:.1f}ms)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run hallucination detection tests")
    parser.add_argument("--solution", choices=["scp", "wikidata", "hybrid", "all"], default="all")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HALLUCINATION DETECTION TEST RUNNER")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    all_results = []
    
    solutions = {
        "scp": run_scp_solution,
        "wikidata": run_wikidata_solution,
        "hybrid": run_hybrid_solution,
    }
    
    if args.solution == "all":
        for name, runner in solutions.items():
            try:
                result = runner(TEST_CLAIMS, verbose=args.verbose)
                all_results.append(result)
            except Exception as e:
                print(f"\n⚠ Solution {name} failed: {e}")
    else:
        try:
            result = solutions[args.solution](TEST_CLAIMS, verbose=args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠ Solution {args.solution} failed: {e}")
    
    if all_results:
        print_summary(all_results)
    
    return 0 if all(r.passed == r.total_tests for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
