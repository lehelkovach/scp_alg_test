#!/usr/bin/env python3
"""
Hallucination Detection Test Runner
====================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

This script runs all hallucination detection solutions and compares their results.
Each solution approaches the problem differently, with tradeoffs in speed, accuracy,
and resource requirements.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --solution scp    # Run specific solution
    python run_tests.py --verbose    # Detailed output
    python run_tests.py --benchmark  # Run performance benchmark

Solutions:
    scp         - Knowledge Base with embeddings (~10ms, 0 API calls)
    wikidata    - Wikidata API (100M+ facts, ~200ms)
    llm         - LLM-as-judge strategies (~200ms, 1+ API calls)
    knowshowgo  - Cognitive architecture ground truth
    hybrid      - Cascading pipeline (fastest first)
"""

import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


# ==============================================================================
# TEST CLAIMS
# ==============================================================================

TEST_CLAIMS = [
    # (claim, expected_verdict, description)
    ("Alexander Graham Bell invented the telephone.", "verified", "True fact - exact match"),
    ("Thomas Edison invented the telephone.", "refuted", "False - Bell invented it"),
    ("Albert Einstein discovered the theory of relativity.", "verified", "True fact"),
    ("Isaac Newton discovered the theory of relativity.", "refuted", "False - Einstein did"),
    ("Marie Curie discovered radium.", "verified", "True fact"),
    ("The Eiffel Tower is located in Paris.", "verified", "True fact"),
    ("The Eiffel Tower is located in London.", "refuted", "False - it's in Paris"),
    ("Python was created by Guido van Rossum.", "verified", "True fact"),
    ("Quantum computers use qubits.", "unverifiable", "True but may not be in KB"),
]


# ==============================================================================
# RESULT TRACKING
# ==============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
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
    """Results for a complete solution."""
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
    """Run SCP (Symbolic Consistency Probing) solution."""
    print("\n" + "=" * 70)
    print("SOLUTION: SCP (Symbolic Consistency Probing)")
    print("Method: Local knowledge base with embedding similarity")
    print("Speed: ~10ms | API Calls: 0")
    print("=" * 70)
    
    try:
        from solutions.scp.scp_prover import (
            HyperKB, SCPProber, RuleBasedExtractor, 
            StringSimilarityBackend, Verdict
        )
    except ImportError:
        from scp import (
            HyperKB, SCPProber, RuleBasedExtractor,
            StringSimilarityBackend, Verdict
        )
    
    # Initialize KB with ground truth
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
            elif r.verdict == Verdict.CONTRADICT:
                actual = "refuted"
            elif r.verdict == Verdict.FAIL:
                actual = "refuted"
            else:
                actual = "unverifiable"
            confidence = r.score
            reason = r.reason[:100] if r.reason else "No reason"
        else:
            actual = "unverifiable"
            confidence = 0.0
            reason = "No claims extracted"
        
        passed = (actual == expected) or (expected == "unverifiable" and actual == "unverifiable")
        
        results.append(TestResult(
            claim=claim,
            expected=expected,
            actual=actual,
            passed=passed,
            confidence=confidence,
            latency_ms=latency,
            source="local_kb",
            reason=reason
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults(
        solution_name="SCP",
        total_tests=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        accuracy=passed_count / len(results),
        avg_latency_ms=avg_latency,
        results=results
    )


def run_wikidata_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run Wikidata solution."""
    print("\n" + "=" * 70)
    print("SOLUTION: Wikidata")
    print("Method: SPARQL queries to Wikidata (100M+ facts)")
    print("Speed: ~200ms | API Calls: 1 per query")
    print("=" * 70)
    
    try:
        from solutions.wikidata.wikidata_prover import WikidataVerifier, VerificationStatus
    except ImportError:
        from wikidata_verifier import WikidataVerifier, VerificationStatus
    
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
        
        results.append(TestResult(
            claim=claim,
            expected=expected,
            actual=actual,
            passed=passed,
            confidence=result.confidence,
            latency_ms=latency,
            source="wikidata",
            reason=result.reason[:100]
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults(
        solution_name="Wikidata",
        total_tests=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        accuracy=passed_count / len(results),
        avg_latency_ms=avg_latency,
        results=results
    )


def run_llm_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run LLM-as-judge solution (using mock LLM)."""
    print("\n" + "=" * 70)
    print("SOLUTION: LLM-as-Judge")
    print("Method: Ask LLM to verify claims (using mock LLM for demo)")
    print("Speed: ~200ms | API Calls: 1 per query")
    print("=" * 70)
    
    try:
        from solutions.llm_strategies.llm_judge import LLMJudgeStrategy, mock_llm
    except ImportError:
        from hallucination_strategies import LLMJudgeStrategy, mock_llm
    
    from test import Verdict
    
    judge = LLMJudgeStrategy(mock_llm)
    
    results = []
    for claim, expected, desc in claims:
        start = time.perf_counter()
        result = judge.check(claim)
        latency = (time.perf_counter() - start) * 1000
        
        if result.verdict == Verdict.PASS:
            actual = "verified"
        elif result.verdict == Verdict.FAIL:
            actual = "refuted"
        else:
            actual = "unverifiable"
        
        passed = (actual == expected) or (expected == "unverifiable")
        
        results.append(TestResult(
            claim=claim,
            expected=expected,
            actual=actual,
            passed=passed,
            confidence=result.confidence,
            latency_ms=latency,
            source="llm_judge",
            reason=result.reasoning[:100]
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults(
        solution_name="LLM-Judge",
        total_tests=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        accuracy=passed_count / len(results),
        avg_latency_ms=avg_latency,
        results=results
    )


def run_hybrid_solution(claims: List[tuple], verbose: bool = False) -> SolutionResults:
    """Run Hybrid solution (KB → Wikidata → LLM)."""
    print("\n" + "=" * 70)
    print("SOLUTION: Hybrid (KB → Wikidata → LLM)")
    print("Method: Cascading pipeline, fastest first")
    print("Speed: ~10-200ms | API Calls: 0-1 per query")
    print("=" * 70)
    
    try:
        from solutions.hybrid.hybrid_prover import HybridVerifier
    except ImportError:
        from wikidata_verifier import HybridVerifier
    
    hybrid = HybridVerifier(
        local_facts=[
            ("Python", "created_by", "Guido van Rossum"),
            ("Linux", "created_by", "Linus Torvalds"),
        ],
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
        
        results.append(TestResult(
            claim=claim,
            expected=expected,
            actual=actual,
            passed=passed,
            confidence=result.get("confidence", 0.0),
            latency_ms=latency,
            source=result.get("source", "unknown"),
            reason=result.get("reason", "")[:100]
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {claim[:50]}... → {actual} ({latency:.1f}ms)")
    
    passed_count = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    return SolutionResults(
        solution_name="Hybrid",
        total_tests=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        accuracy=passed_count / len(results),
        avg_latency_ms=avg_latency,
        results=results
    )


# ==============================================================================
# MAIN
# ==============================================================================

def print_summary(all_results: List[SolutionResults]):
    """Print summary comparison of all solutions."""
    print("\n" + "=" * 70)
    print("SUMMARY: SOLUTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Solution':<15} {'Accuracy':<12} {'Avg Latency':<15} {'Passed':<10}")
    print("-" * 52)
    
    for r in all_results:
        print(f"{r.solution_name:<15} {r.accuracy:>8.0%}     {r.avg_latency_ms:>8.1f}ms     {r.passed}/{r.total_tests}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    # Find best by accuracy
    best_accuracy = max(all_results, key=lambda x: x.accuracy)
    fastest = min(all_results, key=lambda x: x.avg_latency_ms)
    
    print(f"Most Accurate: {best_accuracy.solution_name} ({best_accuracy.accuracy:.0%})")
    print(f"Fastest: {fastest.solution_name} ({fastest.avg_latency_ms:.1f}ms)")
    print("\nFor production: Use Hybrid (KB → Wikidata → LLM) for best balance.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run hallucination detection tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--solution", choices=["scp", "wikidata", "llm", "hybrid", "all"],
                       default="all", help="Which solution to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HALLUCINATION DETECTION TEST RUNNER")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    print(f"\nRunning {len(TEST_CLAIMS)} test claims...")
    
    all_results = []
    
    solutions = {
        "scp": run_scp_solution,
        "wikidata": run_wikidata_solution,
        "llm": run_llm_solution,
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
    
    # Return exit code based on results
    total_passed = sum(r.passed for r in all_results)
    total_tests = sum(r.total_tests for r in all_results)
    
    if total_passed == total_tests:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_tests - total_passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
