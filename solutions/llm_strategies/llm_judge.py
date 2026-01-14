"""
LLM-Based Hallucination Detection Strategies
=============================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Multiple strategies for detecting hallucinations using LLMs themselves.
Each strategy trades off speed, cost, and accuracy differently.

STRATEGIES IMPLEMENTED:

1. LLM-AS-JUDGE (Single Call)
   ─────────────────────────
   How: Ask an LLM "Is this claim true?"
   Speed: ~200ms (1 API call)
   Accuracy: Medium (judge can also hallucinate)
   Best for: Quick spot-checks
   
   Example:
       judge = LLMJudgeStrategy(llm_fn)
       result = judge.check("Edison invented telephone")

2. SELF-CONSISTENCY (Same LLM, Multiple Samples)
   ─────────────────────────────────────────────
   How: Ask same LLM 3-5 times, check if answers are consistent
   Speed: ~500ms (3-5 API calls)
   Accuracy: Medium (catches inconsistency, not confident hallucinations)
   Best for: When you have no KB
   
   Example:
       checker = SelfConsistencyStrategy(llm_fn, num_samples=3)
       result = checker.check("Edison invented telephone")
       # If LLM says TRUE, FALSE, TRUE → inconsistent → likely hallucination

3. CROSS-MODEL VERIFICATION
   ─────────────────────────
   How: Ask multiple different LLMs, compare answers
   Speed: ~400ms (2+ API calls in parallel)
   Accuracy: High (catches model-specific hallucinations)
   Best for: High-stakes decisions
   
   Example:
       checker = CrossModelStrategy({
           "gpt4": gpt4_fn,
           "claude": claude_fn,
           "llama": llama_fn
       })
       result = checker.check("Edison invented telephone")
       # If all agree FALSE → high confidence refutation

4. SEMANTIC CACHE
   ───────────────
   How: Cache LLM verifications by semantic similarity
   Speed: ~10ms (cache hit), ~200ms (cache miss)
   Accuracy: Depends on underlying strategy
   Best for: Repeated/similar queries
   
   Example:
       cache = SemanticCacheStrategy(verifier_fn, similarity_threshold=0.9)
       result1 = cache.check("Bell invented telephone")  # ~200ms (miss)
       result2 = cache.check("Bell invented the phone")  # ~10ms (hit, similar)

USAGE:
    from llm_judge import LLMJudgeStrategy, SelfConsistencyStrategy
    
    # Your LLM function
    def my_llm(prompt):
        return call_openai(prompt)
    
    judge = LLMJudgeStrategy(my_llm)
    result = judge.check("Edison invented the telephone")

Dependencies:
    None (bring your own LLM function)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hallucination_strategies import (
    Strategy,
    StrategyResult,
    LocalKBStrategy,
    SelfConsistencyStrategy,
    LLMJudgeStrategy,
    CrossModelStrategy,
    HybridStrategy,
    SemanticCacheStrategy,
    create_sample_kb,
    mock_llm,
)

__all__ = [
    'Strategy',
    'StrategyResult',
    'LocalKBStrategy',
    'SelfConsistencyStrategy',
    'LLMJudgeStrategy',
    'CrossModelStrategy',
    'HybridStrategy',
    'SemanticCacheStrategy',
    'mock_llm',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate LLM-based hallucination detection strategies."""
    print("=" * 70)
    print("LLM HALLUCINATION DETECTION STRATEGIES DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    test_claims = [
        "The Eiffel Tower is located in Paris.",
        "Edison invented the telephone.",
        "Einstein discovered relativity.",
    ]
    
    # Demo LLM Judge
    print("\n--- LLM-AS-JUDGE STRATEGY ---")
    judge = LLMJudgeStrategy(mock_llm)
    for claim in test_claims:
        result = judge.check(claim)
        print(f"Claim: {claim}")
        print(f"  Verdict: {result.verdict.value}")
        print(f"  Calls: {result.external_calls}")
        print()
    
    # Demo Self-Consistency
    print("\n--- SELF-CONSISTENCY STRATEGY ---")
    consistency = SelfConsistencyStrategy(mock_llm, num_samples=3)
    for claim in test_claims:
        result = consistency.check(claim)
        print(f"Claim: {claim}")
        print(f"  Verdict: {result.verdict.value}")
        print(f"  Calls: {result.external_calls}")
        print()


if __name__ == "__main__":
    demo()
