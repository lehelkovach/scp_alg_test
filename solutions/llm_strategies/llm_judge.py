"""
LLM-Based Hallucination Detection Strategies
=============================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Multiple strategies for detecting hallucinations using LLMs themselves.

STRATEGIES:
1. LLM-AS-JUDGE: Single call, ~200ms, medium accuracy
2. SELF-CONSISTENCY: 3-5 calls, catches inconsistency
3. CROSS-MODEL: 2+ LLMs, catches model-specific errors
4. SEMANTIC CACHE: Amortized cost via similarity caching

USAGE:
    from solutions.llm_strategies import LLMJudgeStrategy
    
    judge = LLMJudgeStrategy(my_llm_fn)
    result = judge.check("Edison invented the telephone")

Dependencies:
    None (bring your own LLM function)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.hallucination_strategies import (
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
    'Strategy', 'StrategyResult', 'LocalKBStrategy',
    'SelfConsistencyStrategy', 'LLMJudgeStrategy',
    'CrossModelStrategy', 'HybridStrategy',
    'SemanticCacheStrategy', 'mock_llm',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate LLM-based strategies."""
    print("=" * 70)
    print("LLM HALLUCINATION DETECTION STRATEGIES")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    judge = LLMJudgeStrategy(mock_llm)
    
    test_claims = [
        "The Eiffel Tower is located in Paris.",
        "Edison invented the telephone.",
    ]
    
    for claim in test_claims:
        result = judge.check(claim)
        print(f"Claim: {claim}")
        print(f"  Verdict: {result.verdict.value}")
        print()


if __name__ == "__main__":
    demo()
