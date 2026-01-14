"""
LLM-Based Hallucination Detection
Author: Lehel Kovach | AI: Claude Opus 4.5

Strategies:
- LLM-as-judge: Single LLM call
- Self-consistency: Multiple samples
- Cross-model: Multiple LLMs
"""
from .llm_judge import (
    LLMJudgeStrategy,
    SelfConsistencyStrategy,
    CrossModelStrategy,
    mock_llm,
)
