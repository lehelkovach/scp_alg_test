"""
SCP (Symbolic Consistency Probing) Solution
Author: Lehel Kovach | AI: Claude Opus 4.5

Three modes:
1. KB mode: Verify against knowledge base
2. Context mode: RAG faithfulness checking
3. API mode: REST service
"""
from .scp_prover import (
    SCPKBProver,
    check_faithfulness,
    verify_against_context,
    create_api_app,
    Verdict,
    HyperKB,
    SCPProber,
)
