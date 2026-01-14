"""
Hallucination Detection Solutions
=================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)

Three main approaches to detecting LLM hallucinations:

1. SCP (solutions/scp/): Knowledge base verification
   - KB mode: Pre-built knowledge base
   - Context mode: RAG faithfulness checking
   - API mode: REST service

2. Wikidata (solutions/wikidata/): External knowledge graph
   - 100M+ facts from Wikidata
   - No setup required

3. LLM (solutions/llm/): LLM-based strategies
   - LLM-as-judge
   - Self-consistency
   - Cross-model verification
"""

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"
