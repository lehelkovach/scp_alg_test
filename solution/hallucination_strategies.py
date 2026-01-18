"""
Hallucination Detection Strategies - Efficiency Comparison
===========================================================

Ranked from MOST to LEAST efficient (latency/resources):

1. LOCAL KB (current SCP) - Zero external calls, ~10ms
2. SELF-CONSISTENCY - Same LLM, multiple samples
3. LLM-AS-JUDGE - Single call to judge model  
4. CROSS-MODEL - Compare across LLMs
5. RAG-VERIFY - Retrieve + verify

This module provides implementations and benchmarks for each.
"""

from __future__ import annotations
import json
import os
import time
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod

# Import from existing SCP implementation
from scp import (
    Claim, Verdict, HyperKB, SCPProber, RuleBasedExtractor,
    StringSimilarityBackend, EMBEDDINGS_AVAILABLE
)
if EMBEDDINGS_AVAILABLE:
    from scp import SentenceTransformerBackend


class OpenAIChatClient:
    """Minimal OpenAI chat client using standard library only."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful fact-checker. Follow the user's format exactly."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 20,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI response missing choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("OpenAI response missing content.")
        return content.strip()


def get_openai_judge_fn(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 20,
    system_prompt: Optional[str] = None,
) -> Callable[[str], str]:
    client = OpenAIChatClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        system_prompt=system_prompt,
    )
    return client.complete


class Strategy(Enum):
    """Detection strategy types."""
    LOCAL_KB = "local_kb"           # Pre-built knowledge base
    SELF_CONSISTENCY = "self_consistency"  # Same LLM, multiple samples
    LLM_AS_JUDGE = "llm_as_judge"   # Dedicated judge model
    CROSS_MODEL = "cross_model"     # Different LLM verification
    RAG_VERIFY = "rag_verify"       # Retrieval + verification


@dataclass
class StrategyResult:
    """Result from a detection strategy."""
    strategy: Strategy
    verdict: Verdict
    confidence: float
    latency_ms: float
    external_calls: int
    reasoning: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# STRATEGY 1: LOCAL KB (Most Efficient - Current SCP)
# =============================================================================

class LocalKBStrategy:
    """
    Most efficient: Zero external calls at inference time.
    
    Pros:
    - ~10ms latency (just embedding similarity)
    - No API costs at runtime
    - Deterministic results
    
    Cons:
    - Requires pre-built KB (one-time cost)
    - Limited to facts in KB
    - KB maintenance overhead
    
    Best for: High-volume, low-latency requirements
    """
    
    def __init__(self, kb: HyperKB, soft_threshold: float = 0.7):
        self.prober = SCPProber(
            kb=kb,
            extractor=RuleBasedExtractor(),
            soft_threshold=soft_threshold
        )
    
    def check(self, claim_text: str) -> StrategyResult:
        start = time.perf_counter()
        
        report = self.prober.probe(claim_text)
        
        latency = (time.perf_counter() - start) * 1000
        
        if report.results:
            result = report.results[0]
            return StrategyResult(
                strategy=Strategy.LOCAL_KB,
                verdict=result.verdict,
                confidence=result.score,
                latency_ms=latency,
                external_calls=0,
                reasoning=result.reason or "Checked against local KB",
                metadata={"matched_facts": result.matched_facts}
            )
        
        return StrategyResult(
            strategy=Strategy.LOCAL_KB,
            verdict=Verdict.UNKNOWN,
            confidence=0.0,
            latency_ms=latency,
            external_calls=0,
            reasoning="No claims extracted"
        )


# =============================================================================
# STRATEGY 2: SELF-CONSISTENCY (No KB Required)
# =============================================================================

class SelfConsistencyStrategy:
    """
    Ask the same LLM multiple times - inconsistency signals hallucination.
    
    Pros:
    - No KB required
    - Catches confident-but-wrong answers
    - Works for any claim type
    
    Cons:
    - 3-5x LLM calls
    - ~500ms+ latency
    - Doesn't catch consistent hallucinations
    
    Best for: When you have no KB and need reasonable accuracy
    """
    
    def __init__(self, llm_fn: Callable[[str], str], num_samples: int = 3):
        """
        Args:
            llm_fn: Function that takes a prompt and returns LLM response
            num_samples: Number of times to query (3-5 recommended)
        """
        self.llm_fn = llm_fn
        self.num_samples = num_samples
    
    def check(self, claim_text: str) -> StrategyResult:
        start = time.perf_counter()
        
        # Generate verification prompt
        prompt = f"""Is the following statement true or false? 
Answer with just TRUE, FALSE, or UNCERTAIN.

Statement: {claim_text}

Answer:"""
        
        # Query multiple times
        responses = []
        for _ in range(self.num_samples):
            try:
                response = self.llm_fn(prompt).strip().upper()
                responses.append(response)
            except Exception:
                responses.append("ERROR")
        
        latency = (time.perf_counter() - start) * 1000
        
        # Analyze consistency
        true_count = sum(1 for r in responses if "TRUE" in r)
        false_count = sum(1 for r in responses if "FALSE" in r)
        
        # Determine verdict based on consistency
        if true_count == self.num_samples:
            verdict = Verdict.PASS
            confidence = 1.0
            reasoning = f"Consistent TRUE across {self.num_samples} samples"
        elif false_count == self.num_samples:
            verdict = Verdict.FAIL
            confidence = 1.0
            reasoning = f"Consistent FALSE across {self.num_samples} samples"
        elif true_count > false_count:
            verdict = Verdict.SOFT_PASS
            confidence = true_count / self.num_samples
            reasoning = f"Majority TRUE ({true_count}/{self.num_samples})"
        elif false_count > true_count:
            verdict = Verdict.FAIL
            confidence = false_count / self.num_samples
            reasoning = f"Majority FALSE ({false_count}/{self.num_samples})"
        else:
            verdict = Verdict.UNKNOWN
            confidence = 0.5
            reasoning = f"Inconsistent responses: {responses}"
        
        return StrategyResult(
            strategy=Strategy.SELF_CONSISTENCY,
            verdict=verdict,
            confidence=confidence,
            latency_ms=latency,
            external_calls=self.num_samples,
            reasoning=reasoning,
            metadata={"responses": responses}
        )


# =============================================================================
# STRATEGY 3: LLM-AS-JUDGE (Single Call)
# =============================================================================

class LLMJudgeStrategy:
    """
    Use a (potentially different) LLM to judge if claim is likely true.
    
    Pros:
    - Single LLM call
    - ~200ms latency
    - Can leverage stronger judge model
    
    Cons:
    - Judge can also hallucinate
    - Requires careful prompting
    - API cost per check
    
    Best for: Quick spot-checks, moderate volume
    """
    
    def __init__(self, judge_fn: Callable[[str], str]):
        """
        Args:
            judge_fn: Function that takes a prompt and returns LLM response
        """
        self.judge_fn = judge_fn
    
    def check(self, claim_text: str) -> StrategyResult:
        start = time.perf_counter()
        
        prompt = f"""You are a fact-checker. Evaluate if this claim is factually accurate.

Claim: {claim_text}

Respond in this exact format:
VERDICT: [TRUE/FALSE/UNCERTAIN]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""
        
        try:
            response = self.judge_fn(prompt)
            external_calls = 1
            
            # Parse response
            verdict, confidence, reasoning = self._parse_response(response)
            
        except Exception as e:
            verdict = Verdict.UNKNOWN
            confidence = 0.0
            reasoning = f"Judge error: {e}"
            external_calls = 1
        
        latency = (time.perf_counter() - start) * 1000
        
        return StrategyResult(
            strategy=Strategy.LLM_AS_JUDGE,
            verdict=verdict,
            confidence=confidence,
            latency_ms=latency,
            external_calls=external_calls,
            reasoning=reasoning
        )
    
    def _parse_response(self, response: str) -> tuple:
        """Parse judge response into verdict, confidence, reasoning."""
        lines = response.strip().split('\n')
        
        verdict = Verdict.UNKNOWN
        confidence = 0.5
        reasoning = response
        
        for line in lines:
            line_upper = line.upper()
            if 'VERDICT:' in line_upper:
                if 'TRUE' in line_upper:
                    verdict = Verdict.PASS
                elif 'FALSE' in line_upper:
                    verdict = Verdict.FAIL
                else:
                    verdict = Verdict.UNKNOWN
            elif 'CONFIDENCE:' in line_upper:
                try:
                    conf_str = line.split(':')[1].strip()
                    confidence = float(conf_str)
                except (ValueError, IndexError):
                    pass
            elif 'REASONING:' in line_upper:
                reasoning = line.split(':', 1)[1].strip() if ':' in line else line
        
        return verdict, confidence, reasoning


# =============================================================================
# STRATEGY 4: CROSS-MODEL VERIFICATION
# =============================================================================

class CrossModelStrategy:
    """
    Verify claim across different LLMs - disagreement signals potential hallucination.
    
    Pros:
    - Catches model-specific hallucinations
    - More robust than single model
    - Can use cheaper models for verification
    
    Cons:
    - 2+ LLM calls
    - ~400ms latency
    - All models might share same hallucination
    
    Best for: High-stakes decisions, catching model-specific errors
    """
    
    def __init__(self, model_fns: Dict[str, Callable[[str], str]]):
        """
        Args:
            model_fns: Dict of model_name -> llm_function
        """
        self.model_fns = model_fns
    
    def check(self, claim_text: str) -> StrategyResult:
        start = time.perf_counter()
        
        prompt = f"""Is this statement factually correct? Answer TRUE or FALSE only.

Statement: {claim_text}

Answer:"""
        
        results = {}
        for model_name, model_fn in self.model_fns.items():
            try:
                response = model_fn(prompt).strip().upper()
                results[model_name] = "TRUE" in response
            except Exception:
                results[model_name] = None
        
        latency = (time.perf_counter() - start) * 1000
        
        # Analyze cross-model agreement
        valid_results = [v for v in results.values() if v is not None]
        
        if not valid_results:
            return StrategyResult(
                strategy=Strategy.CROSS_MODEL,
                verdict=Verdict.UNKNOWN,
                confidence=0.0,
                latency_ms=latency,
                external_calls=len(self.model_fns),
                reasoning="All models failed"
            )
        
        agreement = sum(valid_results) / len(valid_results)
        
        if agreement >= 0.8:
            verdict = Verdict.PASS
            reasoning = f"Cross-model agreement: {agreement:.0%} say TRUE"
        elif agreement <= 0.2:
            verdict = Verdict.FAIL
            reasoning = f"Cross-model agreement: {1-agreement:.0%} say FALSE"
        else:
            verdict = Verdict.UNKNOWN
            reasoning = f"Models disagree: {agreement:.0%} TRUE, {1-agreement:.0%} FALSE"
        
        return StrategyResult(
            strategy=Strategy.CROSS_MODEL,
            verdict=verdict,
            confidence=abs(agreement - 0.5) * 2,  # 0.5 -> 0, 1.0 -> 1.0
            latency_ms=latency,
            external_calls=len(self.model_fns),
            reasoning=reasoning,
            metadata={"model_results": results}
        )


# =============================================================================
# STRATEGY 5: HYBRID - KB + LLM FALLBACK (Recommended)
# =============================================================================

class HybridStrategy:
    """
    RECOMMENDED: Use KB first (fast), fall back to LLM only if needed.
    
    This gives you the best of both worlds:
    - Fast path: KB lookup (~10ms, 0 calls)
    - Slow path: LLM verification (~200ms, 1 call) only for unknowns
    
    Expected performance:
    - 80% of checks: KB hit -> 10ms, 0 calls
    - 20% of checks: LLM fallback -> 200ms, 1 call
    - Average: ~50ms, 0.2 calls
    """
    
    def __init__(
        self,
        kb: HyperKB,
        fallback_fn: Optional[Callable[[str], str]] = None,
        soft_threshold: float = 0.7
    ):
        self.kb_strategy = LocalKBStrategy(kb, soft_threshold)
        self.fallback_fn = fallback_fn
        self.llm_judge = LLMJudgeStrategy(fallback_fn) if fallback_fn else None
    
    def check(self, claim_text: str) -> StrategyResult:
        # Try KB first (fast path)
        kb_result = self.kb_strategy.check(claim_text)
        
        # If KB has a definitive answer, use it
        if kb_result.verdict in [Verdict.PASS, Verdict.SOFT_PASS, Verdict.CONTRADICT]:
            return kb_result
        
        # If KB doesn't know and we have a fallback, use it
        if kb_result.verdict in [Verdict.FAIL, Verdict.UNKNOWN] and self.llm_judge:
            llm_result = self.llm_judge.check(claim_text)
            
            # Combine results
            return StrategyResult(
                strategy=Strategy.LOCAL_KB,  # Report as hybrid
                verdict=llm_result.verdict,
                confidence=llm_result.confidence * 0.8,  # Discount LLM confidence
                latency_ms=kb_result.latency_ms + llm_result.latency_ms,
                external_calls=llm_result.external_calls,
                reasoning=f"KB unknown, LLM says: {llm_result.reasoning}",
                metadata={
                    "kb_result": kb_result.verdict.value,
                    "llm_result": llm_result.verdict.value
                }
            )
        
        return kb_result


# =============================================================================
# STRATEGY 6: SEMANTIC CACHE (Amortized Cost)
# =============================================================================

class SemanticCacheStrategy:
    """
    Cache LLM verification results by semantic similarity.
    
    If we've verified a similar claim before, reuse the result.
    This amortizes LLM costs over time.
    
    Pros:
    - First check: LLM cost
    - Subsequent similar checks: ~10ms, 0 calls
    - Learns from usage
    
    Cons:
    - Cold start (no cache)
    - Cache storage
    - Similar != identical (risk of wrong cache hits)
    """
    
    def __init__(
        self,
        verifier_fn: Callable[[str], str],
        similarity_threshold: float = 0.9
    ):
        self.verifier_fn = verifier_fn
        self.threshold = similarity_threshold
        self.cache: Dict[str, StrategyResult] = {}
        
        # Use embeddings for semantic matching if available
        if EMBEDDINGS_AVAILABLE:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.cache_embeddings: Dict[str, List[float]] = {}
        else:
            self.encoder = None
    
    def _get_cache_key(self, text: str) -> str:
        """Normalize text for cache key."""
        return text.lower().strip()
    
    def _find_similar_cached(self, claim_text: str) -> Optional[StrategyResult]:
        """Find semantically similar cached result."""
        if not self.cache:
            return None
        
        if self.encoder:
            # Semantic similarity via embeddings
            claim_emb = self.encoder.encode([claim_text])[0]
            
            best_score = 0.0
            best_key = None
            
            for key, cached_emb in self.cache_embeddings.items():
                score = self._cosine_sim(claim_emb, cached_emb)
                if score > best_score:
                    best_score = score
                    best_key = key
            
            if best_score >= self.threshold and best_key:
                return self.cache[best_key]
        else:
            # Fallback: exact match only
            key = self._get_cache_key(claim_text)
            return self.cache.get(key)
        
        return None
    
    def _cosine_sim(self, a, b) -> float:
        """Compute cosine similarity."""
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def check(self, claim_text: str) -> StrategyResult:
        start = time.perf_counter()
        
        # Check cache first
        cached = self._find_similar_cached(claim_text)
        if cached:
            latency = (time.perf_counter() - start) * 1000
            return StrategyResult(
                strategy=Strategy.LOCAL_KB,
                verdict=cached.verdict,
                confidence=cached.confidence * 0.95,  # Slight discount for cache
                latency_ms=latency,
                external_calls=0,
                reasoning=f"Cache hit: {cached.reasoning}",
                metadata={"cache_hit": True}
            )
        
        # Cache miss - call LLM
        judge = LLMJudgeStrategy(self.verifier_fn)
        result = judge.check(claim_text)
        
        # Store in cache
        key = self._get_cache_key(claim_text)
        self.cache[key] = result
        if self.encoder:
            self.cache_embeddings[key] = self.encoder.encode([claim_text])[0].tolist()
        
        result.metadata["cache_hit"] = False
        return result


# =============================================================================
# DEMO & BENCHMARKS
# =============================================================================

def create_sample_kb() -> HyperKB:
    """Create a sample KB for testing."""
    if EMBEDDINGS_AVAILABLE:
        backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
    else:
        backend = StringSimilarityBackend()
    
    kb = HyperKB(embedding_backend=backend)
    
    facts = [
        ("The Eiffel Tower", "located_in", "Paris"),
        ("Paris", "is_capital_of", "France"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Albert Einstein", "born_in", "Germany"),
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("Marie Curie", "discovered", "radium"),
        ("Python", "created_by", "Guido van Rossum"),
    ]
    kb.add_facts_bulk(facts, source="demo", confidence=0.95)
    return kb


def mock_llm(prompt: str) -> str:
    """Mock LLM for testing (deterministic responses)."""
    prompt_lower = prompt.lower()
    
    # Simple keyword matching for demo
    if "eiffel" in prompt_lower and "paris" in prompt_lower:
        return "VERDICT: TRUE\nCONFIDENCE: 0.95\nREASONING: The Eiffel Tower is indeed in Paris."
    elif "einstein" in prompt_lower and "relativity" in prompt_lower:
        return "VERDICT: TRUE\nCONFIDENCE: 0.90\nREASONING: Einstein developed the theory of relativity."
    elif "edison" in prompt_lower and "telephone" in prompt_lower:
        return "VERDICT: FALSE\nCONFIDENCE: 0.85\nREASONING: Alexander Graham Bell invented the telephone, not Edison."
    elif "eiffel" in prompt_lower and "london" in prompt_lower:
        return "VERDICT: FALSE\nCONFIDENCE: 0.95\nREASONING: The Eiffel Tower is in Paris, not London."
    else:
        return "VERDICT: UNCERTAIN\nCONFIDENCE: 0.50\nREASONING: Cannot verify this claim."


def resolve_judge_fn() -> Callable[[str], str]:
    """Resolve judge function from env or fall back to mock."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return get_openai_judge_fn()
        except Exception as exc:
            print(f"Warning: {exc} Falling back to mock_llm.")
    return mock_llm


def benchmark_strategies():
    """Benchmark different strategies."""
    print("=" * 70)
    print("HALLUCINATION DETECTION STRATEGY BENCHMARK")
    print("=" * 70)
    
    # Setup
    kb = create_sample_kb()
    
    judge_fn = resolve_judge_fn()
    strategies = {
        "1. Local KB (fastest)": LocalKBStrategy(kb),
        "2. LLM Judge": LLMJudgeStrategy(judge_fn),
        "3. Self-Consistency": SelfConsistencyStrategy(judge_fn, num_samples=3),
        "4. Hybrid (recommended)": HybridStrategy(kb, fallback_fn=judge_fn),
    }
    
    test_claims = [
        ("The Eiffel Tower is located in Paris.", "TRUE - exact match"),
        ("Einstein discovered relativity.", "TRUE - semantic match"),
        ("Edison invented the telephone.", "FALSE - wrong attribution"),
        ("The Eiffel Tower is in London.", "FALSE - contradiction"),
        ("Quantum computers use qubits.", "UNKNOWN - not in KB"),
    ]
    
    print("\nTest Claims:")
    for i, (claim, expected) in enumerate(test_claims, 1):
        print(f"  {i}. {claim}")
        print(f"     Expected: {expected}")
    print()
    
    # Run benchmarks
    for strategy_name, strategy in strategies.items():
        print(f"\n{'-' * 70}")
        print(f"Strategy: {strategy_name}")
        print(f"{'-' * 70}")
        
        total_latency = 0
        total_calls = 0
        
        for claim, expected in test_claims:
            result = strategy.check(claim)
            total_latency += result.latency_ms
            total_calls += result.external_calls
            
            print(f"\n  Claim: {claim[:50]}...")
            print(f"  Verdict: {result.verdict.value} (confidence: {result.confidence:.2f})")
            print(f"  Latency: {result.latency_ms:.1f}ms | External calls: {result.external_calls}")
            print(f"  Reason: {result.reasoning[:60]}...")
        
        print(f"\n  TOTALS: {total_latency:.1f}ms | {total_calls} external calls")
        print(f"  AVG: {total_latency/len(test_claims):.1f}ms/claim | {total_calls/len(test_claims):.1f} calls/claim")


def print_strategy_comparison():
    """Print strategy comparison table."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              HALLUCINATION DETECTION STRATEGY COMPARISON                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Strategy          │ Latency │ API Calls │ Accuracy │ Setup │ Best For        ║
╠═══════════════════╪═════════╪═══════════╪══════════╪═══════╪═════════════════╣
║ 1. Local KB       │ ~10ms   │ 0         │ Limited  │ High  │ High volume     ║
║ 2. LLM Judge      │ ~200ms  │ 1         │ Medium   │ None  │ Quick checks    ║
║ 3. Self-Consist   │ ~500ms  │ 3-5       │ Medium   │ None  │ No KB available ║
║ 4. Cross-Model    │ ~400ms  │ 2+        │ High     │ None  │ High stakes     ║
║ 5. RAG Verify     │ ~300ms  │ 1-2       │ High     │ Med   │ Dynamic facts   ║
║ 6. Hybrid (rec)   │ ~50ms*  │ 0.2*      │ High     │ High  │ Production      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ * Hybrid averages: 80% KB hits (10ms), 20% LLM fallback (200ms)              ║
╚══════════════════════════════════════════════════════════════════════════════╝

RECOMMENDATION BY USE CASE:
─────────────────────────────────────────────────────────────────────────────────
• High volume, low latency    → Local KB (Strategy 1)
• No setup, moderate accuracy → LLM Judge (Strategy 2)  
• Production system           → Hybrid KB + LLM fallback (Strategy 6)
• Critical decisions          → Cross-Model + Human review (Strategy 4)
• Dynamic/current events      → RAG Verification (Strategy 5)
""")


if __name__ == "__main__":
    print_strategy_comparison()
    print("\n")
    benchmark_strategies()
