"""
Wikidata-Powered Hallucination Detection
=========================================

Uses Wikidata (100M+ facts) as ground truth for verification.
No training needed - instant access to structured knowledge.

This is the FASTEST path to production-quality hallucination detection.

Usage:
    verifier = WikidataVerifier()
    result = verifier.verify("Edison invented the telephone")
    # Returns: REFUTED, reason="Alexander Graham Bell invented telephone"
"""

import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class VerificationStatus(Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


@dataclass
class WikidataResult:
    """Result from Wikidata verification."""
    claim: str
    status: VerificationStatus
    confidence: float
    wikidata_facts: List[Dict]
    reason: str
    query_time_ms: float
    
    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "confidence": self.confidence,
            "wikidata_facts": self.wikidata_facts,
            "reason": self.reason,
            "query_time_ms": self.query_time_ms
        }


class WikidataVerifier:
    """
    Verify claims against Wikidata knowledge graph.
    
    Wikidata contains 100M+ items with structured facts:
    - People, places, organizations
    - Inventions, discoveries, events
    - Relationships, properties
    
    All with provenance (references) built-in!
    """
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    # Common predicates mapping
    PREDICATE_MAP = {
        "invented": "P61",      # discoverer or inventor
        "discovered": "P61",    # discoverer or inventor
        "created": "P170",      # creator
        "founded": "P112",      # founded by
        "born in": "P19",       # place of birth
        "born": "P569",         # date of birth
        "died": "P570",         # date of death
        "located in": "P131",   # located in administrative entity
        "capital of": "P36",    # capital
        "wrote": "P50",         # author
        "directed": "P57",      # director
        "CEO of": "P169",       # chief executive officer
        "spouse": "P26",        # spouse
    }
    
    def __init__(self, cache_file: str = "./wikidata_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, WikidataResult] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                # Convert back to WikidataResult objects
                for k, v in data.items():
                    self.cache[k] = WikidataResult(
                        claim=v["claim"],
                        status=VerificationStatus(v["status"]),
                        confidence=v["confidence"],
                        wikidata_facts=v["wikidata_facts"],
                        reason=v["reason"],
                        query_time_ms=v["query_time_ms"]
                    )
        except FileNotFoundError:
            pass
    
    def _save_cache(self):
        """Save cache to disk."""
        data = {k: v.to_dict() for k, v in self.cache.items()}
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _sparql_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query against Wikidata."""
        url = f"{self.SPARQL_ENDPOINT}?query={urllib.parse.quote(query)}"
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "HallucinationDetector/1.0"
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"Wikidata query error: {e}")
            return []
    
    def _search_entity(self, name: str) -> Optional[str]:
        """Search for Wikidata entity ID by name."""
        # Use Wikidata search API
        search_url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbsearchentities&search={urllib.parse.quote(name)}"
            f"&language=en&format=json"
        )
        
        try:
            with urllib.request.urlopen(search_url, timeout=5) as response:
                data = json.loads(response.read().decode())
                results = data.get("search", [])
                if results:
                    return results[0]["id"]  # e.g., "Q317521" for Elon Musk
        except Exception:
            pass
        
        return None
    
    def _get_entity_claims(self, entity_id: str, property_id: str) -> List[Dict]:
        """Get claims for an entity property."""
        query = f"""
        SELECT ?value ?valueLabel WHERE {{
          wd:{entity_id} wdt:{property_id} ?value.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 10
        """
        return self._sparql_query(query)
    
    def _find_inventor(self, item_name: str) -> List[Dict]:
        """Find who invented/discovered something."""
        query = f"""
        SELECT ?inventor ?inventorLabel ?item ?itemLabel WHERE {{
          ?item rdfs:label "{item_name}"@en.
          ?item wdt:P61 ?inventor.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 5
        """
        return self._sparql_query(query)
    
    def _verify_invention(self, subject: str, obj: str) -> Tuple[VerificationStatus, float, List[Dict], str]:
        """Verify an invention/discovery claim."""
        # Search for the object (the invention)
        results = self._find_inventor(obj)
        
        if not results:
            # Try with "the" prefix variations
            for variant in [obj, f"the {obj}", obj.replace("the ", "")]:
                results = self._find_inventor(variant)
                if results:
                    break
        
        if not results:
            return (
                VerificationStatus.UNVERIFIABLE,
                0.5,
                [],
                f"Could not find '{obj}' in Wikidata"
            )
        
        # Check if claimed subject matches
        inventors = [r.get("inventorLabel", {}).get("value", "").lower() for r in results]
        subject_lower = subject.lower()
        
        # Check for match
        for inventor in inventors:
            if subject_lower in inventor or inventor in subject_lower:
                return (
                    VerificationStatus.VERIFIED,
                    0.95,
                    results,
                    f"Wikidata confirms: {inventor} invented {obj}"
                )
        
        # No match - it's a refutation
        actual_inventor = results[0].get("inventorLabel", {}).get("value", "Unknown")
        return (
            VerificationStatus.REFUTED,
            0.90,
            results,
            f"Wikidata shows {actual_inventor} invented {obj}, not {subject}"
        )
    
    def _extract_claim_parts(self, claim: str) -> Optional[Tuple[str, str, str]]:
        """Extract subject, predicate, object from claim."""
        import re
        
        patterns = [
            r"^(.+?)\s+(invented|discovered|created|founded|wrote)\s+(.+?)\.?$",
            r"^(.+?)\s+(was born in|is located in|is the capital of)\s+(.+?)\.?$",
            r"^(.+?)\s+(is|was)\s+(.+?)\.?$",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, claim, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip().lower(), match.group(3).strip()
        
        return None
    
    def verify(self, claim: str) -> WikidataResult:
        """
        Verify a claim against Wikidata.
        
        Examples:
            verify("Edison invented the telephone")
            → REFUTED: "Alexander Graham Bell invented telephone"
            
            verify("Einstein discovered relativity")
            → VERIFIED: "Wikidata confirms Einstein discovered relativity"
        """
        # Check cache first
        cache_key = claim.lower().strip()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached.query_time_ms = 0  # Cache hit
            return cached
        
        start = time.perf_counter()
        
        # Extract claim parts
        parts = self._extract_claim_parts(claim)
        
        if not parts:
            result = WikidataResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                wikidata_facts=[],
                reason="Could not parse claim structure",
                query_time_ms=(time.perf_counter() - start) * 1000
            )
            return result
        
        subject, predicate, obj = parts
        
        # Route to appropriate verification method
        if predicate in ["invented", "discovered", "created"]:
            status, confidence, facts, reason = self._verify_invention(subject, obj)
        else:
            # Generic verification (TODO: implement more predicates)
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.5
            facts = []
            reason = f"Predicate '{predicate}' not yet supported"
        
        query_time = (time.perf_counter() - start) * 1000
        
        result = WikidataResult(
            claim=claim,
            status=status,
            confidence=confidence,
            wikidata_facts=facts,
            reason=reason,
            query_time_ms=query_time
        )
        
        # Cache the result
        self.cache[cache_key] = result
        self._save_cache()
        
        return result
    
    def batch_verify(self, claims: List[str]) -> List[WikidataResult]:
        """Verify multiple claims."""
        return [self.verify(claim) for claim in claims]


# =============================================================================
# HYBRID VERIFIER: SCP + Wikidata + LLM
# =============================================================================

class HybridVerifier:
    """
    Best of all worlds:
    1. Local KB (SCP) - fastest, ~10ms
    2. Wikidata - 100M+ facts, ~100-500ms
    3. LLM Judge - fallback, ~200ms
    
    This gives you production-quality verification TODAY.
    """
    
    def __init__(
        self,
        local_facts: List[Tuple[str, str, str]] = None,
        llm_fn=None,
        use_wikidata: bool = True
    ):
        # Local KB
        try:
            from verified_memory import create_verifier
            self.local = create_verifier(initial_facts=local_facts or [])
            self.has_local = True
        except ImportError:
            self.has_local = False
        
        # Wikidata
        self.wikidata = WikidataVerifier() if use_wikidata else None
        
        # LLM fallback
        self.llm_fn = llm_fn
    
    def verify(self, claim: str) -> Dict:
        """
        Verify using cascading strategy:
        1. Try local KB (fastest)
        2. Try Wikidata (most comprehensive)
        3. Fall back to LLM (last resort)
        """
        results = {
            "claim": claim,
            "status": "unverifiable",
            "confidence": 0.0,
            "source": None,
            "reason": "",
            "details": {}
        }
        
        # Step 1: Local KB
        if self.has_local:
            local_result = self.local.verify(claim, use_llm_fallback=False)
            if local_result.overall_status.value in ["verified", "refuted"]:
                return {
                    "claim": claim,
                    "status": local_result.overall_status.value,
                    "confidence": local_result.confidence,
                    "source": "local_kb",
                    "reason": local_result.summary,
                    "details": {"audit_trail": local_result.audit_trail}
                }
        
        # Step 2: Wikidata
        if self.wikidata:
            wiki_result = self.wikidata.verify(claim)
            if wiki_result.status != VerificationStatus.UNVERIFIABLE:
                return {
                    "claim": claim,
                    "status": wiki_result.status.value,
                    "confidence": wiki_result.confidence,
                    "source": "wikidata",
                    "reason": wiki_result.reason,
                    "details": {
                        "wikidata_facts": wiki_result.wikidata_facts,
                        "query_time_ms": wiki_result.query_time_ms
                    }
                }
        
        # Step 3: LLM fallback
        if self.llm_fn:
            try:
                from hallucination_strategies import LLMJudgeStrategy
                judge = LLMJudgeStrategy(self.llm_fn)
                llm_result = judge.check(claim)
                return {
                    "claim": claim,
                    "status": llm_result.verdict.value.lower(),
                    "confidence": llm_result.confidence,
                    "source": "llm_judge",
                    "reason": llm_result.reasoning,
                    "details": {}
                }
            except Exception as e:
                results["details"]["llm_error"] = str(e)
        
        return results


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the Wikidata verifier."""
    print("=" * 70)
    print("WIKIDATA HALLUCINATION DETECTION DEMO")
    print("=" * 70)
    print("\nThis uses Wikidata (100M+ facts) as ground truth.")
    print("No training needed - instant verification!\n")
    
    verifier = WikidataVerifier()
    
    test_claims = [
        "Alexander Graham Bell invented the telephone",  # Should VERIFY
        "Thomas Edison invented the telephone",          # Should REFUTE
        "Albert Einstein discovered the theory of relativity",  # Should VERIFY
        "Isaac Newton discovered the theory of relativity",     # Should REFUTE
        "Marie Curie discovered radium",                 # Should VERIFY
    ]
    
    print("Verifying claims against Wikidata:\n")
    
    for claim in test_claims:
        print(f"Claim: {claim}")
        result = verifier.verify(claim)
        print(f"  Status: {result.status.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Reason: {result.reason}")
        print(f"  Query time: {result.query_time_ms:.0f}ms")
        print()
    
    print("=" * 70)
    print("HYBRID VERIFIER DEMO")
    print("=" * 70)
    print("\nCombines: Local KB → Wikidata → LLM fallback\n")
    
    hybrid = HybridVerifier(
        local_facts=[
            ("Python", "created_by", "Guido van Rossum"),
        ],
        use_wikidata=True
    )
    
    test_claims_hybrid = [
        "Python was created by Guido van Rossum",  # Local KB hit
        "Bell invented the telephone",              # Wikidata hit
    ]
    
    for claim in test_claims_hybrid:
        print(f"Claim: {claim}")
        result = hybrid.verify(claim)
        print(f"  Status: {result['status']}")
        print(f"  Source: {result['source']}")
        print(f"  Reason: {result['reason']}")
        print()


# =============================================================================
# KB POPULATION STRATEGIES
# =============================================================================

KB_POPULATION_STRATEGIES = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    KB POPULATION STRATEGIES                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  STRATEGY 1: Use Wikidata Directly (RECOMMENDED FOR NOW)                      ║
║  ═══════════════════════════════════════════════════════                      ║
║  Work: ~2-3 days                                                              ║
║  Result: 100M+ facts instantly                                                ║
║                                                                               ║
║  No training needed - just query the API!                                     ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Claim → SPARQL Query → Wikidata → Structured Result                   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  STRATEGY 2: LLM Extraction + Multi-LLM Consensus                             ║
║  ════════════════════════════════════════════════                             ║
║  Work: ~1 week                                                                ║
║  Result: Custom KB from your documents                                        ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Documents → LLM1 Extract → LLM2 Verify → LLM3 Verify → Consensus → KB │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  Only facts agreed by 2+ LLMs get added to KB.                                ║
║  Start with low confidence (0.5), boost when confirmed.                       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  STRATEGY 3: Verification-Driven Population                                   ║
║  ══════════════════════════════════════════                                   ║
║  Work: ~3-5 days                                                              ║
║  Result: KB grows from usage                                                  ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Claim → Verify (Wikidata/LLM) → If verified → Add to Local KB         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  KB automatically grows from verified claims.                                 ║
║  Local cache for speed, Wikidata for ground truth.                            ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  STRATEGY 4: Import Existing KGs                                              ║
║  ═══════════════════════════════                                              ║
║  Work: ~1-2 weeks                                                             ║
║  Result: Millions of facts                                                    ║
║                                                                               ║
║  Sources:                                                                     ║
║  • Wikidata dump (~100GB, 100M+ items)                                        ║
║  • DBpedia (~5GB, 6M+ items)                                                  ║
║  • ConceptNet (~2GB, 21M+ edges)                                              ║
║  • YAGO (~10GB, 50M+ facts)                                                   ║
║                                                                               ║
║  Import into KnowShowGo format:                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  KG Dump → Transform to Propositions → Import to KnowShowGo            │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def print_strategies():
    print(KB_POPULATION_STRATEGIES)


if __name__ == "__main__":
    demo()
    print("\n")
    print_strategies()
