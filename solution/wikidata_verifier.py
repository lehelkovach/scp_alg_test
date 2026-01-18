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
import re
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
        "invented": {"property": "P61", "direction": "object_to_subject"},
        "discovered": {"property": "P61", "direction": "object_to_subject"},
        "created": {"property": "P170", "direction": "object_to_subject"},
        "created by": {"property": "P170", "direction": "subject_to_object"},
        "founded": {"property": "P112", "direction": "object_to_subject"},
        "founded by": {"property": "P112", "direction": "subject_to_object"},
        "born in": {"property": "P19", "direction": "subject_to_object"},
        "born": {"property": "P569", "direction": "subject_to_object"},
        "died in": {"property": "P20", "direction": "subject_to_object"},
        "died": {"property": "P570", "direction": "subject_to_object"},
        "located in": {"property": "P131", "direction": "subject_to_object"},
        "capital of": {"property": "P36", "direction": "object_to_subject"},
        "wrote": {"property": "P50", "direction": "object_to_subject"},
        "written by": {"property": "P50", "direction": "subject_to_object"},
        "directed": {"property": "P57", "direction": "object_to_subject"},
        "directed by": {"property": "P57", "direction": "subject_to_object"},
        "ceo of": {"property": "P169", "direction": "object_to_subject"},
        "spouse of": {"property": "P26", "direction": "subject_to_object"},
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
    
    def _normalize_predicate(self, predicate: str) -> str:
        pred = predicate.strip().lower().replace("_", " ")
        pred = re.sub(r"\s+", " ", pred)
        pred = re.sub(r"^(is|was|were)\s+", "", pred)
        pred = re.sub(r"^the\s+", "", pred)
        return pred.strip()

    def _normalize_label(self, label: str) -> str:
        text = label.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'[\"\'`,.;:!?()\[\]{}]', "", text)
        return text.strip()

    def _labels_match(self, a: str, b: str) -> bool:
        a_norm = self._normalize_label(a)
        b_norm = self._normalize_label(b)
        return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm

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

    def _search_entity_with_variants(self, name: str) -> Optional[str]:
        variants = [name, f"the {name}", name.replace("the ", "")]
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned:
                continue
            entity_id = self._search_entity(cleaned)
            if entity_id:
                return entity_id
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
    
    def _verify_invention(
        self,
        subject: str,
        obj: str,
        predicate_label: str,
    ) -> Tuple[VerificationStatus, float, List[Dict], str]:
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
                    f"Wikidata confirms: {inventor} {predicate_label} {obj}"
                )
        
        # No match - it's a refutation
        actual_inventor = results[0].get("inventorLabel", {}).get("value", "Unknown")
        return (
            VerificationStatus.REFUTED,
            0.90,
            results,
            f"Wikidata shows {actual_inventor} {predicate_label} {obj}, not {subject}"
        )

    def _verify_property(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> Tuple[VerificationStatus, float, List[Dict], str]:
        predicate_key = self._normalize_predicate(predicate)
        mapping = self.PREDICATE_MAP.get(predicate_key)
        if not mapping:
            return (
                VerificationStatus.UNVERIFIABLE,
                0.5,
                [],
                f"Predicate '{predicate}' not yet supported",
            )

        property_id = mapping["property"]
        direction = mapping.get("direction", "subject_to_object")

        if direction == "subject_to_object":
            subject_id = self._search_entity_with_variants(subject)
            if not subject_id:
                return (
                    VerificationStatus.UNVERIFIABLE,
                    0.5,
                    [],
                    f"Could not find '{subject}' in Wikidata",
                )

            results = self._get_entity_claims(subject_id, property_id)
            if not results:
                return (
                    VerificationStatus.UNVERIFIABLE,
                    0.5,
                    [],
                    f"No Wikidata values for '{subject}' and predicate '{predicate_key}'",
                )

            for result in results:
                value = result.get("valueLabel", {}).get("value", "")
                if value and self._labels_match(obj, value):
                    return (
                        VerificationStatus.VERIFIED,
                        0.90,
                        results,
                        f"Wikidata confirms: {subject} {predicate_key} {value}",
                    )

            actual_value = results[0].get("valueLabel", {}).get("value", "Unknown")
            return (
                VerificationStatus.REFUTED,
                0.90,
                results,
                f"Wikidata shows {subject} {predicate_key} {actual_value}, not {obj}",
            )

        if direction == "object_to_subject":
            object_id = self._search_entity_with_variants(obj)
            if not object_id:
                return (
                    VerificationStatus.UNVERIFIABLE,
                    0.5,
                    [],
                    f"Could not find '{obj}' in Wikidata",
                )

            results = self._get_entity_claims(object_id, property_id)
            if not results:
                return (
                    VerificationStatus.UNVERIFIABLE,
                    0.5,
                    [],
                    f"No Wikidata values for '{obj}' and predicate '{predicate_key}'",
                )

            for result in results:
                value = result.get("valueLabel", {}).get("value", "")
                if value and self._labels_match(subject, value):
                    return (
                        VerificationStatus.VERIFIED,
                        0.90,
                        results,
                        f"Wikidata confirms: {obj} {predicate_key} {value}",
                    )

            actual_value = results[0].get("valueLabel", {}).get("value", "Unknown")
            return (
                VerificationStatus.REFUTED,
                0.90,
                results,
                f"Wikidata shows {obj} {predicate_key} {actual_value}, not {subject}",
            )

        return (
            VerificationStatus.UNVERIFIABLE,
            0.5,
            [],
            f"Predicate direction '{direction}' not supported",
        )
    
    def _extract_claim_parts(self, claim: str) -> Optional[Tuple[str, str, str]]:
        """Extract subject, predicate, object from claim."""
        import re
        
        patterns = [
            r"^(.+?)\s+(invented|discovered|created|founded|wrote|directed)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:was|were)\s+(created by|founded by|written by|directed by)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:was|were)\s+(born in|died in)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:was|were)\s+(born|died)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:is|was|were)\s+(located in)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:is|was|were)\s+the\s+(capital of)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:is|was|were)\s+the\s+(ceo of)\s+(.+?)\.?$",
            r"^(.+?)\s+(?:is|was|were)\s+the\s+(spouse of)\s+(.+?)\.?$",
            r"^(.+?)\s+(is|was)\s+(.+?)\.?$",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, claim, re.IGNORECASE)
            if match:
                predicate = self._normalize_predicate(match.group(2))
                return match.group(1).strip(), predicate, match.group(3).strip()
        
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
        predicate = self._normalize_predicate(predicate)
        
        # Route to appropriate verification method
        if predicate in ["invented", "discovered"]:
            status, confidence, facts, reason = self._verify_invention(subject, obj, predicate)
        elif predicate in self.PREDICATE_MAP:
            status, confidence, facts, reason = self._verify_property(subject, predicate, obj)
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
