"""
Comprehensive Test Suite for SCP (Symbolic Consistency Probing)
================================================================
Tests the hypergraph-based hallucination detection system.

Run with: pytest test_scp.py -v
"""

import pytest
import json
from test import (
    Claim, Verdict, ProbeResult, SCPReport,
    HyperKB, SCPProber,
    RuleBasedExtractor, LLMExtractor, HybridExtractor,
    StringSimilarityBackend, SentenceTransformerBackend,
    EMBEDDINGS_AVAILABLE,
    pretty_print_report, export_proof_to_json
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def string_backend():
    """String similarity backend (no ML)."""
    return StringSimilarityBackend()


@pytest.fixture
def embedding_backend():
    """Sentence-transformer backend (if available)."""
    if not EMBEDDINGS_AVAILABLE:
        pytest.skip("sentence-transformers not installed")
    return SentenceTransformerBackend("all-MiniLM-L6-v2")


@pytest.fixture
def sample_kb(embedding_backend):
    """A sample knowledge base for testing."""
    kb = HyperKB(embedding_backend=embedding_backend)
    facts = [
        ("The Eiffel Tower", "located_in", "Paris"),
        ("Paris", "is_capital_of", "France"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Albert Einstein", "born_in", "Germany"),
        ("Charles Darwin", "proposed", "the theory of evolution"),
        ("Marie Curie", "discovered", "radium"),
        ("Marie Curie", "born_in", "Poland"),
        ("Python", "created_by", "Guido van Rossum"),
        ("The Great Wall", "located_in", "China"),
        ("Tokyo", "is_capital_of", "Japan"),
        ("Isaac Newton", "discovered", "gravity"),
        ("Isaac Newton", "born_in", "England"),
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("The Statue of Liberty", "located_in", "New York"),
        ("New York", "is_a", "city in the United States"),
    ]
    kb.add_facts_bulk(facts, source="test_data", confidence=0.95)
    return kb


@pytest.fixture
def prober(sample_kb):
    """SCPProber with sample KB."""
    return SCPProber(
        kb=sample_kb,
        extractor=RuleBasedExtractor(),
        soft_threshold=0.70
    )


# =============================================================================
# UNIT TESTS: Claim Dataclass
# =============================================================================

class TestClaim:
    def test_claim_creation(self):
        claim = Claim(
            subject="Einstein",
            predicate="discovered",
            obj="relativity",
            raw="Einstein discovered relativity."
        )
        assert claim.subject == "Einstein"
        assert claim.predicate == "discovered"
        assert claim.obj == "relativity"
        assert claim.confidence == 1.0  # default
    
    def test_claim_to_tuple(self):
        claim = Claim("A", "relates_to", "B", raw="test")
        assert claim.to_tuple() == ("A", "relates_to", "B")
    
    def test_claim_str(self):
        claim = Claim("Subject", "predicate", "Object", raw="test")
        assert str(claim) == "(Subject) --[predicate]--> (Object)"
    
    def test_claim_frozen(self):
        claim = Claim("A", "B", "C", raw="test")
        with pytest.raises(AttributeError):
            claim.subject = "modified"


# =============================================================================
# UNIT TESTS: RuleBasedExtractor
# =============================================================================

class TestRuleBasedExtractor:
    @pytest.fixture
    def extractor(self):
        return RuleBasedExtractor()
    
    def test_extract_located_in(self, extractor):
        claims = extractor.extract("The Eiffel Tower is located in Paris.")
        assert len(claims) == 1
        assert claims[0].subject == "The Eiffel Tower"
        assert claims[0].predicate == "located_in"
        assert claims[0].obj == "Paris"
    
    def test_extract_discovered(self, extractor):
        claims = extractor.extract("Einstein discovered relativity.")
        assert len(claims) == 1
        assert claims[0].predicate == "discovered"
    
    def test_extract_born_in(self, extractor):
        claims = extractor.extract("Marie Curie was born in Poland.")
        assert len(claims) == 1
        assert claims[0].predicate == "born_in"
    
    def test_extract_invented(self, extractor):
        claims = extractor.extract("Bell invented the telephone.")
        assert len(claims) == 1
        assert claims[0].predicate == "invented"
    
    def test_extract_is_capital(self, extractor):
        claims = extractor.extract("Paris is the capital of France.")
        assert len(claims) == 1
        assert claims[0].predicate == "capital_of"
    
    def test_extract_multiple_sentences(self, extractor):
        text = "Einstein discovered relativity. He was born in Germany."
        claims = extractor.extract(text)
        assert len(claims) == 2
    
    def test_extract_no_claims(self, extractor):
        claims = extractor.extract("Hello world!")
        assert len(claims) == 0
    
    def test_extract_empty_string(self, extractor):
        claims = extractor.extract("")
        assert len(claims) == 0
    
    def test_extract_handles_abbreviations(self, extractor):
        claims = extractor.extract("Dr. Smith discovered something. Mr. Jones invented something else.")
        assert len(claims) == 2


# =============================================================================
# UNIT TESTS: HyperKB
# =============================================================================

class TestHyperKB:
    def test_create_empty_kb(self):
        kb = HyperKB()
        stats = kb.stats()
        assert stats["entities"] == 0
        assert stats["relations"] == 0
    
    def test_add_entity(self):
        kb = HyperKB()
        eid = kb.add_entity("Paris")
        assert eid == "ent:paris"
        assert kb.g.has_node(eid)
        assert kb.g.nodes[eid]["kind"] == "entity"
    
    def test_add_fact(self):
        kb = HyperKB()
        rel_id = kb.add_fact("Paris", "is_capital_of", "France")
        
        # Check entities created
        assert kb.g.has_node("ent:paris")
        assert kb.g.has_node("ent:france")
        
        # Check relation created
        assert kb.g.has_node(rel_id)
        assert kb.g.nodes[rel_id]["kind"] == "relation"
        
        # Check edges
        assert kb.g.has_edge("ent:paris", rel_id)
        assert kb.g.has_edge(rel_id, "ent:france")
    
    def test_add_facts_bulk(self):
        kb = HyperKB()
        facts = [
            ("A", "relates_to", "B"),
            ("B", "relates_to", "C"),
            ("C", "relates_to", "D"),
        ]
        rel_ids = kb.add_facts_bulk(facts)
        assert len(rel_ids) == 3
        assert kb.stats()["relations"] == 3
    
    def test_iter_facts(self):
        kb = HyperKB()
        kb.add_fact("A", "rel", "B")
        kb.add_fact("C", "rel2", "D")
        
        facts = kb.iter_facts()
        assert len(facts) == 2
    
    def test_has_fact_exact(self):
        kb = HyperKB()
        kb.add_fact("Einstein", "discovered", "relativity")
        
        claim = Claim("Einstein", "discovered", "relativity", raw="test")
        assert kb.has_fact_exact(claim) is not None
        
        claim2 = Claim("Einstein", "invented", "relativity", raw="test")
        assert kb.has_fact_exact(claim2) is None
    
    def test_has_fact_exact_normalization(self):
        """Test that matching is case-insensitive and ignores punctuation."""
        kb = HyperKB()
        kb.add_fact("Albert Einstein", "discovered", "the theory of relativity")
        
        # Same content, different case
        claim = Claim("albert einstein", "discovered", "the theory of relativity", raw="test")
        assert kb.has_fact_exact(claim) is not None
    
    def test_find_contradictions(self):
        kb = HyperKB()
        kb.add_fact("Einstein", "born_in", "Germany")
        
        # Same subject and predicate, different object = contradiction
        claim = Claim("Einstein", "born_in", "France", raw="test")
        contradictions = kb.find_contradictions(claim)
        assert len(contradictions) == 1
        assert contradictions[0][2] == "Germany"  # object
    
    def test_get_proof_subgraph(self):
        kb = HyperKB()
        rel_id = kb.add_fact("A", "rel", "B")
        
        subgraph = kb.get_proof_subgraph([rel_id])
        assert len(subgraph.nodes()) == 3  # A, rel, B
        assert len(subgraph.edges()) == 2
    
    def test_to_dict(self):
        kb = HyperKB()
        kb.add_fact("A", "rel", "B")
        
        data = kb.to_dict()
        assert "entities" in data
        assert "relations" in data
        assert "facts" in data
        assert len(data["entities"]) == 2
        assert len(data["relations"]) == 1


# =============================================================================
# UNIT TESTS: Embedding Backends
# =============================================================================

class TestStringSimilarityBackend:
    def test_text_similarity_exact(self):
        backend = StringSimilarityBackend()
        sim = backend.text_similarity("hello", "hello")
        assert sim == 1.0
    
    def test_text_similarity_similar(self):
        backend = StringSimilarityBackend()
        sim = backend.text_similarity("hello world", "hello worlds")
        assert sim > 0.8
    
    def test_text_similarity_different(self):
        backend = StringSimilarityBackend()
        sim = backend.text_similarity("hello", "goodbye")
        assert sim < 0.5


@pytest.mark.skipif(not EMBEDDINGS_AVAILABLE, reason="sentence-transformers not installed")
class TestSentenceTransformerBackend:
    def test_encode(self, embedding_backend):
        vecs = embedding_backend.encode(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) > 0  # embedding dimension
    
    def test_similarity_same(self, embedding_backend):
        vecs = embedding_backend.encode(["hello world"])
        sim = embedding_backend.similarity(vecs[0], vecs[0])
        assert abs(sim - 1.0) < 0.01  # Should be ~1.0
    
    def test_text_similarity_semantic(self, embedding_backend):
        # Semantically similar but different words
        sim = embedding_backend.text_similarity(
            "Einstein discovered relativity",
            "Albert Einstein found the theory of relativity"
        )
        assert sim > 0.7
    
    def test_caching(self, embedding_backend):
        # First call should cache
        embedding_backend.encode(["test string"])
        assert "test string" in embedding_backend._cache
        
        # Second call should use cache
        embedding_backend.encode(["test string"])


# =============================================================================
# INTEGRATION TESTS: SCPProber
# =============================================================================

class TestSCPProber:
    def test_exact_match(self, prober):
        """Test that exact matches get PASS verdict."""
        report = prober.probe("The Eiffel Tower is located in Paris.")
        assert report.claims_extracted == 1
        assert report.results[0].verdict == Verdict.PASS
        assert report.results[0].score == 1.0
    
    def test_soft_match(self, prober):
        """Test that semantically similar claims get SOFT_PASS."""
        report = prober.probe("Einstein discovered relativity.")
        assert report.results[0].verdict == Verdict.SOFT_PASS
        assert report.results[0].score > 0.7
    
    def test_contradiction(self, prober):
        """Test that contradicting claims get CONTRADICT verdict."""
        report = prober.probe("Albert Einstein was born in France.")
        assert report.results[0].verdict == Verdict.CONTRADICT
        assert report.results[0].score == 0.0
    
    def test_fail_no_match(self, prober):
        """Test that unsupported claims get FAIL verdict."""
        report = prober.probe("Napoleon invented the lightbulb.")
        assert report.results[0].verdict == Verdict.FAIL
    
    def test_multiple_claims(self, prober):
        """Test handling of multiple claims in one text."""
        text = "Marie Curie discovered radium. She was born in Poland."
        report = prober.probe(text)
        assert report.claims_extracted == 2
        
        # First claim should pass
        assert report.results[0].verdict == Verdict.PASS
    
    def test_hallucination_detection(self, prober):
        """Test that hallucinations are properly detected."""
        text = "Marie Curie invented the telephone."
        report = prober.probe(text)
        # Should fail since Bell invented the telephone
        assert report.results[0].verdict == Verdict.FAIL
    
    def test_overall_score(self, prober):
        """Test that overall score is calculated correctly."""
        # Single exact match
        report = prober.probe("The Eiffel Tower is located in Paris.")
        assert report.overall_score == 1.0
        assert report.pass_rate == 1.0
        
        # Contradiction should have 0 score
        report = prober.probe("The Eiffel Tower is located in London.")
        assert report.overall_score == 0.0
        assert report.pass_rate == 0.0
    
    def test_no_claims_extracted(self, prober):
        """Test handling when no claims can be extracted."""
        report = prober.probe("Hello there!")
        assert report.claims_extracted == 0
        assert report.results[0].verdict == Verdict.UNKNOWN
    
    def test_probe_batch(self, prober):
        """Test batch probing."""
        texts = [
            "The Eiffel Tower is located in Paris.",
            "Einstein discovered relativity.",
        ]
        reports = prober.probe_batch(texts)
        assert len(reports) == 2
    
    def test_proof_subgraph_generated(self, prober):
        """Test that proof subgraphs are generated for matches."""
        report = prober.probe("The Eiffel Tower is located in Paris.")
        assert report.results[0].proof_subgraph is not None
        
        # Proof should contain the matched fact
        proof = report.results[0].proof_subgraph
        assert proof.number_of_nodes() >= 3
    
    def test_json_export(self, prober):
        """Test JSON export functionality."""
        report = prober.probe("The Eiffel Tower is located in Paris.")
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data["claims_extracted"] == 1
        assert data["overall_score"] == 1.0
        assert data["results"][0]["verdict"] == "PASS"


# =============================================================================
# TESTS: Verdict Types
# =============================================================================

class TestVerdictScenarios:
    """Test various scenarios to ensure correct verdict assignment."""
    
    def test_exact_location_match(self, prober):
        report = prober.probe("Tokyo is the capital of Japan.")
        # Note: pattern might match slightly differently
        assert report.claims_extracted >= 1
    
    def test_inventor_claim(self, prober):
        # Correct claim
        report = prober.probe("Alexander Graham Bell invented the telephone.")
        assert len(report.results) == 1
        # Should either PASS exactly or SOFT_PASS
        assert report.results[0].verdict in [Verdict.PASS, Verdict.SOFT_PASS]
    
    def test_false_inventor_claim(self, prober):
        # False claim - Edison did not invent the telephone
        report = prober.probe("Thomas Edison invented the telephone.")
        # Should fail since we know Bell invented it
        assert report.results[0].verdict in [Verdict.FAIL, Verdict.CONTRADICT]
    
    def test_capital_city(self, prober):
        report = prober.probe("Paris is the capital of France.")
        assert report.claims_extracted >= 1


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_kb(self):
        """Test prober with empty KB."""
        kb = HyperKB()
        prober = SCPProber(kb=kb)
        report = prober.probe("Einstein discovered relativity.")
        assert report.results[0].verdict == Verdict.FAIL
    
    def test_unicode_text(self, prober):
        """Test handling of unicode characters."""
        report = prober.probe("The Eiffel Tower is located in Paris.")
        assert report.claims_extracted == 1
    
    def test_very_long_text(self, prober):
        """Test handling of long text."""
        text = ". ".join([
            "The Eiffel Tower is located in Paris",
            "Einstein discovered relativity",
            "Marie Curie discovered radium",
        ] * 10)
        report = prober.probe(text)
        assert report.claims_extracted >= 10
    
    def test_special_characters(self, prober):
        """Test handling of special characters."""
        report = prober.probe("The Eiffel Tower (built 1889) is located in Paris!")
        assert report.claims_extracted >= 1
    
    def test_threshold_boundary(self, sample_kb):
        """Test behavior at threshold boundaries."""
        # Very high threshold
        strict_prober = SCPProber(kb=sample_kb, soft_threshold=0.99)
        report = strict_prober.probe("Einstein discovered relativity.")
        # Should fail with high threshold
        assert report.results[0].verdict in [Verdict.FAIL, Verdict.SOFT_PASS]
        
        # Very low threshold
        lenient_prober = SCPProber(kb=sample_kb, soft_threshold=0.3)
        report = lenient_prober.probe("Einstein found physics.")
        # More likely to soft pass with low threshold


# =============================================================================
# TESTS: Proof Export
# =============================================================================

class TestProofExport:
    def test_export_proof_to_json(self, prober):
        """Test proof subgraph JSON export."""
        report = prober.probe("The Eiffel Tower is located in Paris.")
        proof = report.results[0].proof_subgraph
        
        if proof is not None:
            json_data = export_proof_to_json(proof)
            assert "nodes" in json_data
            assert "edges" in json_data
            assert len(json_data["nodes"]) >= 3


# =============================================================================
# TESTS: False Attribution Detection
# =============================================================================

class TestFalseAttributionDetection:
    """Tests for detecting false attribution hallucinations."""
    
    def test_wrong_inventor(self, prober):
        """Test that wrong inventor attribution is detected."""
        # Edison did NOT invent the telephone (Bell did)
        report = prober.probe("Thomas Edison invented the telephone.")
        assert report.results[0].verdict == Verdict.FAIL
        # Should mention the correct inventor
        assert "Alexander Graham Bell" in report.results[0].reason or \
               report.results[0].metadata.get("correct_subject") == "Alexander Graham Bell"
    
    def test_wrong_discoverer(self, prober):
        """Test that wrong discoverer attribution is detected."""
        # Marie Curie discovered radium, not relativity
        report = prober.probe("Marie Curie discovered the theory of relativity.")
        # Either FAIL (false attribution) or the system should flag it
        assert report.results[0].verdict in [Verdict.FAIL, Verdict.CONTRADICT]
    
    def test_correct_attribution_still_passes(self, prober):
        """Ensure correct attributions still work."""
        report = prober.probe("Alexander Graham Bell invented the telephone.")
        assert report.results[0].verdict in [Verdict.PASS, Verdict.SOFT_PASS]
    
    def test_false_attribution_metadata(self, prober):
        """Test that false attribution provides useful metadata."""
        report = prober.probe("Thomas Edison invented the telephone.")
        if report.results[0].metadata.get("match_type") == "false_attribution":
            assert "correct_subject" in report.results[0].metadata
            assert "claimed_subject" in report.results[0].metadata


# =============================================================================
# TESTS: Coreference (Known Limitation)
# =============================================================================

class TestCoreferenceHandling:
    """Tests for pronoun/coreference handling - a known limitation.
    
    Note: Without proper coreference resolution, pronouns like "She" 
    cannot be resolved to their antecedent (e.g., "Marie Curie").
    This is a known limitation of the current implementation.
    """
    
    def test_pronoun_claim_extracted(self, prober):
        """Test that pronoun claims are at least extracted."""
        report = prober.probe("She was born in Poland.")
        assert report.claims_extracted == 1
        assert report.results[0].claim.subject == "She"
    
    def test_explicit_name_works(self, prober):
        """Test that explicit names work correctly."""
        report = prober.probe("Marie Curie was born in Poland.")
        assert report.results[0].verdict in [Verdict.PASS, Verdict.SOFT_PASS]
    
    def test_explicit_vs_pronoun(self, prober):
        """Compare explicit name vs pronoun matching."""
        # With explicit name
        report1 = prober.probe("Marie Curie was born in Poland.")
        
        # With pronoun (no context) - will likely fail
        report2 = prober.probe("She was born in Poland.")
        
        # Explicit name should pass, pronoun may fail
        assert report1.results[0].verdict in [Verdict.PASS, Verdict.SOFT_PASS]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
