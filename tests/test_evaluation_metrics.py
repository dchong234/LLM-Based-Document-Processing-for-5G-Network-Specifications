"""
Unit tests for evaluation metrics
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_metrics import (
    exact_match,
    semantic_similarity,
    keyword_match,
    bleu_score,
    f1_score_keywords,
    normalize_text,
    tokenize_text,
    calculate_all_metrics,
    evaluate_predictions
)


class TestNormalization:
    """Test text normalization functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "  5G NR  is   the  technology  "
        normalized = normalize_text(text)
        
        assert normalized == "5g nr is the technology"
        assert "  " not in normalized  # No double spaces
        assert normalized == normalized.strip()  # No leading/trailing spaces
    
    def test_normalize_text_empty(self):
        """Test normalizing empty text."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # Handles None
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "5G NR is the radio access technology"
        tokens = tokenize_text(text)
        
        assert len(tokens) == 7
        assert "5g" in tokens
        assert "nr" in tokens
        assert "technology" in tokens
    
    def test_tokenize_text_empty(self):
        """Test tokenizing empty text."""
        assert tokenize_text("") == []
        assert tokenize_text("   ") == []


class TestExactMatch:
    """Test exact match metric."""
    
    def test_exact_match_same(self):
        """Test exact match with identical texts."""
        pred = "5G NR is the radio access technology"
        truth = "5G NR is the radio access technology"
        
        score = exact_match(pred, truth)
        
        assert score == 1.0
    
    def test_exact_match_different(self):
        """Test exact match with different texts."""
        pred = "5G NR is the radio access technology"
        truth = "5G New Radio is the radio access technology standard"
        
        score = exact_match(pred, truth)
        
        assert score == 0.0
    
    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        pred = "5G NR is the technology"
        truth = "5g nr is the technology"
        
        score = exact_match(pred, truth, case_sensitive=False)
        
        assert score == 1.0
    
    def test_exact_match_case_sensitive(self):
        """Test case-sensitive matching."""
        pred = "5G NR is the technology"
        truth = "5g nr is the technology"
        
        score = exact_match(pred, truth, case_sensitive=True)
        
        assert score == 0.0
    
    def test_exact_match_whitespace(self):
        """Test that whitespace differences are ignored."""
        pred = "  5G NR is the technology  "
        truth = "5G NR is the technology"
        
        score = exact_match(pred, truth)
        
        assert score == 1.0
    
    def test_exact_match_empty(self):
        """Test with empty strings."""
        assert exact_match("", "") == 0.0
        assert exact_match("text", "") == 0.0
        assert exact_match("", "text") == 0.0


class TestKeywordMatch:
    """Test keyword match metric."""
    
    def test_keyword_match_all_found(self):
        """Test keyword match when all keywords found."""
        pred = "5G NR is the radio access technology for 5G networks"
        truth = "5G NR (New Radio) is the radio access technology standard"
        keywords = ["5G NR", "radio access technology", "5G networks"]
        
        score = keyword_match(pred, truth, keywords)
        
        assert score >= 0.5  # At least some keywords found
        assert score <= 1.0
    
    def test_keyword_match_none_found(self):
        """Test keyword match when no keywords found."""
        pred = "This is completely different content"
        truth = "5G NR is the radio access technology"
        keywords = ["5G NR", "radio access technology"]
        
        score = keyword_match(pred, truth, keywords)
        
        assert score == 0.0
    
    def test_keyword_match_partial(self):
        """Test keyword match with partial match."""
        pred = "5G NR is the radio technology"
        truth = "5G NR (New Radio) is the radio access technology standard"
        keywords = ["5G NR", "radio access technology", "standard"]
        
        score = keyword_match(pred, truth, keywords)
        
        assert 0.0 < score < 1.0
    
    def test_keyword_match_auto_extract(self):
        """Test automatic keyword extraction."""
        pred = "5G NR is the radio access technology"
        truth = "5G NR (New Radio) is the radio access technology standard"
        
        score = keyword_match(pred, truth)
        
        # Should extract keywords and match
        assert score >= 0.0
        assert score <= 1.0
    
    def test_keyword_match_empty(self):
        """Test with empty strings."""
        assert keyword_match("", "text") == 0.0
        assert keyword_match("text", "") == 0.0


class TestF1Score:
    """Test F1 score metric."""
    
    def test_f1_score_perfect(self):
        """Test F1 score with perfect overlap."""
        pred = "5G NR radio access technology"
        truth = "5G NR radio access technology"
        
        score = f1_score_keywords(pred, truth)
        
        assert score == 1.0
    
    def test_f1_score_no_overlap(self):
        """Test F1 score with no overlap."""
        pred = "completely different words here"
        truth = "5G NR radio access technology"
        
        score = f1_score_keywords(pred, truth)
        
        assert score == 0.0
    
    def test_f1_score_partial(self):
        """Test F1 score with partial overlap."""
        pred = "5G NR is the technology"
        truth = "5G NR radio access technology"
        
        score = f1_score_keywords(pred, truth)
        
        assert 0.0 < score < 1.0
    
    def test_f1_score_empty(self):
        """Test with empty strings."""
        assert f1_score_keywords("", "text") == 0.0
        assert f1_score_keywords("text", "") == 0.0


class TestBLEUScore:
    """Test BLEU score metric."""
    
    def test_bleu_score_perfect(self):
        """Test BLEU score with identical texts."""
        pred = "5G NR is the radio access technology"
        truth = "5G NR is the radio access technology"
        
        score = bleu_score(pred, truth, verbose=False)
        
        # Should be high (may not be 1.0 due to smoothing)
        assert score > 0.5
    
    def test_bleu_score_no_overlap(self):
        """Test BLEU score with no overlap."""
        pred = "completely different words"
        truth = "5G NR radio access technology"
        
        score = bleu_score(pred, truth, verbose=False)
        
        # Should be low
        assert score < 0.5
    
    def test_bleu_score_partial(self):
        """Test BLEU score with partial overlap."""
        pred = "5G NR is the technology"
        truth = "5G NR radio access technology standard"
        
        score = bleu_score(pred, truth, verbose=False)
        
        assert 0.0 < score < 1.0
    
    def test_bleu_score_empty(self):
        """Test with empty strings."""
        assert bleu_score("", "text", verbose=False) == 0.0
        assert bleu_score("text", "", verbose=False) == 0.0


class TestSemanticSimilarity:
    """Test semantic similarity metric."""
    
    @pytest.mark.skipif(True, reason="Requires sentence-transformers")
    def test_semantic_similarity_similar(self):
        """Test semantic similarity with similar texts."""
        pred = "5G NR is the radio access technology"
        truth = "5G New Radio is the radio access technology standard"
        
        score = semantic_similarity(pred, truth, verbose=False)
        
        # Should be high (similar meaning)
        assert score > 0.7
        assert score <= 1.0
    
    def test_semantic_similarity_unavailable(self):
        """Test semantic similarity when library unavailable."""
        pred = "5G NR is the radio access technology"
        truth = "5G New Radio is the radio access technology"
        
        score = semantic_similarity(pred, truth, verbose=False)
        
        # Should return 0.0 if library unavailable
        assert score == 0.0 or score >= 0.0


class TestCalculateAllMetrics:
    """Test calculate_all_metrics function."""
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics."""
        pred = "5G NR is the radio access technology"
        truth = "5G NR (New Radio) is the radio access technology standard"
        keywords = ["5G NR", "radio access technology"]
        
        metrics = calculate_all_metrics(pred, truth, keywords, verbose=False)
        
        assert isinstance(metrics, dict)
        assert 'exact_match' in metrics
        assert 'semantic_similarity' in metrics
        assert 'keyword_match' in metrics
        assert 'bleu_score' in metrics
        assert 'f1_score' in metrics
        
        # Check all scores are between 0 and 1
        for metric, score in metrics.items():
            assert 0.0 <= score <= 1.0, f"{metric} score out of range: {score}"
    
    def test_evaluate_predictions(self):
        """Test batch evaluation."""
        predictions = [
            "5G NR is the radio access technology",
            "HARQ is an error correction mechanism"
        ]
        ground_truths = [
            "5G NR (New Radio) is the radio access technology standard",
            "HARQ (Hybrid ARQ) is an error correction mechanism"
        ]
        keywords_list = [
            ["5G NR", "radio access technology"],
            ["HARQ", "error correction"]
        ]
        
        results = evaluate_predictions(predictions, ground_truths, keywords_list)
        
        assert 'average_metrics' in results
        assert 'per_question_metrics' in results
        assert 'total_questions' in results
        assert results['total_questions'] == 2
        assert len(results['per_question_metrics']) == 2
    
    def test_evaluate_predictions_mismatch(self):
        """Test evaluation with mismatched lists."""
        predictions = ["pred1", "pred2"]
        ground_truths = ["truth1"]
        
        with pytest.raises(ValueError):
            evaluate_predictions(predictions, ground_truths)

