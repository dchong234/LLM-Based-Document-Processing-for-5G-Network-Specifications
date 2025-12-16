"""
Evaluation Metrics for Fine-tuned Model Testing
Implements various metrics to evaluate model predictions against ground truth.
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers or scikit-learn not installed. Install with: pip install sentence-transformers scikit-learn")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    try:
        # Try alternative BLEU implementation
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False
        print("Warning: nltk not installed. Install with: pip install nltk")

# Global sentence transformer model (lazy loading)
_sentence_model = None


def get_sentence_model():
    """
    Get or initialize the sentence transformer model.
    
    Returns:
        SentenceTransformer model
    """
    global _sentence_model
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers scikit-learn")
    
    if _sentence_model is None:
        print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded")
    
    return _sentence_model


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Removes extra whitespace, converts to lowercase, and removes punctuation.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text_preserve_punctuation(text: str) -> str:
    """
    Normalize text while preserving punctuation.
    
    Removes extra whitespace and converts to lowercase, but keeps punctuation.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text with punctuation preserved
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
    
    Returns:
        List of tokens (words)
    """
    if not text:
        return []
    
    # Normalize text
    normalized = normalize_text(text)
    
    # Split on whitespace
    tokens = normalized.split()
    
    # Remove empty tokens
    tokens = [t for t in tokens if t]
    
    return tokens


def exact_match(prediction: str, ground_truth: str, case_sensitive: bool = False) -> float:
    """
    Calculate exact match score (case-insensitive by default).
    
    Returns 1.0 if texts match exactly (after normalization), 0.0 otherwise.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        case_sensitive: Whether to perform case-sensitive comparison (default: False)
    
    Returns:
        Exact match score (1.0 for match, 0.0 for no match)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    if case_sensitive:
        pred_norm = prediction.strip()
        truth_norm = ground_truth.strip()
    else:
        pred_norm = normalize_text(prediction)
        truth_norm = normalize_text(ground_truth)
    
    return 1.0 if pred_norm == truth_norm else 0.0


def semantic_similarity(prediction: str, ground_truth: str, verbose: bool = False) -> float:
    """
    Calculate semantic similarity using sentence-transformers.
    
    Uses cosine similarity between sentence embeddings from all-MiniLM-L6-v2 model.
    Returns a score between 0.0 and 1.0.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        verbose: Whether to print warnings (default: False)
    
    Returns:
        Semantic similarity score (0.0 to 1.0)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        if verbose:
            print("Warning: sentence-transformers not available. Semantic similarity cannot be calculated.")
        return 0.0
    
    try:
        model = get_sentence_model()
        
        # Get embeddings
        embeddings = model.encode([prediction, ground_truth])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Ensure score is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    except Exception as e:
        if verbose:
            print(f"Warning: Error calculating semantic similarity: {e}")
        return 0.0


def keyword_match(prediction: str, ground_truth: str, keywords: Optional[List[str]] = None) -> float:
    """
    Calculate keyword match score.
    
    Checks if keywords from ground truth (or provided keywords) appear in the prediction.
    Returns the fraction of keywords found in the prediction.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer (used to extract keywords if keywords not provided)
        keywords: Optional list of keywords to check (default: None, extracts from ground_truth)
    
    Returns:
        Keyword match score (0.0 to 1.0)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    # Extract keywords if not provided
    if keywords is None:
        # Extract keywords from ground truth (words that are not common stop words)
        # Simple approach: use all significant words
        truth_tokens = tokenize_text(ground_truth)
        # Filter out very short words and common words
        keywords = [t for t in truth_tokens if len(t) > 3]
    
    if not keywords:
        return 0.0
    
    # Normalize keywords and prediction
    keywords_lower = [normalize_text(kw) for kw in keywords]
    pred_normalized = normalize_text(prediction)
    
    # Count how many keywords appear in prediction
    found_keywords = 0
    for keyword in keywords_lower:
        if keyword in pred_normalized:
            found_keywords += 1
    
    # Return fraction of keywords found
    return found_keywords / len(keywords) if keywords else 0.0


def bleu_score(prediction: str, ground_truth: str, smoothing: bool = True, verbose: bool = False) -> float:
    """
    Calculate BLEU score for text similarity.
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
    prediction and ground truth. Returns a score between 0.0 and 1.0.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        smoothing: Whether to use smoothing (default: True)
        verbose: Whether to print warnings (default: False)
    
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    if not NLTK_AVAILABLE:
        if verbose:
            print("Warning: nltk not available. BLEU score cannot be calculated.")
        return 0.0
    
    try:
        # Tokenize texts
        if NLTK_AVAILABLE:
            try:
                pred_tokens = word_tokenize(normalize_text_preserve_punctuation(prediction))
                truth_tokens = word_tokenize(normalize_text_preserve_punctuation(ground_truth))
            except LookupError:
                # Fallback to simple tokenization if word_tokenize not available
                pred_tokens = tokenize_text(prediction)
                truth_tokens = tokenize_text(ground_truth)
        else:
            pred_tokens = tokenize_text(prediction)
            truth_tokens = tokenize_text(ground_truth)
        
        # Prepare reference (ground truth) as list of tokens
        reference = [truth_tokens]
        
        # Calculate BLEU score
        if smoothing:
            smoothing_function = SmoothingFunction().method1
            score = sentence_bleu(reference, pred_tokens, smoothing_function=smoothing_function)
        else:
            score = sentence_bleu(reference, pred_tokens)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return float(score)
    
    except Exception as e:
        if verbose:
            print(f"Warning: Error calculating BLEU score: {e}")
        return 0.0


def f1_score_keywords(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score based on keyword overlap.
    
    Treats prediction and ground truth as sets of keywords and calculates
    F1 score based on precision and recall of keyword overlap.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    # Tokenize and create sets of tokens
    pred_tokens = set(tokenize_text(prediction))
    truth_tokens = set(tokenize_text(ground_truth))
    
    # Calculate intersection (common tokens)
    intersection = pred_tokens & truth_tokens
    
    # Calculate precision and recall
    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(pred_tokens)
    
    if len(truth_tokens) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(truth_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def calculate_all_metrics(
    prediction: str,
    ground_truth: str,
    keywords: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a prediction.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        keywords: Optional list of keywords for keyword matching
        verbose: Whether to print warnings (default: False)
    
    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        'exact_match': exact_match(prediction, ground_truth),
        'semantic_similarity': semantic_similarity(prediction, ground_truth, verbose=verbose),
        'keyword_match': keyword_match(prediction, ground_truth, keywords),
        'bleu_score': bleu_score(prediction, ground_truth, verbose=verbose),
        'f1_score': f1_score_keywords(prediction, ground_truth),
    }
    
    return metrics


def evaluate_predictions(
    predictions: List[str],
    ground_truths: List[str],
    keywords_list: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    Evaluate a list of predictions against ground truths.
    
    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        keywords_list: Optional list of keyword lists (one per question)
    
    Returns:
        Dictionary with average scores and per-question scores
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})")
    
    all_metrics = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        keywords = keywords_list[i] if keywords_list else None
        metrics = calculate_all_metrics(pred, truth, keywords)
        all_metrics.append(metrics)
    
    # Calculate averages
    avg_metrics = {
        'exact_match': sum(m['exact_match'] for m in all_metrics) / len(all_metrics),
        'semantic_similarity': sum(m['semantic_similarity'] for m in all_metrics) / len(all_metrics),
        'keyword_match': sum(m['keyword_match'] for m in all_metrics) / len(all_metrics),
        'bleu_score': sum(m['bleu_score'] for m in all_metrics) / len(all_metrics),
        'f1_score': sum(m['f1_score'] for m in all_metrics) / len(all_metrics),
    }
    
    return {
        'average_metrics': avg_metrics,
        'per_question_metrics': all_metrics,
        'total_questions': len(predictions),
    }


def print_evaluation_results(results: Dict[str, Any]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Results dictionary from evaluate_predictions
    """
    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total questions: {results['total_questions']}")
    print(f"\nAverage Metrics:")
    for metric, score in results['average_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {score:.4f}")
    print(f"{'=' * 60}\n")


def main():
    """Main function to test evaluation metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test evaluation metrics for model predictions"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with sample predictions'
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing evaluation metrics...")
        print("=" * 60)
        
        # Test cases
        test_cases = [
            {
                'prediction': '5G NR is the radio access technology for 5G networks.',
                'ground_truth': '5G NR (New Radio) is the radio access technology standard for 5G networks.',
                'keywords': ['5G NR', 'radio access technology', '5G networks']
            },
            {
                'prediction': 'HARQ is an error correction mechanism.',
                'ground_truth': 'HARQ (Hybrid Automatic Repeat Request) is an error correction mechanism in 5G that combines forward error correction with automatic repeat request.',
                'keywords': ['HARQ', 'error correction', 'FEC', 'ARQ']
            },
            {
                'prediction': 'Beamforming uses multiple antennas.',
                'ground_truth': 'Beamforming in 5G NR is a technique that uses multiple antennas to focus radio frequency energy in specific directions.',
                'keywords': ['beamforming', 'antennas', 'directional']
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"  Prediction: {test['prediction']}")
            print(f"  Ground Truth: {test['ground_truth']}")
            
            metrics = calculate_all_metrics(
                test['prediction'],
                test['ground_truth'],
                test['keywords']
            )
            
            print(f"  Metrics:")
            for metric, score in metrics.items():
                print(f"    {metric}: {score:.4f}")
        
        # Test batch evaluation
        print(f"\n{'=' * 60}")
        print("Batch Evaluation Test")
        print(f"{'=' * 60}")
        
        predictions = [t['prediction'] for t in test_cases]
        ground_truths = [t['ground_truth'] for t in test_cases]
        keywords_list = [t['keywords'] for t in test_cases]
        
        results = evaluate_predictions(predictions, ground_truths, keywords_list)
        print_evaluation_results(results)
        
        print("✓ Evaluation metrics test completed!")
        return 0
    
    print("Use --test to run test cases")
    return 0


if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)

