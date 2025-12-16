"""
Benchmark Evaluation Script for Fine-tuned Llama 3 8B
Runs comprehensive evaluation on benchmark questions.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

try:
    from benchmark_questions import BENCHMARK_QUESTIONS, get_questions_by_difficulty, get_questions_by_category
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("Warning: benchmark_questions.py not found")

try:
    from test_inference import load_fine_tuned_model, generate_answer
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: test_inference.py not found")

try:
    from evaluation_metrics import (
        calculate_all_metrics,
        exact_match,
        semantic_similarity,
        keyword_match,
        bleu_score,
        f1_score_keywords
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: evaluation_metrics.py not found")

try:
    import config
    OUTPUT_DIR = config.OUTPUT_DIR
    MODEL_NAME = config.MODEL_NAME
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"


def run_benchmark(
    model_path: str = None,
    base_model_name: str = MODEL_NAME,
    questions: List[Dict[str, Any]] = None,
    use_4bit: bool = True,
    use_8bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    accuracy_threshold: float = 0.7,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run benchmark evaluation on fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model (default: OUTPUT_DIR/final_model)
        base_model_name: Base model name (default: from config)
        questions: List of benchmark questions (default: all questions)
        use_4bit: Whether to use 4-bit quantization (default: True)
        use_8bit: Whether to use 8-bit quantization (default: False)
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        top_k: Top-k sampling parameter (default: 50)
        do_sample: Whether to use sampling (default: True)
        accuracy_threshold: Threshold for semantic similarity accuracy (default: 0.7)
        output_file: Path to save results JSON (default: benchmark_results.json)
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print("Benchmark Evaluation")
    print(f"{'=' * 60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    
    # Check dependencies
    if not BENCHMARK_AVAILABLE:
        raise ImportError("benchmark_questions.py is required")
    if not INFERENCE_AVAILABLE:
        raise ImportError("test_inference.py is required")
    if not METRICS_AVAILABLE:
        raise ImportError("evaluation_metrics.py is required")
    
    # Determine model path
    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Get questions
    if questions is None:
        questions = BENCHMARK_QUESTIONS
    
    print(f"Loading model from: {model_path}")
    print(f"Number of questions: {len(questions)}")
    print(f"Accuracy threshold: {accuracy_threshold}")
    print()
    
    # Load model
    print("Loading fine-tuned model...")
    model, tokenizer = load_fine_tuned_model(
        model_path=model_path,
        base_model_name=base_model_name,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
    )
    print("✓ Model loaded\n")
    
    # Prepare generation kwargs
    generation_kwargs = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'do_sample': do_sample,
    }
    
    # Run evaluation
    print(f"{'=' * 60}")
    print("Running Benchmark Evaluation")
    print(f"{'=' * 60}\n")
    
    results = []
    total_inference_time = 0.0
    
    for i, question_data in enumerate(questions, 1):
        question_id = question_data.get('id', f'Q{i}')
        question_text = question_data['question']
        ground_truth = question_data['ground_truth']
        keywords = question_data.get('keywords', [])
        difficulty = question_data.get('difficulty', 'Unknown')
        category = question_data.get('category', 'Unknown')
        
        print(f"[{i}/{len(questions)}] {question_id} ({difficulty}) - {category}")
        print(f"  Question: {question_text[:80]}...")
        
        # Generate answer
        try:
            inference_result = generate_answer(
                model, tokenizer, question_text, **generation_kwargs
            )
            prediction = inference_result['answer']
            inference_time = inference_result['inference_time']
            total_inference_time += inference_time
            
            # Calculate metrics
            metrics = calculate_all_metrics(
                prediction=prediction,
                ground_truth=ground_truth,
                keywords=keywords,
                verbose=False
            )
            
            # Determine if answer is accurate (semantic similarity >= threshold)
            is_accurate = metrics['semantic_similarity'] >= accuracy_threshold
            
            # Store result
            result = {
                'question_id': question_id,
                'question': question_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'keywords': keywords,
                'difficulty': difficulty,
                'category': category,
                'metrics': metrics,
                'is_accurate': is_accurate,
                'inference_time': inference_time,
                'tokens_generated': inference_result['tokens_generated'],
            }
            
            results.append(result)
            
            # Print quick summary
            print(f"  ✓ Generated ({inference_time:.2f}s)")
            print(f"    Semantic similarity: {metrics['semantic_similarity']:.4f}")
            print(f"    Keyword match: {metrics['keyword_match']:.4f}")
            print(f"    Accurate: {'Yes' if is_accurate else 'No'}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Store error result
            result = {
                'question_id': question_id,
                'question': question_text,
                'ground_truth': ground_truth,
                'prediction': None,
                'error': str(e),
                'difficulty': difficulty,
                'category': category,
                'metrics': None,
                'is_accurate': False,
            }
            results.append(result)
            print()
    
    # Compute aggregate statistics
    print(f"{'=' * 60}")
    print("Computing Aggregate Statistics")
    print(f"{'=' * 60}\n")
    
    # Filter out error results
    valid_results = [r for r in results if r.get('metrics') is not None]
    
    if not valid_results:
        print("No valid results to analyze.")
        return {'error': 'No valid results'}
    
    # Overall statistics
    total_questions = len(valid_results)
    accurate_count = sum(1 for r in valid_results if r.get('is_accurate', False))
    overall_accuracy = accurate_count / total_questions if total_questions > 0 else 0.0
    
    # Average metrics
    avg_metrics = {
        'exact_match': sum(r['metrics']['exact_match'] for r in valid_results) / total_questions,
        'semantic_similarity': sum(r['metrics']['semantic_similarity'] for r in valid_results) / total_questions,
        'keyword_match': sum(r['metrics']['keyword_match'] for r in valid_results) / total_questions,
        'bleu_score': sum(r['metrics']['bleu_score'] for r in valid_results) / total_questions,
        'f1_score': sum(r['metrics']['f1_score'] for r in valid_results) / total_questions,
    }
    
    # Performance by difficulty
    difficulty_stats = defaultdict(lambda: {'count': 0, 'accurate': 0, 'metrics_sum': defaultdict(float)})
    
    for result in valid_results:
        diff = result['difficulty']
        difficulty_stats[diff]['count'] += 1
        if result.get('is_accurate', False):
            difficulty_stats[diff]['accurate'] += 1
        
        for metric, value in result['metrics'].items():
            difficulty_stats[diff]['metrics_sum'][metric] += value
    
    # Calculate averages by difficulty
    performance_by_difficulty = {}
    for diff, stats in difficulty_stats.items():
        count = stats['count']
        performance_by_difficulty[diff] = {
            'total': count,
            'accurate': stats['accurate'],
            'accuracy': stats['accurate'] / count if count > 0 else 0.0,
            'average_metrics': {
                metric: stats['metrics_sum'][metric] / count
                for metric in stats['metrics_sum']
            }
        }
    
    # Performance by category
    category_stats = defaultdict(lambda: {'count': 0, 'accurate': 0, 'metrics_sum': defaultdict(float)})
    
    for result in valid_results:
        cat = result['category']
        category_stats[cat]['count'] += 1
        if result.get('is_accurate', False):
            category_stats[cat]['accurate'] += 1
        
        for metric, value in result['metrics'].items():
            category_stats[cat]['metrics_sum'][metric] += value
    
    # Calculate averages by category
    performance_by_category = {}
    for cat, stats in category_stats.items():
        count = stats['count']
        performance_by_category[cat] = {
            'total': count,
            'accurate': stats['accurate'],
            'accuracy': stats['accurate'] / count if count > 0 else 0.0,
            'average_metrics': {
                metric: stats['metrics_sum'][metric] / count
                for metric in stats['metrics_sum']
            }
        }
    
    # Compile final results
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'base_model': base_model_name,
        'total_questions': len(questions),
        'valid_results': len(valid_results),
        'total_inference_time': total_inference_time,
        'average_inference_time': total_inference_time / len(valid_results) if valid_results else 0.0,
        'overall_accuracy': overall_accuracy,
        'accuracy_threshold': accuracy_threshold,
        'average_metrics': avg_metrics,
        'performance_by_difficulty': performance_by_difficulty,
        'performance_by_category': performance_by_category,
        'detailed_results': results,
    }
    
    # Save results to JSON
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {output_path}\n")
    
    return benchmark_results


def print_summary_table(results: Dict[str, Any]):
    """
    Print summary table of benchmark results.
    
    Args:
        results: Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print("Benchmark Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Model: {results.get('base_model', 'N/A')}")
    print(f"Total questions: {results.get('total_questions', 0)}")
    print(f"Valid results: {results.get('valid_results', 0)}")
    print(f"Total inference time: {results.get('total_inference_time', 0):.2f} seconds")
    print(f"Average inference time: {results.get('average_inference_time', 0):.3f} seconds")
    print()
    
    # Overall metrics
    avg_metrics = results.get('average_metrics', {})
    overall_accuracy = results.get('overall_accuracy', 0.0)
    threshold = results.get('accuracy_threshold', 0.7)
    
    print(f"Overall Accuracy (semantic similarity ≥ {threshold}): {overall_accuracy:.2%}")
    print(f"\nAverage Metrics:")
    print(f"  Exact Match:        {avg_metrics.get('exact_match', 0):.4f}")
    print(f"  Semantic Similarity: {avg_metrics.get('semantic_similarity', 0):.4f}")
    print(f"  Keyword Match:      {avg_metrics.get('keyword_match', 0):.4f}")
    print(f"  BLEU Score:         {avg_metrics.get('bleu_score', 0):.4f}")
    print(f"  F1 Score:           {avg_metrics.get('f1_score', 0):.4f}")
    
    # Performance by difficulty
    print(f"\n{'=' * 60}")
    print("Performance by Difficulty")
    print(f"{'=' * 60}")
    print(f"{'Difficulty':<12} {'Total':<8} {'Accurate':<10} {'Accuracy':<10} {'Sem. Sim.':<10} {'Keyword':<10} {'BLEU':<10} {'F1':<10}")
    print("-" * 80)
    
    perf_by_diff = results.get('performance_by_difficulty', {})
    for difficulty in ['Easy', 'Medium', 'Hard']:
        if difficulty in perf_by_diff:
            stats = perf_by_diff[difficulty]
            metrics = stats.get('average_metrics', {})
            print(f"{difficulty:<12} {stats['total']:<8} {stats['accurate']:<10} "
                  f"{stats['accuracy']:<10.2%} {metrics.get('semantic_similarity', 0):<10.4f} "
                  f"{metrics.get('keyword_match', 0):<10.4f} {metrics.get('bleu_score', 0):<10.4f} "
                  f"{metrics.get('f1_score', 0):<10.4f}")
    
    # Performance by category
    print(f"\n{'=' * 60}")
    print("Performance by Category")
    print(f"{'=' * 60}")
    print(f"{'Category':<25} {'Total':<8} {'Accurate':<10} {'Accuracy':<10} {'Sem. Sim.':<10}")
    print("-" * 75)
    
    perf_by_cat = results.get('performance_by_category', {})
    for category in sorted(perf_by_cat.keys()):
        stats = perf_by_cat[category]
        metrics = stats.get('average_metrics', {})
        print(f"{category:<25} {stats['total']:<8} {stats['accurate']:<10} "
              f"{stats['accuracy']:<10.2%} {metrics.get('semantic_similarity', 0):<10.4f}")
    
    print(f"\n{'=' * 60}\n")


def main():
    """Main function to run benchmark evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on fine-tuned model"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to fine-tuned model directory (default: OUTPUT_DIR/final_model)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=MODEL_NAME,
        help=f'Base model name (default: {MODEL_NAME})'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['Easy', 'Medium', 'Hard'],
        default=None,
        help='Filter questions by difficulty (default: all)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter questions by category (default: all)'
    )
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        default=True,
        help='Use 4-bit quantization (default: True)'
    )
    parser.add_argument(
        '--use-8bit',
        action='store_true',
        default=False,
        help='Use 8-bit quantization (default: False)'
    )
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        default=False,
        help='Disable quantization (default: False)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--accuracy-threshold',
        type=float,
        default=0.7,
        help='Semantic similarity threshold for accuracy (default: 0.7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON (default: OUTPUT_DIR/benchmark_results.json)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not BENCHMARK_AVAILABLE:
        print("Error: benchmark_questions.py is required")
        return 1
    if not INFERENCE_AVAILABLE:
        print("Error: test_inference.py is required")
        return 1
    if not METRICS_AVAILABLE:
        print("Error: evaluation_metrics.py is required")
        return 1
    
    # Get questions
    questions = BENCHMARK_QUESTIONS
    
    if args.difficulty:
        questions = get_questions_by_difficulty(args.difficulty)
        print(f"Filtered to {args.difficulty} questions: {len(questions)}")
    
    if args.category:
        questions = get_questions_by_category(args.category)
        print(f"Filtered to {args.category} category: {len(questions)}")
    
    if not questions:
        print("No questions found with specified filters.")
        return 1
    
    # Determine quantization settings
    use_4bit = args.use_4bit and not args.no_quantization and not args.use_8bit
    use_8bit = args.use_8bit and not args.no_quantization
    
    try:
        # Run benchmark
        results = run_benchmark(
            model_path=args.model_path,
            base_model_name=args.base_model,
            questions=questions,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            accuracy_threshold=args.accuracy_threshold,
            output_file=args.output,
        )
        
        # Print summary
        print_summary_table(results)
        
        print("✓ Benchmark evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during benchmark evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

