"""
Integration Test for Full Pipeline
Runs a quick end-to-end test with minimal data and training steps.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch is not installed")

try:
    from test_inference import load_fine_tuned_model, generate_answer
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: test_inference.py not found")

try:
    from evaluation_metrics import calculate_all_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: evaluation_metrics.py not found")

try:
    import config
    MODEL_NAME = config.MODEL_NAME
except ImportError:
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"


# Small sample dataset for testing
SAMPLE_DATASET = [
    {
        "instruction": "What is 5G NR?",
        "context": "5G specifications",
        "response": "5G NR (New Radio) is the radio access technology standard for 5G networks. It defines the physical layer and medium access control layer specifications for 5G wireless communication systems.",
        "source": "test_spec.json",
        "section": "1.1"
    },
    {
        "instruction": "What are the frequency ranges in 5G?",
        "context": "5G frequency bands",
        "response": "5G operates in two main frequency ranges: FR1 (Frequency Range 1) covering sub-6 GHz bands and FR2 (Frequency Range 2) covering millimeter wave bands.",
        "source": "test_spec.json",
        "section": "2.1"
    },
    {
        "instruction": "What is numerology in 5G?",
        "context": "5G physical layer",
        "response": "Numerology in 5G refers to the subcarrier spacing configuration. It defines the spacing between subcarriers in the OFDM waveform.",
        "source": "test_spec.json",
        "section": "3.1"
    },
    {
        "instruction": "Explain HARQ in 5G",
        "context": "5G protocols",
        "response": "HARQ (Hybrid Automatic Repeat Request) is an error correction mechanism in 5G that combines forward error correction with automatic repeat request.",
        "source": "test_spec.json",
        "section": "4.1"
    },
    {
        "instruction": "What is beamforming?",
        "context": "5G antenna technology",
        "response": "Beamforming in 5G NR is a technique that uses multiple antennas to focus radio frequency energy in specific directions, creating directional beams.",
        "source": "test_spec.json",
        "section": "5.1"
    },
    {
        "instruction": "What is network slicing?",
        "context": "5G architecture",
        "response": "Network slicing in 5G allows the creation of multiple virtual networks on top of a shared physical infrastructure, each tailored to specific use cases.",
        "source": "test_spec.json",
        "section": "6.1"
    },
    {
        "instruction": "Explain carrier aggregation",
        "context": "5G bandwidth",
        "response": "Carrier aggregation in 5G NR allows multiple component carriers to be aggregated to increase bandwidth and data rates.",
        "source": "test_spec.json",
        "section": "7.1"
    },
    {
        "instruction": "What is dual connectivity?",
        "context": "5G connectivity",
        "response": "Dual connectivity in 5G NR allows a UE to simultaneously connect to two base stations: a master node and a secondary node.",
        "source": "test_spec.json",
        "section": "8.1"
    },
    {
        "instruction": "What are bandwidth parts?",
        "context": "5G resource allocation",
        "response": "A bandwidth part (BWP) in 5G NR is a contiguous set of physical resource blocks configured within a carrier.",
        "source": "test_spec.json",
        "section": "9.1"
    },
    {
        "instruction": "Explain modulation schemes in 5G",
        "context": "5G physical layer",
        "response": "5G NR uses QPSK, 16-QAM, 64-QAM, and 256-QAM modulation schemes depending on channel conditions and data rate requirements.",
        "source": "test_spec.json",
        "section": "10.1"
    }
]


class IntegrationTest:
    """Integration test runner."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize integration test.
        
        Args:
            output_dir: Directory for test outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'steps': {},
            'errors': [],
            'warnings': []
        }
    
    def log_step(self, step_name: str, status: str, message: str = ""):
        """
        Log a test step.
        
        Args:
            step_name: Name of the step
            status: "success", "failure", "warning", or "skipped"
            message: Optional message
        """
        self.test_results['steps'][step_name] = {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        status_symbol = {
            'success': '✓',
            'failure': '✗',
            'warning': '⚠',
            'skipped': '⊘'
        }.get(status, '?')
        
        print(f"{status_symbol} {step_name}: {message}")
    
    def create_sample_dataset(self) -> Path:
        """
        Create a small sample dataset for testing.
        
        Returns:
            Path to the training JSONL file
        """
        print("\n" + "=" * 60)
        print("Step 1: Creating Sample Dataset")
        print("=" * 60)
        
        try:
            train_file = self.output_dir / "train_sample.jsonl"
            val_file = self.output_dir / "val_sample.jsonl"
            
            # Split into train (8) and validation (2)
            train_examples = SAMPLE_DATASET[:8]
            val_examples = SAMPLE_DATASET[8:]
            
            # Write training file
            with open(train_file, 'w', encoding='utf-8') as f:
                for example in train_examples:
                    f.write(json.dumps(example) + '\n')
            
            # Write validation file
            with open(val_file, 'w', encoding='utf-8') as f:
                for example in val_examples:
                    f.write(json.dumps(example) + '\n')
            
            print(f"✓ Created training file: {train_file} ({len(train_examples)} examples)")
            print(f"✓ Created validation file: {val_file} ({len(val_examples)} examples)")
            
            self.log_step("create_dataset", "success", f"Created {len(train_examples)} train + {len(val_examples)} val examples")
            
            return train_file, val_file
        
        except Exception as e:
            self.log_step("create_dataset", "failure", str(e))
            self.test_results['errors'].append(f"Dataset creation failed: {e}")
            raise
    
    def run_training(self, train_file: Path, val_file: Path) -> Path:
        """
        Run abbreviated training.
        
        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file
        
        Returns:
            Path to trained model
        """
        print("\n" + "=" * 60)
        print("Step 2: Running Abbreviated Training")
        print("=" * 60)
        
        if not INFERENCE_AVAILABLE:
            self.log_step("training", "skipped", "test_inference.py not available")
            return None
        
        try:
            # Import training modules
            from train_model import train_model
            from training_config import create_training_arguments
            
            # Create a very short training configuration
            model_output_dir = self.output_dir / "test_model"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            print("Training configuration:")
            print(f"  Epochs: 1")
            print(f"  Max steps: 10")
            print(f"  Batch size: 2")
            print(f"  Learning rate: 2e-4")
            print(f"  Output: {model_output_dir}")
            
            # Run training with minimal steps
            try:
                # Use training_kwargs for TrainingArguments parameters
                trainer, model, tokenizer = train_model(
                    train_file=train_file,
                    val_file=val_file,
                    output_dir=str(model_output_dir),
                    model_name=MODEL_NAME,
                    use_4bit=True,
                    use_8bit=False,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    max_steps=10,  # Very short training
                    save_steps=10,
                    eval_steps=5,
                    logging_steps=2,
                    warmup_steps=2,
                    gradient_accumulation_steps=1,
                )
                
                final_model_dir = model_output_dir / "final_model"
                if final_model_dir.exists():
                    self.log_step("training", "success", f"Model saved to {final_model_dir}")
                    return final_model_dir
                else:
                    # If final_model doesn't exist, use last checkpoint
                    checkpoints = list(model_output_dir.glob("checkpoint-*"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                        self.log_step("training", "success", f"Using checkpoint: {latest_checkpoint}")
                        return latest_checkpoint
                    else:
                        self.log_step("training", "failure", "No model or checkpoint found")
                        return None
            
            except Exception as e:
                # Training might fail due to dependencies, log but continue
                self.log_step("training", "warning", f"Training error (expected if dependencies missing): {e}")
                self.test_results['warnings'].append(f"Training: {e}")
                return None
        
        except ImportError as e:
            self.log_step("training", "skipped", f"Training modules not available: {e}")
            return None
    
    def test_inference(self, model_path: Path) -> Dict[str, Any]:
        """
        Test inference on sample questions.
        
        Args:
            model_path: Path to trained model
        
        Returns:
            Dictionary with inference results
        """
        print("\n" + "=" * 60)
        print("Step 3: Testing Inference")
        print("=" * 60)
        
        if not model_path or not model_path.exists():
            self.log_step("inference", "skipped", "Model not available")
            return {}
        
        if not INFERENCE_AVAILABLE:
            self.log_step("inference", "skipped", "test_inference.py not available")
            return {}
        
        try:
            # Load model
            print("Loading model...")
            model, tokenizer = load_fine_tuned_model(
                model_path=str(model_path),
                base_model_name=MODEL_NAME,
                use_4bit=True,
                use_8bit=False,
            )
            print("✓ Model loaded")
            
            # Test questions
            test_questions = [
                "What is 5G NR?",
                "What is numerology in 5G?",
                "Explain HARQ",
            ]
            
            results = []
            
            for question in test_questions:
                print(f"\nQuestion: {question}")
                
                start_time = time.time()
                result = generate_answer(
                    model,
                    tokenizer,
                    question,
                    max_new_tokens=100,
                    temperature=0.7,
                )
                inference_time = time.time() - start_time
                
                answer = result['answer']
                tokens = result['tokens_generated']
                
                print(f"  Answer: {answer[:100]}...")
                print(f"  Time: {inference_time:.2f}s, Tokens: {tokens}")
                
                results.append({
                    'question': question,
                    'answer': answer,
                    'inference_time': inference_time,
                    'tokens_generated': tokens,
                })
            
            self.log_step("inference", "success", f"Tested {len(test_questions)} questions")
            
            return {'results': results}
        
        except Exception as e:
            self.log_step("inference", "failure", str(e))
            self.test_results['errors'].append(f"Inference failed: {e}")
            return {}
    
    def test_metrics(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test evaluation metrics calculation.
        
        Args:
            inference_results: Results from inference testing
        
        Returns:
            Dictionary with metrics results
        """
        print("\n" + "=" * 60)
        print("Step 4: Testing Evaluation Metrics")
        print("=" * 60)
        
        if not METRICS_AVAILABLE:
            self.log_step("metrics", "skipped", "evaluation_metrics.py not available")
            return {}
        
        if not inference_results.get('results'):
            self.log_step("metrics", "skipped", "No inference results available")
            return {}
        
        try:
            # Get ground truth for test questions
            ground_truths = {
                "What is 5G NR?": "5G NR (New Radio) is the radio access technology standard for 5G networks.",
                "What is numerology in 5G?": "Numerology in 5G refers to the subcarrier spacing configuration.",
                "Explain HARQ": "HARQ (Hybrid Automatic Repeat Request) is an error correction mechanism.",
            }
            
            metrics_results = []
            
            for result in inference_results['results']:
                question = result['question']
                prediction = result['answer']
                ground_truth = ground_truths.get(question, "")
                
                if not ground_truth:
                    continue
                
                # Calculate metrics
                metrics = calculate_all_metrics(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    verbose=False
                )
                
                print(f"\nQuestion: {question}")
                print(f"  Exact match: {metrics['exact_match']:.4f}")
                print(f"  Semantic similarity: {metrics['semantic_similarity']:.4f}")
                print(f"  Keyword match: {metrics['keyword_match']:.4f}")
                print(f"  BLEU score: {metrics['bleu_score']:.4f}")
                print(f"  F1 score: {metrics['f1_score']:.4f}")
                
                metrics_results.append({
                    'question': question,
                    'metrics': metrics
                })
            
            if metrics_results:
                # Calculate averages
                avg_metrics = {
                    'exact_match': sum(m['metrics']['exact_match'] for m in metrics_results) / len(metrics_results),
                    'semantic_similarity': sum(m['metrics']['semantic_similarity'] for m in metrics_results) / len(metrics_results),
                    'keyword_match': sum(m['metrics']['keyword_match'] for m in metrics_results) / len(metrics_results),
                    'bleu_score': sum(m['metrics']['bleu_score'] for m in metrics_results) / len(metrics_results),
                    'f1_score': sum(m['metrics']['f1_score'] for m in metrics_results) / len(metrics_results),
                }
                
                print(f"\nAverage Metrics:")
                for metric, value in avg_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                self.log_step("metrics", "success", f"Calculated metrics for {len(metrics_results)} questions")
                
                return {'metrics': metrics_results, 'average_metrics': avg_metrics}
            else:
                self.log_step("metrics", "warning", "No metrics calculated")
                return {}
        
        except Exception as e:
            self.log_step("metrics", "failure", str(e))
            self.test_results['errors'].append(f"Metrics calculation failed: {e}")
            return {}
    
    def run_all_tests(self) -> bool:
        """
        Run all integration tests.
        
        Returns:
            True if all critical tests passed, False otherwise
        """
        print("\n" + "=" * 60)
        print("Integration Test Suite")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Create sample dataset
            train_file, val_file = self.create_sample_dataset()
            
            # Step 2: Run abbreviated training
            model_path = self.run_training(train_file, val_file)
            
            # Step 3: Test inference
            inference_results = self.test_inference(model_path)
            
            # Step 4: Test metrics
            metrics_results = self.test_metrics(inference_results)
            
            # Calculate duration
            self.test_results['end_time'] = time.time()
            self.test_results['duration'] = self.test_results['end_time'] - self.test_results['start_time']
            
            # Print summary
            self.print_summary()
            
            # Determine success (training might be skipped, but dataset and metrics should work)
            critical_steps = ['create_dataset', 'metrics']
            critical_passed = all(
                self.test_results['steps'].get(step, {}).get('status') == 'success'
                for step in critical_steps
            )
            
            return critical_passed
        
        except KeyboardInterrupt:
            print("\n\n⚠ Integration test interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['errors'].append(f"Test suite failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        duration = self.test_results.get('duration', 0)
        
        print("\n" + "=" * 60)
        print("Integration Test Summary")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Target: < 600 seconds (10 minutes)")
        
        if duration > 600:
            print("⚠ Warning: Test exceeded 10 minute target")
        
        print(f"\nSteps:")
        for step_name, step_info in self.test_results['steps'].items():
            status = step_info['status']
            message = step_info['message']
            symbol = {
                'success': '✓',
                'failure': '✗',
                'warning': '⚠',
                'skipped': '⊘'
            }.get(status, '?')
            print(f"  {symbol} {step_name}: {message}")
        
        if self.test_results['errors']:
            print(f"\nErrors ({len(self.test_results['errors'])}):")
            for error in self.test_results['errors']:
                print(f"  ✗ {error}")
        
        if self.test_results['warnings']:
            print(f"\nWarnings ({len(self.test_results['warnings'])}):")
            for warning in self.test_results['warnings']:
                print(f"  ⚠ {warning}")
        
        print("=" * 60)
        
        # Save results
        results_file = self.output_dir / "integration_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {results_file}")


def main():
    """Main function to run integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run integration tests for the full pipeline"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for test files (default: temp directory)'
    )
    parser.add_argument(
        '--keep-outputs',
        action='store_true',
        help='Keep test outputs after completion (default: cleanup)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use temporary directory
        temp_path = tempfile.mkdtemp(prefix="integration_test_")
        output_dir = Path(temp_path)
        if not args.keep_outputs:
            import atexit
            atexit.register(lambda: shutil.rmtree(temp_path, ignore_errors=True))
    
    print(f"Integration test output directory: {output_dir}")
    
    # Run tests
    tester = IntegrationTest(output_dir)
    
    try:
        success = tester.run_all_tests()
        
        if success:
            print("\n✓ Integration tests passed!")
            return 0
        else:
            print("\n✗ Some integration tests failed or were skipped")
            print("  This may be normal if dependencies are missing.")
            return 1
    
    except Exception as e:
        print(f"\n✗ Integration test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import shutil
    exit_code = main()
    sys.exit(exit_code)

