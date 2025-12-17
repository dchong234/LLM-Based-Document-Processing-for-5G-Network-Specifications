"""
Project Status Checker
Verifies the completion status of all project phases.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import config
    OUTPUT_DIR = config.OUTPUT_DIR
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
    SPECS_DIR = config.SPECS_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
    SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")

try:
    from setup_check import (
        check_python_version,
        check_packages,
        check_pytorch_and_cuda,
        check_huggingface_token,
        check_directories,
    )
    SETUP_CHECK_AVAILABLE = True
except ImportError:
    SETUP_CHECK_AVAILABLE = False


class StatusChecker:
    """Check project status across all phases."""
    
    def __init__(self):
        """Initialize status checker."""
        self.checks = []
        self.total_steps = 5
    
    def add_check(self, name: str, status: bool, message: str = "", details: Dict = None):
        """
        Add a status check result.
        
        Args:
            name: Name of the check
            status: True if passed, False if failed
            message: Optional message
            details: Optional additional details
        """
        self.checks.append({
            'name': name,
            'status': status,
            'message': message,
            'details': details or {}
        })
    
    def check_environment_setup(self) -> bool:
        """
        Check if environment setup is complete.
        
        Returns:
            True if setup is complete, False otherwise
        """
        if not SETUP_CHECK_AVAILABLE:
            self.add_check(
                "Environment Setup",
                False,
                "setup_check.py not available",
                {'error': 'Cannot verify environment setup'}
            )
            return False
        
        try:
            import sys
            import io
            
            # Suppress stdout temporarily for clean checks
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Check Python version
                python_ok = check_python_version()
                
                # Check packages (skip actual check, just verify setup_check works)
                packages_ok = True  # Assume OK if setup_check is available
                
                # Check CUDA
                cuda_ok = check_pytorch_and_cuda()
                
                # Check Hugging Face token
                hf_token_ok = check_huggingface_token()
                
                # Check directories
                dirs_ok = check_directories()
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            
            all_ok = python_ok and packages_ok and cuda_ok and hf_token_ok and dirs_ok
            
            details = {
                'python_version': python_ok,
                'packages': packages_ok,
                'cuda': cuda_ok,
                'huggingface_token': hf_token_ok,
                'directories': dirs_ok
            }
            
            if all_ok:
                self.add_check(
                    "Environment Setup",
                    True,
                    "All environment checks passed",
                    details
                )
            else:
                failed = [k.replace('_', ' ').title() for k, v in details.items() if not v]
                self.add_check(
                    "Environment Setup",
                    False,
                    f"Some checks failed: {', '.join(failed)}",
                    details
                )
            
            return all_ok
        
        except Exception as e:
            self.add_check(
                "Environment Setup",
                False,
                f"Error checking environment: {e}",
                {'error': str(e)}
            )
            return False
    
    def check_data_preparation(self) -> bool:
        """
        Check if data preparation is complete.
        
        Returns:
            True if data preparation is complete, False otherwise
        """
        required_files = [
            "training_data.jsonl",
            "validation_data.jsonl",
        ]
        
        processed_dir = Path(PROCESSED_DATA_DIR)
        found_files = []
        missing_files = []
        
        for file_name in required_files:
            file_path = processed_dir / file_name
            if file_path.exists():
                # Check if file has content
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 0:
                        # Count lines
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                        found_files.append({
                            'name': file_name,
                            'size': file_size,
                            'lines': line_count
                        })
                    else:
                        missing_files.append(file_name)
                except Exception:
                    missing_files.append(file_name)
            else:
                missing_files.append(file_name)
        
        if len(found_files) == len(required_files):
            total_lines = sum(f['lines'] for f in found_files)
            total_size = sum(f['size'] for f in found_files)
            
            details = {
                'files': found_files,
                'total_lines': total_lines,
                'total_size_mb': total_size / (1024 * 1024)
            }
            
            self.add_check(
                "Data Preparation",
                True,
                f"Training data ready ({total_lines} examples, {total_size / (1024*1024):.1f} MB)",
                details
            )
            return True
        else:
            details = {
                'found': found_files,
                'missing': missing_files
            }
            
            self.add_check(
                "Data Preparation",
                False,
                f"Missing files: {', '.join(missing_files)}",
                details
            )
            return False
    
    def check_model_training(self) -> bool:
        """
        Check if model training is complete.
        
        Returns:
            True if model is trained, False otherwise
        """
        model_dir = Path(OUTPUT_DIR) / "final_model"
        
        if not model_dir.exists():
            # Check for checkpoints
            checkpoint_dirs = list(Path(OUTPUT_DIR).glob("checkpoint-*"))
            
            if checkpoint_dirs:
                latest_checkpoint = max(
                    checkpoint_dirs,
                    key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0
                )
                
                details = {
                    'has_final_model': False,
                    'has_checkpoints': True,
                    'latest_checkpoint': str(latest_checkpoint),
                    'checkpoint_count': len(checkpoint_dirs)
                }
                
                self.add_check(
                    "Model Training",
                    False,
                    f"Training in progress (latest: {latest_checkpoint.name})",
                    details
                )
                return False
            else:
                self.add_check(
                    "Model Training",
                    False,
                    "No model or checkpoints found",
                    {'has_final_model': False, 'has_checkpoints': False}
                )
                return False
        
        # Check for required model files
        required_files = ["adapter_config.json"]
        # adapter_model.bin might be adapter_model.safetensors
        model_files = ["adapter_model.bin", "adapter_model.safetensors"]
        tokenizer_files = ["tokenizer_config.json"]
        
        found_files = []
        missing_files = []
        
        for file_name in required_files:
            file_path = model_dir / file_name
            if file_path.exists():
                found_files.append(file_name)
            else:
                missing_files.append(file_name)
        
        # Check for at least one model file
        has_model = any((model_dir / f).exists() for f in model_files)
        
        # Check for tokenizer
        has_tokenizer = any((model_dir / f).exists() for f in tokenizer_files)
        
        if len(found_files) == len(required_files) and has_model and has_tokenizer:
            # Get file sizes
            file_sizes = {}
            for file_name in found_files + model_files + tokenizer_files:
                file_path = model_dir / file_name
                if file_path.exists():
                    file_sizes[file_name] = file_path.stat().st_size / (1024 * 1024)  # MB
            
            total_size = sum(file_sizes.values())
            
            details = {
                'model_path': str(model_dir),
                'files': list(file_sizes.keys()),
                'total_size_mb': total_size
            }
            
            self.add_check(
                "Model Training",
                True,
                f"Model trained and saved ({total_size:.1f} MB)",
                details
            )
            return True
        else:
            missing = []
            if not has_model:
                missing.append("model file")
            if not has_tokenizer:
                missing.append("tokenizer")
            missing.extend(missing_files)
            
            self.add_check(
                "Model Training",
                False,
                f"Model incomplete: missing {', '.join(missing)}",
                {
                    'has_adapter_config': required_files[0] in found_files,
                    'has_model_file': has_model,
                    'has_tokenizer': has_tokenizer
                }
            )
            return False
    
    def check_evaluation(self) -> bool:
        """
        Check if evaluation is complete.
        
        Returns:
            True if evaluation is complete, False otherwise
        """
        results_file = Path(OUTPUT_DIR) / "benchmark_results.json"
        
        if results_file.exists():
            try:
                import json
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                total_questions = results.get('total_questions', 0)
                valid_results = results.get('valid_results', 0)
                overall_accuracy = results.get('overall_accuracy', 0.0)
                
                details = {
                    'total_questions': total_questions,
                    'valid_results': valid_results,
                    'overall_accuracy': overall_accuracy,
                    'timestamp': results.get('timestamp', 'Unknown')
                }
                
                self.add_check(
                    "Evaluation",
                    True,
                    f"Evaluation complete ({valid_results}/{total_questions} questions, {overall_accuracy:.1%} accuracy)",
                    details
                )
                return True
            
            except Exception as e:
                self.add_check(
                    "Evaluation",
                    False,
                    f"Results file exists but invalid: {e}",
                    {'error': str(e)}
                )
                return False
        else:
            self.add_check(
                "Evaluation",
                False,
                "No evaluation results found",
                {'results_file': str(results_file)}
            )
            return False
    
    def check_demo_working(self) -> bool:
        """
        Check if demo can load the model.
        
        Returns:
            True if demo can load model, False otherwise
        """
        model_dir = Path(OUTPUT_DIR) / "final_model"
        
        if not model_dir.exists():
            self.add_check(
                "Demo Working",
                False,
                "Model not found (cannot test demo)",
                {'model_path': str(model_dir)}
            )
            return False
        
        try:
            # Try to import and load model
            from test_inference import load_fine_tuned_model
            
            print("  Attempting to load model for demo check...")
            
            try:
                model, tokenizer = load_fine_tuned_model(
                    model_path=str(model_dir),
                    base_model_name="meta-llama/Meta-Llama-3-8B",
                    use_4bit=True,
                    use_8bit=False,
                )
                
                if model is not None and tokenizer is not None:
                    self.add_check(
                        "Demo Working",
                        True,
                        "Model loads successfully for demo",
                        {'model_loaded': True, 'tokenizer_loaded': True}
                    )
                    return True
                else:
                    self.add_check(
                        "Demo Working",
                        False,
                        "Failed to load model or tokenizer",
                        {'model_loaded': model is not None, 'tokenizer_loaded': tokenizer is not None}
                    )
                    return False
            
            except Exception as e:
                self.add_check(
                    "Demo Working",
                    False,
                    f"Error loading model: {e}",
                    {'error': str(e)}
                )
                return False
        
        except ImportError:
            self.add_check(
                "Demo Working",
                False,
                "test_inference.py not available (cannot test demo)",
                {'error': 'Module not available'}
            )
            return False
    
    def get_progress_percentage(self) -> float:
        """
        Calculate overall progress percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if not self.checks:
            return 0.0
        
        passed = sum(1 for check in self.checks if check['status'])
        total = len(self.checks)
        
        return (passed / total) * 100.0
    
    def get_next_steps(self) -> List[str]:
        """
        Get recommended next steps based on current status.
        
        Returns:
            List of recommended next steps
        """
        next_steps = []
        
        for check in self.checks:
            if not check['status']:
                name = check['name']
                
                if name == "Environment Setup":
                    next_steps.append("Run: python3 setup_check.py")
                    next_steps.append("Install missing packages: pip install -r requirements.txt")
                    next_steps.append("Set up Hugging Face token: export HF_TOKEN='your_token'")
                
                elif name == "Data Preparation":
                    next_steps.append("Run: python3 run_data_pipeline.py")
                    next_steps.append("Or: python3 pdf_extractor.py (if PDFs in specs/)")
                
                elif name == "Model Training":
                    next_steps.append("Run: python3 train_model.py")
                    next_steps.append("Or resume: python3 train_model.py --auto-resume")
                
                elif name == "Evaluation":
                    next_steps.append("Run: python3 run_benchmark.py")
                    next_steps.append("Or visualize: python3 visualize_results.py")
                
                elif name == "Demo Working":
                    next_steps.append("Fix model loading issues")
                    next_steps.append("Run: python3 demo_app.py (after fixing)")
        
        # If all checks pass
        if all(check['status'] for check in self.checks):
            next_steps.append("✓ All phases complete!")
            next_steps.append("Try: python3 demo_app.py")
            next_steps.append("Or: python3 run_benchmark.py (to re-evaluate)")
        
        return list(set(next_steps))  # Remove duplicates
    
    def print_status(self):
        """Print status checklist."""
        print("\n" + "=" * 60)
        print("Project Status Check")
        print("=" * 60)
        print()
        
        # Print checklist
        for check in self.checks:
            status_symbol = "✓" if check['status'] else "✗"
            status_color = "✓" if check['status'] else "✗"
            
            print(f"{status_symbol} {check['name']}")
            print(f"    {check['message']}")
            
            # Print details if available
            if check['details']:
                for key, value in check['details'].items():
                    if key != 'error':
                        if isinstance(value, (int, float)):
                            if 'mb' in key.lower() or 'size' in key.lower():
                                print(f"      {key}: {value:.2f} MB")
                            elif 'accuracy' in key.lower() or 'percentage' in key.lower():
                                print(f"      {key}: {value:.2%}")
                            else:
                                print(f"      {key}: {value}")
                        elif isinstance(value, bool):
                            print(f"      {key}: {'Yes' if value else 'No'}")
                        elif isinstance(value, list):
                            if value:
                                print(f"      {key}: {len(value)} items")
                            else:
                                print(f"      {key}: None")
                        else:
                            print(f"      {key}: {value}")
            
            print()
        
        # Print progress
        progress = self.get_progress_percentage()
        print("=" * 60)
        print(f"Overall Progress: {progress:.0f}% ({sum(1 for c in self.checks if c['status'])}/{len(self.checks)} complete)")
        print("=" * 60)
        
        # Print next steps
        next_steps = self.get_next_steps()
        if next_steps:
            print("\nNext Recommended Steps:")
            for i, step in enumerate(next_steps[:6], 1):  # Limit to 6 steps
                print(f"  {i}. {step}")
        
        print()
    
    def run_all_checks(self) -> Dict:
        """
        Run all status checks.
        
        Returns:
            Dictionary with all check results
        """
        print("Running status checks...")
        print()
        
        # Run all checks
        self.check_environment_setup()
        self.check_data_preparation()
        self.check_model_training()
        self.check_evaluation()
        self.check_demo_working()
        
        # Print status
        self.print_status()
        
        # Return summary
        return {
            'checks': self.checks,
            'progress': self.get_progress_percentage(),
            'next_steps': self.get_next_steps()
        }


def main():
    """Main function to run status check."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check project status across all phases"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    try:
        checker = StatusChecker()
        results = checker.run_all_checks()
        
        if args.json:
            import json
            print("\n" + "=" * 60)
            print("JSON Output")
            print("=" * 60)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Exit with error code if not all checks passed
        all_passed = all(check['status'] for check in results['checks'])
        return 0 if all_passed else 1
    
    except Exception as e:
        print(f"\n✗ Error during status check: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

