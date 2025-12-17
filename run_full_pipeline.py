"""
Full Pipeline Runner for Llama 3 8B Fine-tuning
Orchestrates all phases: data preparation, training, evaluation, and demo.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import config
    OUTPUT_DIR = config.OUTPUT_DIR
    SPECS_DIR = config.SPECS_DIR
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")
    SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")


class PipelineLogger:
    """Logger for pipeline execution."""
    
    def __init__(self, log_file: Path):
        """
        Initialize pipeline logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def phase_start(self, phase_name: str):
        """Log phase start."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Phase: {phase_name}")
        self.logger.info(f"{'=' * 60}")
    
    def phase_complete(self, phase_name: str, duration: float):
        """Log phase completion."""
        self.logger.info(f"Phase '{phase_name}' completed in {duration:.2f} seconds")
        self.logger.info(f"{'=' * 60}\n")


class PipelineRunner:
    """Main pipeline runner."""
    
    def __init__(self, log_file: Path):
        """
        Initialize pipeline runner.
        
        Args:
            log_file: Path to log file
        """
        self.logger = PipelineLogger(log_file)
        self.start_time = None
        self.phase_times = {}
        self.phase_status = {}
    
    def check_prerequisites(self, phase: str) -> bool:
        """
        Check prerequisites for a phase.
        
        Args:
            phase: Phase name
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        if phase == "data_preparation":
            # Check if specs directory exists and has PDFs
            if not os.path.exists(SPECS_DIR):
                self.logger.warning(f"Specs directory does not exist: {SPECS_DIR}")
                return False
            
            pdf_files = list(Path(SPECS_DIR).glob("*.pdf"))
            if not pdf_files:
                self.logger.warning(f"No PDF files found in {SPECS_DIR}")
                return False
            
            self.logger.info(f"Found {len(pdf_files)} PDF files in specs directory")
            return True
        
        elif phase == "training":
            # Check if training data exists
            train_file = Path(PROCESSED_DATA_DIR) / "training_data.jsonl"
            if not train_file.exists():
                self.logger.warning(f"Training data not found: {train_file}")
                return False
            
            self.logger.info(f"Training data found: {train_file}")
            return True
        
        elif phase == "evaluation":
            # Check if model exists
            model_path = Path(OUTPUT_DIR) / "final_model"
            if not model_path.exists():
                self.logger.warning(f"Model not found: {model_path}")
                return False
            
            self.logger.info(f"Model found: {model_path}")
            return True
        
        elif phase == "demo":
            # Check if model exists
            model_path = Path(OUTPUT_DIR) / "final_model"
            if not model_path.exists():
                self.logger.warning(f"Model not found: {model_path}")
                return False
            
            self.logger.info(f"Model found: {model_path}")
            return True
        
        return True
    
    def check_data_preparation_done(self) -> bool:
        """
        Check if data preparation is already done.
        
        Returns:
            True if data preparation is complete, False otherwise
        """
        required_files = [
            "training_data.jsonl",
            "validation_data.jsonl",
        ]
        
        for file in required_files:
            file_path = Path(PROCESSED_DATA_DIR) / file
            if not file_path.exists():
                return False
        
        self.logger.info("Data preparation appears to be complete")
        return True
    
    def run_data_preparation(self) -> Tuple[bool, float]:
        """
        Run data preparation phase.
        
        Returns:
            Tuple of (success, duration)
        """
        self.logger.phase_start("Data Preparation")
        
        start_time = time.time()
        
        try:
            # Check if already done
            if self.check_data_preparation_done():
                self.logger.info("Data preparation already complete. Skipping.")
                duration = time.time() - start_time
                return True, duration
            
            # Check prerequisites
            if not self.check_prerequisites("data_preparation"):
                self.logger.error("Prerequisites not met for data preparation")
                duration = time.time() - start_time
                return False, duration
            
            # Run data pipeline
            self.logger.info("Running data preparation pipeline...")
            result = subprocess.run(
                [sys.executable, "run_data_pipeline.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False,
                text=True
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.phase_complete("Data Preparation", duration)
                return True, duration
            else:
                self.logger.error(f"Data preparation failed with return code {result.returncode}")
                return False, duration
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error during data preparation: {e}")
            return False, duration
    
    def run_training(self, resume: bool = False) -> Tuple[bool, float]:
        """
        Run training phase.
        
        Args:
            resume: Whether to resume from checkpoint (default: False)
        
        Returns:
            Tuple of (success, duration)
        """
        self.logger.phase_start("Model Fine-tuning")
        
        start_time = time.time()
        
        try:
            # Check prerequisites
            if not self.check_prerequisites("training"):
                self.logger.error("Prerequisites not met for training")
                duration = time.time() - start_time
                return False, duration
            
            # Build command
            cmd = [sys.executable, "train_model.py"]
            if resume:
                cmd.append("--auto-resume")
            
            self.logger.info("Running model training...")
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False,
                text=True
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.phase_complete("Model Fine-tuning", duration)
                return True, duration
            else:
                self.logger.error(f"Training failed with return code {result.returncode}")
                return False, duration
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error during training: {e}")
            return False, duration
    
    def run_evaluation(self) -> Tuple[bool, float]:
        """
        Run evaluation phase.
        
        Returns:
            Tuple of (success, duration)
        """
        self.logger.phase_start("Benchmark Evaluation")
        
        start_time = time.time()
        
        try:
            # Check prerequisites
            if not self.check_prerequisites("evaluation"):
                self.logger.error("Prerequisites not met for evaluation")
                duration = time.time() - start_time
                return False, duration
            
            # Run benchmark evaluation
            self.logger.info("Running benchmark evaluation...")
            result = subprocess.run(
                [sys.executable, "run_benchmark.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False,
                text=True
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.phase_complete("Benchmark Evaluation", duration)
                return True, duration
            else:
                self.logger.error(f"Evaluation failed with return code {result.returncode}")
                return False, duration
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error during evaluation: {e}")
            return False, duration
    
    def run_demo(self) -> Tuple[bool, float]:
        """
        Run demo app phase.
        
        Returns:
            Tuple of (success, duration) - note: demo runs indefinitely
        """
        self.logger.phase_start("Demo App")
        
        start_time = time.time()
        
        try:
            # Check prerequisites
            if not self.check_prerequisites("demo"):
                self.logger.error("Prerequisites not met for demo")
                duration = time.time() - start_time
                return False, duration
            
            # Run demo app
            self.logger.info("Launching demo app...")
            self.logger.info("Demo app will run until interrupted (Ctrl+C)")
            
            result = subprocess.run(
                [sys.executable, "demo_app.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False,
                text=True
            )
            
            duration = time.time() - start_time
            
            # Demo app runs until interrupted, so this may not be reached
            if result.returncode == 0:
                self.logger.phase_complete("Demo App", duration)
                return True, duration
            else:
                self.logger.error(f"Demo app failed with return code {result.returncode}")
                return False, duration
        
        except KeyboardInterrupt:
            duration = time.time() - start_time
            self.logger.info(f"Demo app stopped by user after {duration:.2f} seconds")
            return True, duration
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error during demo: {e}")
            return False, duration
    
    def estimate_time(self, phases: List[str]) -> Dict[str, float]:
        """
        Estimate time for each phase.
        
        Args:
            phases: List of phases to estimate
        
        Returns:
            Dictionary mapping phase names to estimated times (seconds)
        """
        estimates = {
            "data_preparation": 3600.0,  # 1 hour (depends on data size)
            "training": 14400.0,  # 4 hours (depends on data and epochs)
            "evaluation": 600.0,  # 10 minutes (depends on number of questions)
            "demo": 0.0,  # Runs indefinitely
        }
        
        return {phase: estimates.get(phase, 0.0) for phase in phases}
    
    def run_pipeline(
        self,
        skip_data_prep: bool = False,
        skip_training: bool = False,
        skip_evaluation: bool = False,
        skip_demo: bool = False,
        resume_training: bool = False,
    ) -> bool:
        """
        Run the full pipeline.
        
        Args:
            skip_data_prep: Skip data preparation phase (default: False)
            skip_training: Skip training phase (default: False)
            skip_evaluation: Skip evaluation phase (default: False)
            skip_demo: Skip demo phase (default: False)
            resume_training: Resume training from checkpoint (default: False)
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        self.start_time = time.time()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Full Pipeline Execution")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Skip data preparation: {skip_data_prep}")
        self.logger.info(f"Skip training: {skip_training}")
        self.logger.info(f"Skip evaluation: {skip_evaluation}")
        self.logger.info(f"Skip demo: {skip_demo}")
        self.logger.info(f"Resume training: {resume_training}")
        self.logger.info("=" * 60 + "\n")
        
        # Determine phases to run
        phases = []
        if not skip_data_prep:
            phases.append("data_preparation")
        if not skip_training:
            phases.append("training")
        if not skip_evaluation:
            phases.append("evaluation")
        if not skip_demo:
            phases.append("demo")
        
        # Estimate time
        if phases:
            estimates = self.estimate_time(phases)
            total_estimate = sum(estimates.values())
            self.logger.info(f"Estimated time for {len(phases)} phase(s): {total_estimate/3600:.2f} hours")
            self.logger.info("")
        
        # Run phases
        for phase in phases:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Starting phase: {phase}")
            self.logger.info(f"{'=' * 60}\n")
            
            if phase == "data_preparation":
                success, duration = self.run_data_preparation()
            elif phase == "training":
                success, duration = self.run_training(resume=resume_training)
            elif phase == "evaluation":
                success, duration = self.run_evaluation()
            elif phase == "demo":
                success, duration = self.run_demo()
            else:
                self.logger.error(f"Unknown phase: {phase}")
                success, duration = False, 0.0
            
            self.phase_times[phase] = duration
            self.phase_status[phase] = success
            
            if not success:
                self.logger.error(f"Phase '{phase}' failed. Stopping pipeline.")
                self.print_summary()
                return False
            
            # Continue to next phase
            self.logger.info(f"Phase '{phase}' completed successfully")
        
        # Print summary
        self.print_summary()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info("=" * 60)
        
        return True
    
    def print_summary(self):
        """Print pipeline summary."""
        total_time = time.time() - self.start_time if self.start_time else 0.0
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")
        self.logger.info("\nPhase Summary:")
        
        for phase, duration in self.phase_times.items():
            status = "✓ Success" if self.phase_status.get(phase, False) else "✗ Failed"
            self.logger.info(f"  {phase}: {duration/60:.2f} minutes - {status}")
        
        self.logger.info("=" * 60 + "\n")


def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run full pipeline for Llama 3 8B fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python3 run_full_pipeline.py
  
  # Skip data preparation (if already done)
  python3 run_full_pipeline.py --skip-data-prep
  
  # Run only training and evaluation
  python3 run_full_pipeline.py --skip-data-prep --skip-demo
  
  # Resume training from checkpoint
  python3 run_full_pipeline.py --skip-data-prep --resume-training
  
  # Run only demo
  python3 run_full_pipeline.py --skip-data-prep --skip-training --skip-evaluation
        """
    )
    
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation phase'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training phase'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation phase'
    )
    parser.add_argument(
        '--skip-demo',
        action='store_true',
        help='Skip demo phase'
    )
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: pipeline_full.log)'
    )
    
    args = parser.parse_args()
    
    # Determine log file path
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = Path("pipeline_full.log")
    
    # Create pipeline runner
    runner = PipelineRunner(log_file)
    
    try:
        # Run pipeline
        success = runner.run_pipeline(
            skip_data_prep=args.skip_data_prep,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_demo=args.skip_demo,
            resume_training=args.resume_training,
        )
        
        if success:
            print("\n✓ Pipeline completed successfully!")
            return 0
        else:
            print("\n✗ Pipeline failed. Check the log file for details.")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        runner.print_summary()
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

