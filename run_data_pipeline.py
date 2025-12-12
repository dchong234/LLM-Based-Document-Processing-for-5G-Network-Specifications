"""
Data Pipeline Orchestrator for Llama 3 8B Fine-tuning
Orchestrates the entire data preparation pipeline:
1. Extract text from PDFs
2. Clean extracted text
3. Parse sections
4. Generate Q&A pairs
5. Create training datasets
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime
import traceback

# Import pipeline modules
try:
    from pdf_extractor import PDFExtractor
    from text_cleaner import TextCleaner
    from section_parser import SectionParser
    from qa_generator import QAGenerator
    from create_dataset import DatasetCreator
    import config
    
    SPECS_DIR = config.SPECS_DIR
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all pipeline scripts are in the same directory.")
    sys.exit(1)


class PipelineLogger:
    """Logger for pipeline operations."""
    
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
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_step_start(self, step_name: str, step_num: int, total_steps: int):
        """Log step start."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"STEP {step_num}/{total_steps}: {step_name}")
        self.logger.info(f"{'=' * 60}")
    
    def log_step_complete(self, step_name: str, duration: float):
        """Log step completion."""
        self.logger.info(f"✓ {step_name} completed in {duration:.2f} seconds")
    
    def log_step_error(self, step_name: str, error: Exception):
        """Log step error."""
        self.logger.error(f"✗ {step_name} failed: {error}")
        self.logger.error(traceback.format_exc())
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)


class DataPipeline:
    """Orchestrates the entire data preparation pipeline."""
    
    def __init__(self, 
                 specs_dir: Path = None,
                 processed_data_dir: Path = None,
                 log_file: Path = None):
        """
        Initialize data pipeline.
        
        Args:
            specs_dir: Directory containing PDF files
            processed_data_dir: Directory for processed data
            log_file: Path to log file
        """
        self.specs_dir = Path(specs_dir) if specs_dir else Path(SPECS_DIR)
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else Path(PROCESSED_DATA_DIR)
        
        # Set up logging
        if log_file is None:
            log_file = self.processed_data_dir / "pipeline_log.txt"
        self.logger = PipelineLogger(log_file)
        
        # Pipeline steps
        self.steps = [
            ("Extract text from PDFs", self.step1_extract_pdfs),
            ("Clean extracted text", self.step2_clean_text),
            ("Parse sections", self.step3_parse_sections),
            ("Generate Q&A pairs", self.step4_generate_qa),
            ("Create training datasets", self.step5_create_dataset),
        ]
        
        # Pipeline statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0.0,
            'step_durations': {},
            'step_status': {},
            'errors': []
        }
    
    def step1_extract_pdfs(self) -> Tuple[bool, Dict]:
        """
        Step 1: Extract text from PDFs.
        
        Returns:
            Tuple of (success, stats)
        """
        self.logger.log_info("Starting PDF extraction...")
        
        try:
            # Create extractor
            extractor = PDFExtractor(
                input_dir=self.specs_dir,
                output_dir=self.processed_data_dir
            )
            
            # Process all PDFs
            stats = extractor.process_all()
            
            # Check if extraction was successful
            if stats['total_pdfs'] == 0:
                self.logger.log_warning("No PDF files found in specs directory")
                return False, stats
            
            if stats['successful_extractions'] == 0:
                self.logger.log_error("No PDFs were successfully extracted")
                return False, stats
            
            self.logger.log_info(f"Successfully extracted {stats['successful_extractions']}/{stats['total_pdfs']} PDFs")
            return True, stats
            
        except Exception as e:
            self.logger.log_step_error("PDF extraction", e)
            return False, {'error': str(e)}
    
    def step2_clean_text(self) -> Tuple[bool, Dict]:
        """
        Step 2: Clean extracted text.
        
        Returns:
            Tuple of (success, stats)
        """
        self.logger.log_info("Starting text cleaning...")
        
        try:
            # Create cleaner
            cleaner = TextCleaner(
                input_dir=self.processed_data_dir,
                output_dir=self.processed_data_dir
            )
            
            # Process all text files
            stats = cleaner.process_all()
            
            # Check if cleaning was successful
            if stats['total_files'] == 0:
                self.logger.log_warning("No text files found for cleaning")
                return False, stats
            
            if stats['successful_cleanings'] == 0:
                self.logger.log_error("No text files were successfully cleaned")
                return False, stats
            
            self.logger.log_info(f"Successfully cleaned {stats['successful_cleanings']}/{stats['total_files']} files")
            return True, stats
            
        except Exception as e:
            self.logger.log_step_error("Text cleaning", e)
            return False, {'error': str(e)}
    
    def step3_parse_sections(self) -> Tuple[bool, Dict]:
        """
        Step 3: Parse sections from cleaned text.
        
        Returns:
            Tuple of (success, stats)
        """
        self.logger.log_info("Starting section parsing...")
        
        try:
            # Create parser
            parser = SectionParser(min_section_length=100)
            
            # Process all cleaned text files
            all_sections = parser.process_all_files(
                input_dir=self.processed_data_dir,
                output_dir=self.processed_data_dir
            )
            
            # Check if parsing was successful
            if not all_sections:
                self.logger.log_warning("No sections were parsed")
                return False, {'sections': {}}
            
            total_sections = sum(len(sections) for sections in all_sections.values())
            if total_sections == 0:
                self.logger.log_error("No sections were extracted from files")
                return False, {'sections': all_sections}
            
            self.logger.log_info(f"Successfully parsed {total_sections} sections from {len(all_sections)} files")
            return True, {'sections': all_sections, 'total_sections': total_sections}
            
        except Exception as e:
            self.logger.log_step_error("Section parsing", e)
            return False, {'error': str(e)}
    
    def step4_generate_qa(self) -> Tuple[bool, Dict]:
        """
        Step 4: Generate Q&A pairs from sections.
        
        Returns:
            Tuple of (success, stats)
        """
        self.logger.log_info("Starting Q&A generation...")
        
        try:
            # Create generator
            generator = QAGenerator(
                max_context_length=500,
                max_response_length=1000
            )
            
            # Process all section JSON files
            all_examples = generator.process_all_files(
                input_dir=self.processed_data_dir,
                output_dir=self.processed_data_dir
            )
            
            # Check if generation was successful
            if not all_examples:
                self.logger.log_warning("No Q&A examples were generated")
                return False, {'examples': []}
            
            total_examples = len(all_examples)
            if total_examples == 0:
                self.logger.log_error("No Q&A examples were created")
                return False, {'examples': all_examples}
            
            self.logger.log_info(f"Successfully generated {total_examples} Q&A examples")
            return True, {'examples': all_examples, 'total_examples': total_examples}
            
        except Exception as e:
            self.logger.log_step_error("Q&A generation", e)
            return False, {'error': str(e)}
    
    def step5_create_dataset(self) -> Tuple[bool, Dict]:
        """
        Step 5: Create training datasets from Q&A pairs.
        
        Returns:
            Tuple of (success, stats)
        """
        self.logger.log_info("Starting dataset creation...")
        
        try:
            # Create dataset creator
            creator = DatasetCreator(
                train_split=0.9,
                random_seed=42
            )
            
            # Create datasets
            success = creator.create_dataset(
                input_dir=self.processed_data_dir,
                output_dir=self.processed_data_dir
            )
            
            if not success:
                self.logger.log_error("Dataset creation failed")
                return False, {'success': False}
            
            self.logger.log_info("Successfully created training datasets")
            return True, {'success': True}
            
        except Exception as e:
            self.logger.log_step_error("Dataset creation", e)
            return False, {'error': str(e)}
    
    def estimate_time_remaining(self, current_step: int, total_steps: int, elapsed_time: float) -> float:
        """
        Estimate time remaining based on elapsed time.
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            elapsed_time: Elapsed time so far
            
        Returns:
            Estimated time remaining in seconds
        """
        if current_step == 0:
            return 0.0
        
        # Estimate based on average time per step
        avg_time_per_step = elapsed_time / current_step
        remaining_steps = total_steps - current_step
        estimated_remaining = avg_time_per_step * remaining_steps
        
        return estimated_remaining
    
    def format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f} hours {minutes:.2f} minutes ({seconds:.2f} seconds)"
    
    def print_progress(self, step_num: int, total_steps: int, step_name: str, 
                      elapsed_time: float, estimated_remaining: float):
        """
        Print progress information.
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
            step_name: Step name
            elapsed_time: Elapsed time
            estimated_remaining: Estimated time remaining
        """
        progress_percent = (step_num / total_steps) * 100
        progress_bar_length = 40
        filled_length = int(progress_bar_length * step_num / total_steps)
        bar = '=' * filled_length + '-' * (progress_bar_length - filled_length)
        
        self.logger.log_info(f"\nProgress: [{bar}] {progress_percent:.1f}% ({step_num}/{total_steps})")
        self.logger.log_info(f"Elapsed time: {self.format_duration(elapsed_time)}")
        if estimated_remaining > 0:
            self.logger.log_info(f"Estimated remaining: {self.format_duration(estimated_remaining)}")
        self.logger.log_info("")
    
    def run_pipeline(self, start_from_step: int = 1, stop_at_step: int = None) -> bool:
        """
        Run the entire data preparation pipeline.
        
        Args:
            start_from_step: Step number to start from (1-indexed)
            stop_at_step: Step number to stop at (1-indexed, None = run all)
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        # Initialize pipeline
        self.stats['start_time'] = time.time()
        self.logger.log_info(f"\n{'=' * 60}")
        self.logger.log_info("DATA PREPARATION PIPELINE")
        self.logger.log_info(f"{'=' * 60}")
        self.logger.log_info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log_info(f"Specs directory: {self.specs_dir}")
        self.logger.log_info(f"Processed data directory: {self.processed_data_dir}")
        self.logger.log_info(f"{'=' * 60}\n")
        
        # Determine steps to run
        total_steps = len(self.steps)
        start_idx = max(0, start_from_step - 1)
        stop_idx = stop_at_step if stop_at_step else total_steps
        stop_idx = min(stop_idx, total_steps)
        
        # Run pipeline steps
        pipeline_success = True
        completed_steps = 0
        
        for step_num, (step_name, step_func) in enumerate(self.steps[start_idx:stop_idx], start=start_from_step):
            step_start_time = time.time()
            
            # Calculate elapsed time
            elapsed_time = step_start_time - self.stats['start_time']
            estimated_remaining = self.estimate_time_remaining(
                completed_steps, 
                total_steps, 
                elapsed_time
            )
            
            # Print progress
            self.print_progress(step_num - 1, total_steps, step_name, elapsed_time, estimated_remaining)
            
            # Log step start
            self.logger.log_step_start(step_name, step_num, total_steps)
            
            # Run step
            try:
                success, step_stats = step_func()
                
                # Calculate duration
                step_duration = time.time() - step_start_time
                self.stats['step_durations'][step_name] = step_duration
                self.stats['step_status'][step_name] = 'success' if success else 'failed'
                completed_steps += 1
                
                # Log step completion
                if success:
                    self.logger.log_step_complete(step_name, step_duration)
                    self.logger.log_info(f"✓ Step {step_num}/{total_steps} completed successfully")
                else:
                    self.logger.log_step_error(step_name, Exception(f"Step failed: {step_stats.get('error', 'Unknown error')}"))
                    pipeline_success = False
                    
                    # Ask if user wants to continue
                    if step_num < total_steps:
                        self.logger.log_warning(f"Step {step_num} failed. Pipeline will continue to next step.")
                
            except Exception as e:
                # Calculate duration
                step_duration = time.time() - step_start_time
                self.stats['step_durations'][step_name] = step_duration
                self.stats['step_status'][step_name] = 'error'
                self.stats['errors'].append({
                    'step': step_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
                # Log error
                self.logger.log_step_error(step_name, e)
                pipeline_success = False
                
                # Ask if user wants to continue
                if step_num < total_steps:
                    self.logger.log_warning(f"Step {step_num} failed with error. Pipeline will continue to next step.")
        
        # Calculate total duration
        self.stats['end_time'] = time.time()
        self.stats['total_duration'] = self.stats['end_time'] - self.stats['start_time']
        
        # Print final progress
        self.print_progress(total_steps, total_steps, "Complete", self.stats['total_duration'], 0.0)
        
        # Print pipeline summary
        self.print_pipeline_summary()
        
        return pipeline_success
    
    def print_pipeline_summary(self):
        """Print pipeline summary."""
        self.logger.log_info(f"\n{'=' * 60}")
        self.logger.log_info("PIPELINE SUMMARY")
        self.logger.log_info(f"{'=' * 60}")
        self.logger.log_info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log_info(f"Total duration: {self.format_duration(self.stats['total_duration'])}")
        self.logger.log_info(f"\nStep durations:")
        
        for step_name, duration in self.stats['step_durations'].items():
            status = self.stats['step_status'].get(step_name, 'unknown')
            status_icon = '✓' if status == 'success' else '✗'
            percentage = (duration / self.stats['total_duration'] * 100) if self.stats['total_duration'] > 0 else 0
            self.logger.log_info(f"  {status_icon} {step_name}: {self.format_duration(duration)} ({percentage:.1f}%)")
        
        if self.stats['errors']:
            self.logger.log_info(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                self.logger.log_error(f"  Step: {error['step']}")
                self.logger.log_error(f"  Error: {error['error']}")
        
        self.logger.log_info(f"{'=' * 60}\n")
        
        # Check if pipeline was successful
        all_success = all(
            status == 'success' 
            for status in self.stats['step_status'].values()
        )
        
        if all_success:
            self.logger.log_info("✓ Pipeline completed successfully!")
            self.logger.log_info(f"✓ Log file saved to: {self.logger.log_file}")
        else:
            self.logger.log_warning("⚠ Pipeline completed with errors. Check log file for details.")
            self.logger.log_info(f"⚠ Log file saved to: {self.logger.log_file}")
    
    def verify_prerequisites(self) -> bool:
        """
        Verify that prerequisites are met.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        self.logger.log_info("Verifying prerequisites...")
        
        # Check if specs directory exists
        if not self.specs_dir.exists():
            self.logger.log_error(f"Specs directory does not exist: {self.specs_dir}")
            return False
        
        # Check if there are PDF files
        pdf_files = list(self.specs_dir.glob("*.pdf")) + list(self.specs_dir.glob("*.PDF"))
        if not pdf_files:
            self.logger.log_warning(f"No PDF files found in {self.specs_dir}")
            self.logger.log_warning("Pipeline will skip PDF extraction step")
        else:
            self.logger.log_info(f"Found {len(pdf_files)} PDF file(s) in specs directory")
        
        # Check if processed data directory exists (create if not)
        if not self.processed_data_dir.exists():
            self.logger.log_info(f"Creating processed data directory: {self.processed_data_dir}")
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.log_info("✓ Prerequisites verified")
        return True


def main():
    """Main function to run the data preparation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the complete data preparation pipeline for Llama 3 8B fine-tuning"
    )
    parser.add_argument(
        '--specs-dir',
        type=str,
        default=SPECS_DIR,
        help=f'Directory containing PDF files (default: {SPECS_DIR})'
    )
    parser.add_argument(
        '--processed-data-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory for processed data (default: {PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: processed_data/pipeline_log.txt)'
    )
    parser.add_argument(
        '--start-from-step',
        type=int,
        default=1,
        help='Step number to start from (1-5, default: 1)'
    )
    parser.add_argument(
        '--stop-at-step',
        type=int,
        default=None,
        help='Step number to stop at (1-5, default: run all steps)'
    )
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip prerequisite verification'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DataPipeline(
        specs_dir=Path(args.specs_dir),
        processed_data_dir=Path(args.processed_data_dir),
        log_file=Path(args.log_file) if args.log_file else None
    )
    
    # Verify prerequisites
    if not args.skip_verification:
        if not pipeline.verify_prerequisites():
            print("✗ Prerequisites not met. Exiting.")
            return 1
    
    # Run pipeline
    success = pipeline.run_pipeline(
        start_from_step=args.start_from_step,
        stop_at_step=args.stop_at_step
    )
    
    # Exit with appropriate code
    if success:
        print("\n✓ Data preparation pipeline completed successfully!")
        return 0
    else:
        print("\n⚠ Data preparation pipeline completed with errors. Check log file for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

