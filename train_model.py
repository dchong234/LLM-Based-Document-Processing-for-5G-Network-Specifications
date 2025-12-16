"""
Training Script for Llama 3 8B Fine-tuning
Orchestrates the complete training pipeline with progress tracking and monitoring.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch is not installed. Install with: pip install torch")

try:
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import TrainerCallback
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrainerCallback = object  # Fallback base class
    print("Warning: transformers is not installed. Install with: pip install transformers")

try:
    from model_loader import load_model_and_tokenizer
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    print("Warning: model_loader.py not found")

try:
    from lora_config import configure_and_apply_lora
    LORA_CONFIG_AVAILABLE = True
except ImportError:
    LORA_CONFIG_AVAILABLE = False
    print("Warning: lora_config.py not found")

try:
    from dataset_formatter import load_and_format_dataset, create_data_collator
    DATASET_FORMATTER_AVAILABLE = True
except ImportError:
    DATASET_FORMATTER_AVAILABLE = False
    print("Warning: dataset_formatter.py not found")

try:
    from training_config import create_training_arguments, print_training_config
    TRAINING_CONFIG_AVAILABLE = True
except ImportError:
    TRAINING_CONFIG_AVAILABLE = False
    print("Warning: training_config.py not found")

try:
    import config
    MODEL_NAME = config.MODEL_NAME
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")


class GPUMemoryCallback(TrainerCallback):
    """Callback to print GPU memory usage periodically."""
    
    def __init__(self, log_interval: int = 50):
        """
        Initialize GPU memory callback.
        
        Args:
            log_interval: Number of steps between memory logging (default: 50)
        """
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log GPU memory usage."""
        if state.global_step % self.log_interval == 0:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
                
                print(f"\n[Step {state.global_step}] GPU Memory Usage:")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Max Allocated: {max_allocated:.2f} GB")
                
                # Reset max memory counter
                torch.cuda.reset_peak_memory_stats(device)


class TrainingProgressCallback(TrainerCallback):
    """Callback to track training progress."""
    
    def __init__(self):
        """Initialize training progress callback."""
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins."""
        self.start_time = time.time()
        print(f"\n{'=' * 60}")
        print("Training Started")
        print(f"{'=' * 60}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total steps: {state.max_steps if state.max_steps else 'N/A'}")
        print(f"Total epochs: {args.num_train_epochs}")
        print(f"{'=' * 60}\n")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step > 0:
            elapsed = time.time() - self.start_time
            avg_time_per_step = elapsed / state.global_step
            
            if state.max_steps:
                remaining_steps = state.max_steps - state.global_step
                estimated_remaining = remaining_steps * avg_time_per_step
                progress = (state.global_step / state.max_steps) * 100
                
                print(f"\n[Step {state.global_step}/{state.max_steps}] Progress: {progress:.1f}%")
                print(f"  Elapsed: {elapsed/60:.1f} minutes")
                print(f"  Estimated remaining: {estimated_remaining/60:.1f} minutes")
                print(f"  Avg time per step: {avg_time_per_step:.2f} seconds")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print(f"Epoch {state.epoch:.1f} Completed")
        print(f"{'=' * 60}")
        print(f"Elapsed time: {elapsed/60:.1f} minutes")
        print(f"Steps completed: {state.global_step}")
        if 'train_loss' in state.log_history[-1]:
            print(f"Training loss: {state.log_history[-1]['train_loss']:.4f}")
        print(f"{'=' * 60}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends."""
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print("Training Completed")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Total steps: {state.global_step}")
        print(f"Total epochs: {state.epoch:.2f}")
        print(f"{'=' * 60}\n")


def print_gpu_memory():
    """Print current GPU memory usage."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {(total - reserved):.2f} GB")
    else:
        print("CUDA not available - using CPU")


def train_model(
    train_file: Path,
    val_file: Optional[Path] = None,
    output_dir: str = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    resume_from_checkpoint: Optional[str] = None,
    use_4bit: bool = True,
    use_8bit: bool = False,
    **training_kwargs
) -> tuple:
    """
    Train Llama 3 8B model with LoRA.
    
    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (optional)
        output_dir: Directory to save checkpoints and final model
        model_name: Model name to load
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
        use_4bit: Whether to use 4-bit quantization (default: True)
        use_8bit: Whether to use 8-bit quantization (default: False)
        **training_kwargs: Additional arguments for create_training_arguments
    
    Returns:
        Tuple of (trainer, model, tokenizer)
    """
    print(f"\n{'=' * 60}")
    print("Llama 3 8B Fine-tuning Training")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Training file: {train_file}")
    if val_file:
        print(f"Validation file: {val_file}")
    print(f"Output directory: {output_dir}")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    print(f"{'=' * 60}\n")
    
    # Check dependencies
    if not MODEL_LOADER_AVAILABLE:
        raise ImportError("model_loader.py is required")
    if not LORA_CONFIG_AVAILABLE:
        raise ImportError("lora_config.py is required")
    if not DATASET_FORMATTER_AVAILABLE:
        raise ImportError("dataset_formatter.py is required")
    if not TRAINING_CONFIG_AVAILABLE:
        raise ImportError("training_config.py is required")
    
    try:
        # Step 1: Load model and tokenizer
        print("Step 1: Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
        )
        print("✓ Model and tokenizer loaded\n")
        
        # Print initial GPU memory
        print_gpu_memory()
        
        # Step 2: Apply LoRA configuration
        print("\nStep 2: Applying LoRA configuration...")
        model, lora_config = configure_and_apply_lora(model)
        print("✓ LoRA configuration applied\n")
        
        # Print GPU memory after LoRA
        print_gpu_memory()
        
        # Step 3: Load and format training data
        print("\nStep 3: Loading and formatting training data...")
        dataset_dict = load_and_format_dataset(
            train_file=train_file,
            val_file=val_file,
            tokenizer=tokenizer,
            model_name=model_name,
        )
        print("✓ Training data loaded and formatted\n")
        
        # Step 4: Create data collator
        print("Step 4: Creating data collator...")
        data_collator = create_data_collator(tokenizer)
        print("✓ Data collator created\n")
        
        # Step 5: Create training arguments
        print("Step 5: Creating training arguments...")
        training_args = create_training_arguments(
            output_dir=output_dir,
            **training_kwargs
        )
        print_training_config(training_args)
        
        # Step 6: Create Trainer
        print("Step 6: Creating Trainer...")
        
        # Create callbacks
        callbacks = [
            TrainingProgressCallback(),
            GPUMemoryCallback(log_interval=50),
        ]
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        print("✓ Trainer created\n")
        
        # Step 7: Start training
        print("Step 7: Starting training...")
        print(f"{'=' * 60}\n")
        
        # Train the model
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        print(f"\n{'=' * 60}")
        print("Training Completed Successfully")
        print(f"{'=' * 60}\n")
        
        # Step 8: Save final model and tokenizer
        print("Step 8: Saving final model and tokenizer...")
        
        # Save the model
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_dir)
        print(f"  ✓ Model saved to: {final_model_dir}")
        
        # Save the tokenizer
        tokenizer.save_pretrained(final_model_dir)
        print(f"  ✓ Tokenizer saved to: {final_model_dir}")
        
        # Save training metrics
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        print(f"  ✓ Training metrics saved to: {metrics_file}")
        
        print("\n✓ Final model and tokenizer saved\n")
        
        # Print final GPU memory
        print_gpu_memory()
        
        # Print training summary
        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")
        print(f"Training loss: {train_result.metrics.get('train_loss', 'N/A')}")
        if 'eval_loss' in train_result.metrics:
            print(f"Validation loss: {train_result.metrics['eval_loss']}")
        print(f"Total training steps: {train_result.metrics.get('train_runtime', 'N/A')}")
        print(f"Final model saved to: {final_model_dir}")
        print(f"{'=' * 60}\n")
        
        return trainer, model, tokenizer
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Checkpoint should be saved. You can resume training with --resume-from-checkpoint")
        raise
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.
    
    Args:
        output_dir: Directory to search for checkpoints
    
    Returns:
        Path to latest checkpoint or None
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Look for checkpoint directories
    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0,
        reverse=True
    )
    
    if checkpoints:
        return str(checkpoints[0])
    
    return None


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Llama 3 8B model with LoRA fine-tuning"
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default=None,
        help='Path to training JSONL file (default: processed_data/training_data.jsonl)'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default=None,
        help='Path to validation JSONL file (default: processed_data/validation_data.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for checkpoints (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help=f'Model name to load (default: {MODEL_NAME})'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (default: auto-detect latest)'
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Automatically resume from latest checkpoint if available'
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
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.train_file:
        train_file = Path(args.train_file)
    else:
        train_file = Path(PROCESSED_DATA_DIR) / "training_data.jsonl"
    
    if args.val_file:
        val_file = Path(args.val_file)
    else:
        val_file = Path(PROCESSED_DATA_DIR) / "validation_data.jsonl"
        if not val_file.exists():
            val_file = None
    
    # Check if training file exists
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return 1
    
    # Determine quantization settings
    use_4bit = args.use_4bit and not args.no_quantization and not args.use_8bit
    use_8bit = args.use_8bit and not args.no_quantization
    
    # Determine checkpoint to resume from
    resume_from_checkpoint = args.resume_from_checkpoint
    if args.auto_resume and resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_from_checkpoint:
            print(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
    
    # Prepare training kwargs
    training_kwargs = {}
    if args.learning_rate is not None:
        training_kwargs['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        training_kwargs['num_train_epochs'] = args.epochs
    if args.batch_size is not None:
        training_kwargs['per_device_train_batch_size'] = args.batch_size
    
    try:
        # Train the model
        trainer, model, tokenizer = train_model(
            train_file=train_file,
            val_file=val_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            resume_from_checkpoint=resume_from_checkpoint,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            **training_kwargs
        )
        
        print("✓ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        print("You can resume training with:")
        latest_checkpoint = find_latest_checkpoint(args.output_dir)
        if latest_checkpoint:
            print(f"  python3 train_model.py --resume-from-checkpoint {latest_checkpoint}")
        else:
            print(f"  python3 train_model.py --auto-resume")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

