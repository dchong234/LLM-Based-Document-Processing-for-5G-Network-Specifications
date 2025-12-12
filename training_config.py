"""
Training Configuration for Llama 3 8B Fine-tuning
Sets up TrainingArguments with optimized settings for fine-tuning.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainingArguments

try:
    from transformers import (
        TrainingArguments,
        Trainer,
        TrainerCallback,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrainingArguments = Any  # Fallback type
    print("Warning: transformers is not installed. Install with: pip install transformers")

try:
    from transformers.optimization import get_cosine_schedule_with_warmup
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes is not installed. Install with: pip install bitsandbytes")

try:
    import config
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    # Use user-specified defaults (override config if needed)
    GRADIENT_ACCUMULATION_STEPS = 8  # User requirement: 4 steps
    WARMUP_STEPS = config.WARMUP_STEPS
    SAVE_STEPS = 500  # User requirement: every 100 steps
    LOGGING_STEPS = 10  # User requirement: every 10 steps
    OUTPUT_DIR = config.OUTPUT_DIR
    MAX_SEQ_LENGTH = config.MAX_SEQ_LENGTH
except ImportError:
    # Fallback if config.py is not available
    LEARNING_RATE = 2e-4
    EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 100
    SAVE_STEPS = 100
    LOGGING_STEPS = 10
    MAX_SEQ_LENGTH = 512
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")


def create_training_arguments(
    output_dir: str = OUTPUT_DIR,
    learning_rate: float = LEARNING_RATE,
    num_train_epochs: int = EPOCHS,
    per_device_train_batch_size: int = BATCH_SIZE,
    per_device_eval_batch_size: Optional[int] = None,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    warmup_steps: int = WARMUP_STEPS,
    save_steps: int = SAVE_STEPS,
    eval_steps: int = 50,
    logging_steps: int = LOGGING_STEPS,
    max_grad_norm: float = 0.3,
    fp16: bool = True,
    bf16: bool = False,
    evaluation_strategy: str = "steps",
    save_strategy: str = "steps",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    save_total_limit: Optional[int] = 3,
    logging_dir: Optional[str] = None,
    report_to: Optional[list] = None,
    remove_unused_columns: bool = False,
    dataloader_pin_memory: bool = True,
    dataloader_num_workers: int = 0,
    group_by_length: bool = False,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    optim: str = "paged_adamw_8bit",
    max_steps: int = 0,
    seed: int = 42,
    data_seed: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create TrainingArguments with optimized settings for fine-tuning.
    
    Args:
        output_dir: Directory to save checkpoints and logs (default: from config)
            - Checkpoints will be saved here
            - Logs will be saved to {output_dir}/logs if logging_dir is None
        
        learning_rate: Learning rate for training (default: 2e-4)
            - Initial learning rate
            - Will be adjusted by learning rate scheduler
            - Typical values: 1e-5 to 5e-4
            - Lower for fine-tuning, higher for training from scratch
        
        num_train_epochs: Number of training epochs (default: 3)
            - Number of complete passes through the training dataset
            - More epochs = more training, risk of overfitting
            - Typical values: 1-5 for fine-tuning
        
        per_device_train_batch_size: Batch size per device for training (default: 4)
            - Number of examples processed per device per step
            - Smaller = less memory, more gradient updates
            - Larger = more memory, fewer gradient updates
            - Effective batch size = batch_size * gradient_accumulation_steps * num_devices
        
        per_device_eval_batch_size: Batch size per device for evaluation (default: same as train)
            - Number of examples processed per device during evaluation
            - Can be larger than train batch size (no gradients computed)
        
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 4)
            - Accumulates gradients over multiple steps before updating weights
            - Effective batch size = batch_size * gradient_accumulation_steps
            - Useful for simulating larger batch sizes with limited memory
            - Typical values: 1-8
        
        warmup_steps: Number of warmup steps for learning rate scheduler (default: 100)
            - Learning rate linearly increases from 0 to learning_rate over warmup_steps
            - Helps stabilize training at the beginning
            - Typical values: 50-500
        
        save_steps: Number of steps between checkpoint saves (default: 100)
            - Saves model checkpoint every N steps
            - More frequent = more checkpoints, more disk space
            - Less frequent = fewer checkpoints, less disk space
        
        eval_steps: Number of steps between evaluations (default: 50)
            - Evaluates model on validation set every N steps
            - More frequent = better monitoring, slower training
            - Less frequent = less monitoring, faster training
        
        logging_steps: Number of steps between logging (default: 10)
            - Logs training metrics every N steps
            - More frequent = better visibility, more logs
            - Less frequent = less visibility, fewer logs
        
        max_grad_norm: Maximum gradient norm for clipping (default: 0.3)
            - Clips gradients to prevent exploding gradients
            - Gradients are scaled if their norm exceeds this value
            - Typical values: 0.1-1.0
            - Lower = more aggressive clipping, more stable training
        
        fp16: Whether to use mixed precision training with FP16 (default: True)
            - Uses 16-bit floating point for faster training and less memory
            - Requires CUDA and compatible GPU
            - Reduces memory usage by ~50%
            - May have slight numerical differences
        
        bf16: Whether to use mixed precision training with BF16 (default: False)
            - Uses bfloat16 for faster training and less memory
            - Requires CUDA with Ampere architecture or newer
            - Better numerical stability than FP16
            - Use either fp16 or bf16, not both
        
        evaluation_strategy: When to evaluate (default: "steps")
            - "no": Don't evaluate
            - "steps": Evaluate every eval_steps
            - "epoch": Evaluate at end of each epoch
        
        save_strategy: When to save checkpoints (default: "steps")
            - "no": Don't save
            - "steps": Save every save_steps
            - "epoch": Save at end of each epoch
        
        load_best_model_at_end: Whether to load best model at end (default: True)
            - Loads the model with best validation metric at end of training
            - Requires evaluation_strategy != "no"
            - Useful for getting the best model, not just the last checkpoint
        
        metric_for_best_model: Metric to use for best model selection (default: "loss")
            - "loss": Use validation loss (lower is better)
            - "eval_loss": Same as "loss"
            - Other metrics can be used if computed
        
        greater_is_better: Whether higher metric values are better (default: False)
            - False for loss (lower is better)
            - True for accuracy, F1, etc. (higher is better)
        
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
            - Keeps only the N most recent checkpoints
            - Deletes older checkpoints to save disk space
            - None = keep all checkpoints
        
        logging_dir: Directory for logs (default: None, uses {output_dir}/logs)
            - TensorBoard logs are saved here
            - Can be viewed with: tensorboard --logdir {logging_dir}
        
        report_to: List of integrations to report to (default: None)
            - ["tensorboard"]: Report to TensorBoard
            - ["wandb"]: Report to Weights & Biases
            - None: No external reporting
        
        remove_unused_columns: Whether to remove unused columns (default: False)
            - Removes columns from dataset that aren't used by model
            - Can save memory, but may cause issues with some datasets
        
        dataloader_pin_memory: Whether to pin memory in dataloader (default: True)
            - Pins memory for faster GPU transfer
            - Only useful with CUDA
            - May use more memory
        
        dataloader_num_workers: Number of dataloader workers (default: 0)
            - Number of subprocesses for data loading
            - 0 = main process only
            - Higher = more parallel loading, more CPU usage
        
        group_by_length: Whether to group sequences by length (default: False)
            - Groups sequences of similar length together
            - Reduces padding, improves efficiency
            - May slow down data loading
        
        lr_scheduler_type: Learning rate scheduler type (default: "cosine")
            - "linear": Linear decay
            - "cosine": Cosine decay (smooth, recommended)
            - "cosine_with_restarts": Cosine with restarts
            - "polynomial": Polynomial decay
            - "constant": Constant learning rate
            - "constant_with_warmup": Constant with warmup
        
        weight_decay: Weight decay for regularization (default: 0.01)
            - L2 regularization coefficient
            - Helps prevent overfitting
            - Typical values: 0.0-0.1
        
        optim: Optimizer to use (default: "paged_adamw_8bit")
            - "adamw_hf": Standard AdamW
            - "adamw_torch": PyTorch AdamW
            - "adamw_8bit": 8-bit AdamW (memory efficient)
            - "paged_adamw_8bit": Paged 8-bit AdamW (recommended for large models)
            - "adafactor": Adafactor optimizer
            - "sgd": SGD optimizer
        
        max_steps: Maximum number of training steps (default: 0)
            - If set, overrides num_train_epochs
            - Useful for fine-tuning with limited steps
            - 0 = use num_train_epochs
        
        seed: Random seed for reproducibility (default: 42)
            - Sets random seed for Python, NumPy, PyTorch
            - Same seed = reproducible results
        
        data_seed: Random seed for data shuffling (default: None, uses seed)
            - Separate seed for data operations
            - None = use seed value
        
        **kwargs: Additional arguments to pass to TrainingArguments
    
    Returns:
        TrainingArguments object
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. Install with: pip install transformers")
    
    # Set default eval batch size if not provided
    if per_device_eval_batch_size is None:
        per_device_eval_batch_size = per_device_train_batch_size
    
    # Set default logging directory if not provided
    if logging_dir is None:
        logging_dir = os.path.join(output_dir, "logs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    # Create TrainingArguments
    training_args = TrainingArguments(
        # Output and logging
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        report_to=report_to,
        
        # Training parameters
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Learning rate and optimization
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim=optim,
        max_grad_norm=max_grad_norm,
        
        # Mixed precision
        fp16=fp16,
        bf16=bf16,
        
        # Evaluation and saving
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        
        # Data loading
        remove_unused_columns=remove_unused_columns,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        group_by_length=group_by_length,
        
        # Reproducibility
        seed=seed,
        data_seed=data_seed,
        
        # Additional arguments
        **kwargs
    )
    
    return training_args


def print_training_config(training_args: Any):
    """
    Print training configuration in a readable format.
    
    Args:
        training_args: TrainingArguments object
    """
    print(f"\n{'=' * 60}")
    print("Training Configuration")
    print(f"{'=' * 60}")
    
    print(f"\nOutput and Logging:")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Logging directory: {training_args.logging_dir}")
    print(f"  Logging steps: {training_args.logging_steps}")
    print(f"  Report to: {training_args.report_to}")
    
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    if training_args.max_steps:
        print(f"  Max steps: {training_args.max_steps}")
    print(f"  Train batch size: {training_args.per_device_train_batch_size}")
    print(f"  Eval batch size: {training_args.per_device_eval_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    effective_batch_size = (
        training_args.per_device_train_batch_size * 
        training_args.gradient_accumulation_steps
    )
    print(f"  Effective batch size: {effective_batch_size}")
    
    print(f"\nLearning Rate and Optimization:")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  LR scheduler: {training_args.lr_scheduler_type}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Max gradient norm: {training_args.max_grad_norm}")
    
    print(f"\nMixed Precision:")
    print(f"  FP16: {training_args.fp16}")
    print(f"  BF16: {training_args.bf16}")
    
    print(f"\nEvaluation and Saving:")
    print(f"  Evaluation strategy: {training_args.evaluation_strategy}")
    if training_args.evaluation_strategy != "no":
        print(f"  Eval steps: {training_args.eval_steps}")
    print(f"  Save strategy: {training_args.save_strategy}")
    print(f"  Save steps: {training_args.save_steps}")
    print(f"  Save total limit: {training_args.save_total_limit}")
    print(f"  Load best model at end: {training_args.load_best_model_at_end}")
    if training_args.load_best_model_at_end:
        print(f"  Metric for best model: {training_args.metric_for_best_model}")
        print(f"  Greater is better: {training_args.greater_is_better}")
    
    print(f"\nData Loading:")
    print(f"  Remove unused columns: {training_args.remove_unused_columns}")
    print(f"  Pin memory: {training_args.dataloader_pin_memory}")
    print(f"  Num workers: {training_args.dataloader_num_workers}")
    print(f"  Group by length: {training_args.group_by_length}")
    
    print(f"\nReproducibility:")
    print(f"  Seed: {training_args.seed}")
    print(f"  Data seed: {training_args.data_seed}")
    
    print(f"\n{'=' * 60}\n")


def main():
    """Main function to test training configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create training configuration for Llama 3 8B fine-tuning"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for checkpoints (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of epochs (default: {EPOCHS})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f'Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=WARMUP_STEPS,
        help=f'Warmup steps (default: {WARMUP_STEPS})'
    )
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers is not installed. Install with: pip install transformers")
        return 1
    
    try:
        # Create training arguments
        training_args = create_training_arguments(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
        )
        
        # Print configuration
        print_training_config(training_args)
        
        print("✓ Training configuration created successfully!")
        print(f"✓ Configuration saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error creating training configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

