"""
LoRA Configuration for Llama 3 8B Fine-tuning
Configures Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from peft import LoraConfig

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft is not installed. Install with: pip install peft")

try:
    import config
    LORA_R = config.LORA_R
    LORA_ALPHA = config.LORA_ALPHA
    LORA_DROPOUT = config.LORA_DROPOUT
    LORA_TARGET_MODULES = config.LORA_TARGET_MODULES
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    # Fallback if config.py is not available
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def create_lora_config(
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    target_modules: list = None,
    lora_dropout: float = LORA_DROPOUT,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    inference_mode: bool = False,
    modules_to_save: Optional[list] = None,
) -> Any:
    """
    Create LoRA configuration.
    
    LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
    - Freezes the original model weights
    - Adds trainable low-rank matrices to specific layers
    - Reduces memory usage and training time significantly
    - Maintains model quality with minimal trainable parameters
    
    Args:
        r: LoRA rank (default: 16)
            - Controls the rank of the low-rank matrices
            - Higher r = more trainable parameters, better capacity, more memory
            - Lower r = fewer trainable parameters, less capacity, less memory
            - Typical values: 4, 8, 16, 32, 64
            - Recommended: 16-32 for most tasks
        
        lora_alpha: LoRA alpha scaling parameter (default: 32)
            - Scaling factor for LoRA weights
            - Controls the magnitude of LoRA updates
            - Higher alpha = stronger LoRA influence
            - Typically set to 2*r or equal to r
            - Ratio alpha/r controls the learning rate scaling
            - Recommended: 16-64, often set to 2*r
        
        target_modules: List of module names to apply LoRA to (default: attention + MLP layers)
            - Modules where LoRA adapters will be added
            - For Llama models, typically:
              * q_proj, k_proj, v_proj, o_proj (attention layers)
              * gate_proj, up_proj, down_proj (MLP/feed-forward layers)
            - More modules = more trainable parameters, better capacity
            - Fewer modules = fewer trainable parameters, less capacity
        
        lora_dropout: LoRA dropout rate (default: 0.05)
            - Dropout probability for LoRA layers
            - Helps prevent overfitting
            - Range: 0.0 to 1.0
            - Typical values: 0.05-0.1
            - Higher dropout = more regularization, less overfitting
        
        bias: Bias handling strategy (default: "none")
            - "none": Don't train bias parameters
            - "all": Train all bias parameters
            - "lora_only": Train only bias parameters in LoRA layers
            - Recommended: "none" for most cases
        
        task_type: Task type for PEFT (default: "CAUSAL_LM")
            - "CAUSAL_LM": Causal language modeling (default for Llama)
            - "SEQ_2_SEQ_LM": Sequence-to-sequence language modeling
            - "TOKEN_CLS": Token classification
            - "SEQ_CLS": Sequence classification
        
        inference_mode: Whether to use inference mode (default: False)
            - False: Training mode (default)
            - True: Inference mode (faster, less memory)
        
        modules_to_save: Additional modules to save (default: None)
            - List of module names to save in addition to LoRA adapters
            - Useful for saving embedding layers or other important modules
            - None: Only save LoRA adapters
    
    Returns:
        LoraConfig object
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required. Install with: pip install peft")
    
    if target_modules is None:
        target_modules = LORA_TARGET_MODULES
    
    # Convert task type string to TaskType enum
    if isinstance(task_type, str):
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        task_type = task_type_map.get(task_type.upper(), TaskType.CAUSAL_LM)
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=r,                          # LoRA rank
        lora_alpha=lora_alpha,        # LoRA alpha scaling parameter
        target_modules=target_modules, # Modules to apply LoRA to
        lora_dropout=lora_dropout,    # LoRA dropout rate
        bias=bias,                    # Bias handling strategy
        task_type=task_type,          # Task type
        inference_mode=inference_mode, # Inference mode
        modules_to_save=modules_to_save, # Additional modules to save
    )
    
    return lora_config


def apply_lora_to_model(model, lora_config: Any, prepare_for_kbit_training: bool = True):
    """
    Apply LoRA configuration to model using PEFT.
    
    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration
        prepare_for_kbit_training: Whether to prepare model for k-bit training (default: True)
            - If model is quantized (4-bit/8-bit), this prepares it for training
            - Enables gradient checkpointing and other optimizations
            - Recommended: True for quantized models
    
    Returns:
        Model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required. Install with: pip install peft")
    
    print("\nApplying LoRA to model...")
    
    # Prepare model for k-bit training if needed
    if prepare_for_kbit_training:
        try:
            model = prepare_model_for_kbit_training(model)
            print("  ✓ Model prepared for k-bit training")
        except Exception as e:
            print(f"  ⚠ Could not prepare model for k-bit training: {e}")
            print("  Continuing with standard LoRA application...")
    
    # Apply LoRA using PEFT
    model = get_peft_model(model, lora_config)
    
    print("  ✓ LoRA adapters applied to model")
    
    return model


def print_trainable_parameters(model):
    """
    Print information about trainable parameters.
    
    Args:
        model: Model to analyze
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count non-trainable parameters
    non_trainable_params = total_params - trainable_params
    
    # Calculate percentages
    trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0
    non_trainable_percent = (non_trainable_params / total_params * 100) if total_params > 0 else 0
    
    print(f"\n{'=' * 60}")
    print("Trainable Parameters")
    print(f"{'=' * 60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.4f}%)")
    print(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_percent:.4f}%)")
    print(f"{'=' * 60}\n")
    
    # Print LoRA-specific information if available
    if hasattr(model, 'print_trainable_parameters'):
        print("LoRA-specific information:")
        model.print_trainable_parameters()
        print()


def save_lora_config(lora_config: Any, output_path: Path):
    """
    Save LoRA configuration to JSON file.
    
    Args:
        lora_config: LoRA configuration to save
        output_path: Path to save JSON file
    """
    # Convert LoraConfig to dictionary
    config_dict = {
        'r': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'target_modules': lora_config.target_modules,
        'lora_dropout': lora_config.lora_dropout,
        'bias': lora_config.bias,
        'task_type': str(lora_config.task_type),
        'inference_mode': lora_config.inference_mode,
        'modules_to_save': lora_config.modules_to_save,
    }
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
    
    print(f"✓ LoRA configuration saved to: {output_path}")


def configure_and_apply_lora(
    model,
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    target_modules: list = None,
    lora_dropout: float = LORA_DROPOUT,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    inference_mode: bool = False,
    modules_to_save: Optional[list] = None,
    prepare_for_kbit_training: bool = True,
    save_config: bool = True,
    config_output_path: Optional[Path] = None,
) -> tuple:
    """
    Configure and apply LoRA to model in one step.
    
    Args:
        model: Model to apply LoRA to
        r: LoRA rank (default: from config)
        lora_alpha: LoRA alpha scaling parameter (default: from config)
        target_modules: List of module names to apply LoRA to (default: from config)
        lora_dropout: LoRA dropout rate (default: from config)
        bias: Bias handling strategy (default: "none")
        task_type: Task type for PEFT (default: "CAUSAL_LM")
        inference_mode: Whether to use inference mode (default: False)
        modules_to_save: Additional modules to save (default: None)
        prepare_for_kbit_training: Whether to prepare model for k-bit training (default: True)
        save_config: Whether to save LoRA configuration to JSON (default: True)
        config_output_path: Path to save LoRA configuration (default: OUTPUT_DIR/lora_config.json)
    
    Returns:
        Tuple of (model_with_lora, lora_config)
    """
    print(f"\n{'=' * 60}")
    print("LoRA Configuration")
    print(f"{'=' * 60}")
    
    # Create LoRA configuration
    lora_config = create_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        inference_mode=inference_mode,
        modules_to_save=modules_to_save,
    )
    
    # Print configuration
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Alpha/R ratio: {lora_config.lora_alpha / lora_config.r:.2f}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Bias: {lora_config.bias}")
    print(f"  Task type: {lora_config.task_type}")
    print(f"  Target modules: {lora_config.target_modules}")
    print(f"  Number of target modules: {len(lora_config.target_modules)}")
    if lora_config.modules_to_save:
        print(f"  Modules to save: {lora_config.modules_to_save}")
    print(f"{'=' * 60}\n")
    
    # Apply LoRA to model
    model_with_lora = apply_lora_to_model(model, lora_config, prepare_for_kbit_training)
    
    # Print trainable parameters
    print_trainable_parameters(model_with_lora)
    
    # Save configuration if requested
    if save_config:
        if config_output_path is None:
            config_output_path = Path(OUTPUT_DIR) / "lora_config.json"
        save_lora_config(lora_config, config_output_path)
    
    return model_with_lora, lora_config


def main():
    """Main function to test LoRA configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Configure LoRA for Llama 3 8B fine-tuning"
    )
    parser.add_argument(
        '--r',
        type=int,
        default=LORA_R,
        help=f'LoRA rank (default: {LORA_R})'
    )
    parser.add_argument(
        '--alpha',
        type=int,
        default=LORA_ALPHA,
        help=f'LoRA alpha scaling parameter (default: {LORA_ALPHA})'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=LORA_DROPOUT,
        help=f'LoRA dropout rate (default: {LORA_DROPOUT})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save LoRA configuration JSON (default: models/checkpoints/lora_config.json)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test LoRA configuration (create config without model)'
    )
    
    args = parser.parse_args()
    
    if not PEFT_AVAILABLE:
        print("Error: peft is not installed. Install with: pip install peft")
        return 1
    
    # Create LoRA configuration
    lora_config = create_lora_config(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )
    
    # Print configuration
    print(f"\n{'=' * 60}")
    print("LoRA Configuration")
    print(f"{'=' * 60}")
    print(f"Rank (r): {lora_config.r}")
    print(f"Alpha: {lora_config.lora_alpha}")
    print(f"Alpha/R ratio: {lora_config.lora_alpha / lora_config.r:.2f}")
    print(f"Dropout: {lora_config.lora_dropout}")
    print(f"Bias: {lora_config.bias}")
    print(f"Task type: {lora_config.task_type}")
    print(f"Target modules: {lora_config.target_modules}")
    print(f"Number of target modules: {len(lora_config.target_modules)}")
    print(f"{'=' * 60}\n")
    
    # Save configuration
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(OUTPUT_DIR) / "lora_config.json"
    
    save_lora_config(lora_config, output_path)
    
    print("\n✓ LoRA configuration created successfully!")
    print(f"✓ Configuration saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

