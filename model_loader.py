"""
Model Loader for Llama 3 8B Fine-tuning
Loads Llama 3 8B with 4-bit quantization using BitsAndBytesConfig.
"""

import os
import sys
from typing import Tuple, Optional, Any

# Try to import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch is not installed. Install with: pip install torch")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers is not installed. Install with: pip install transformers")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes is not installed. Install with: pip install bitsandbytes")

try:
    import config
    MODEL_NAME = config.MODEL_NAME
    DEVICE = config.DEVICE
except ImportError:
    # Fallback if config.py is not available
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    if TORCH_AVAILABLE:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = "cpu"


def get_model_size_mb(model) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_model_memory_usage(model, device: str = None) -> dict:
    """
    Get model memory usage information.
    
    Args:
        model: PyTorch model
        device: Device to check memory on (default: model device)
        
    Returns:
        Dictionary with memory usage information
    """
    if not TORCH_AVAILABLE:
        return {'device': 'unknown', 'model_size_mb': 0.0}
    
    if device is None:
        device = next(model.parameters()).device
    
    memory_info = {
        'device': str(device),
        'model_size_mb': get_model_size_mb(model),
    }
    
    # Get CUDA memory usage if available
    if device.type == 'cuda' and torch.cuda.is_available():
        memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated(device) / 1024**2
        memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved(device) / 1024**2
        memory_info['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 1024**2
        memory_info['cuda_max_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 1024**2
        
        # Get total GPU memory
        memory_info['cuda_total_mb'] = torch.cuda.get_device_properties(device).total_memory / 1024**2
    
    return memory_info


def print_memory_usage(memory_info: dict):
    """
    Print memory usage information.
    
    Args:
        memory_info: Dictionary with memory usage information
    """
    print(f"\n{'=' * 60}")
    print("Model Memory Usage")
    print(f"{'=' * 60}")
    print(f"Device: {memory_info['device']}")
    print(f"Model size: {memory_info['model_size_mb']:.2f} MB")
    
    if 'cuda_allocated_mb' in memory_info:
        print(f"\nCUDA Memory:")
        print(f"  Allocated: {memory_info['cuda_allocated_mb']:.2f} MB")
        print(f"  Reserved: {memory_info['cuda_reserved_mb']:.2f} MB")
        print(f"  Max allocated: {memory_info['cuda_max_allocated_mb']:.2f} MB")
        print(f"  Max reserved: {memory_info['cuda_max_reserved_mb']:.2f} MB")
        
        if 'cuda_total_mb' in memory_info:
            print(f"  Total GPU memory: {memory_info['cuda_total_mb']:.2f} MB")
            used_percent = (memory_info['cuda_reserved_mb'] / memory_info['cuda_total_mb']) * 100
            print(f"  GPU memory used: {used_percent:.2f}%")
    
    print(f"{'=' * 60}\n")


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    use_4bit: bool = True,
    use_8bit: bool = False,
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[Any] = None,
    trust_remote_code: bool = False,
    use_auth_token: Optional[str] = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: Optional[Any] = None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_8bit_compute_dtype: Optional[Any] = None,
    bnb_8bit_quant_type: str = "nf8",
) -> Tuple:
    """
    Load Llama 3 8B model with quantization and tokenizer.
    
    Args:
        model_name: Name of the model to load (default: meta-llama/Meta-Llama-3-8B)
        use_4bit: Whether to use 4-bit quantization (default: True)
        use_8bit: Whether to use 8-bit quantization (default: False)
        device_map: Device mapping strategy (default: "auto")
        torch_dtype: Torch data type (default: None, auto-detect)
        trust_remote_code: Whether to trust remote code (default: False)
        use_auth_token: Hugging Face auth token (default: None, uses environment variable)
        load_in_4bit: Whether to load model in 4-bit (default: True)
        load_in_8bit: Whether to load model in 8-bit (default: False)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization (default: None, uses float16)
        bnb_4bit_quant_type: Quantization type for 4-bit (default: "nf4")
        bnb_4bit_use_double_quant: Whether to use double quantization for 4-bit (default: True)
        bnb_8bit_compute_dtype: Compute dtype for 8-bit quantization (default: None, uses float16)
        bnb_8bit_quant_type: Quantization type for 8-bit (default: "nf8")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if required libraries are available
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. Install with: pip install transformers")
    
    if use_4bit and not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for 4-bit quantization. Install with: pip install bitsandbytes")
    
    if use_8bit and not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for 8-bit quantization. Install with: pip install bitsandbytes")
    
    print(f"\n{'=' * 60}")
    print("Loading Model and Tokenizer")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Use 4-bit quantization: {use_4bit}")
    print(f"Use 8-bit quantization: {use_8bit}")
    print(f"{'=' * 60}\n")
    
    # Get auth token from environment if not provided
    if use_auth_token is None:
        use_auth_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    # Configure quantization
    quantization_config = None
    
    if use_4bit:
        # 4-bit quantization configuration using BitsAndBytesConfig
        # This reduces model memory usage from ~16GB (FP16) to ~4-6GB (4-bit)
        # 
        # bnb_4bit_compute_dtype: Data type for computations
        #   - float16: Faster, less precise (default)
        #   - float32: Slower, more precise
        #   - bfloat16: Balanced (requires CUDA with Ampere architecture or newer)
        #
        # bnb_4bit_quant_type: Quantization type
        #   - "nf4": NormalFloat4 (recommended for 4-bit, better quality)
        #   - "fp4": FP4 (alternative, slightly faster but lower quality)
        #
        # bnb_4bit_use_double_quant: Whether to use double quantization
        #   - True: Uses nested quantization (quantizes quantization constants)
        #   - Reduces memory usage further by ~0.4GB with minimal quality loss
        #   - Recommended for memory-constrained environments
        #
        # load_in_4bit: Whether to load model in 4-bit format
        #   - True: Loads model directly in 4-bit (recommended)
        #   - False: Loads in full precision then quantizes (not recommended)
        if bnb_4bit_compute_dtype is None:
            if TORCH_AVAILABLE:
                bnb_4bit_compute_dtype = torch.float16
            else:
                raise ImportError("torch is required for quantization. Install with: pip install torch")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        
        print("4-bit quantization configuration:")
        print(f"  Compute dtype: {bnb_4bit_compute_dtype}")
        print(f"  Quantization type: {bnb_4bit_quant_type}")
        print(f"  Double quantization: {bnb_4bit_use_double_quant}")
        print(f"  Load in 4-bit: {load_in_4bit}")
        
    elif use_8bit:
        # 8-bit quantization configuration using BitsAndBytesConfig
        # This reduces model memory usage from ~16GB (FP16) to ~8-10GB (8-bit)
        # 
        # bnb_8bit_compute_dtype: Data type for computations
        #   - float16: Faster, less precise (default)
        #   - float32: Slower, more precise
        #
        # bnb_8bit_quant_type: Quantization type
        #   - "nf8": NormalFloat8 (recommended for 8-bit)
        #   - "fp8": FP8 (alternative)
        #
        # load_in_8bit: Whether to load model in 8-bit format
        #   - True: Loads model directly in 8-bit (recommended)
        #   - False: Loads in full precision then quantizes (not recommended)
        if bnb_8bit_compute_dtype is None:
            if TORCH_AVAILABLE:
                bnb_8bit_compute_dtype = torch.float16
            else:
                raise ImportError("torch is required for quantization. Install with: pip install torch")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            bnb_8bit_compute_dtype=bnb_8bit_compute_dtype,
            bnb_8bit_quant_type=bnb_8bit_quant_type,
        )
        
        print("8-bit quantization configuration:")
        print(f"  Compute dtype: {bnb_8bit_compute_dtype}")
        print(f"  Quantization type: {bnb_8bit_quant_type}")
        print(f"  Load in 8-bit: {load_in_8bit}")
    
    # Set torch dtype if not using quantization
    if quantization_config is None:
        if torch_dtype is None:
            # Use float16 for CUDA, float32 for CPU
            if TORCH_AVAILABLE:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                raise ImportError("torch is required. Install with: pip install torch")
        print(f"Using dtype: {torch_dtype}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=use_auth_token,
    )
    
    # Set pad_token if not present
    # Llama models don't have a pad_token by default, so we need to set it
    # Using eos_token as pad_token is a common practice for causal language models
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token...")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to right for causal language models
    # This ensures padding tokens are on the right side of sequences
    tokenizer.padding_side = "right"
    
    print(f"✓ Tokenizer loaded")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    print(f"  EOS token: {tokenizer.eos_token}")
    print(f"  BOS token: {tokenizer.bos_token}")
    
    # Load model
    print("\nLoading model...")
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": use_auth_token,
    }
    
    # Add quantization config if using quantization
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        # Use torch_dtype if not using quantization
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        **model_kwargs,
    )
    
    print(f"✓ Model loaded")
    
    # Prepare model for training
    print("\nPreparing model for training...")
    
    # Enable gradient checkpointing to reduce memory usage during training
    # This trades computation time for memory by recomputing activations
    # instead of storing them during the forward pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled")
    
    # Disable cache for training (saves memory)
    # This prevents storing past key-value states during generation
    if hasattr(model, "config"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            print("  ✓ Disabled use_cache for training")
    
    # Set model to training mode
    model.train()
    print("  ✓ Model set to training mode")
    
    # Print model information
    print(f"\nModel information:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Calculate and print memory usage
    memory_info = get_model_memory_usage(model)
    print_memory_usage(memory_info)
    
    return model, tokenizer


def prepare_model_for_lora_training(model):
    """
    Prepare model for LoRA (Low-Rank Adaptation) training.
    
    This function prepares the model for parameter-efficient fine-tuning using LoRA.
    LoRA freezes the original model weights and adds trainable low-rank matrices
    to specific layers, significantly reducing memory usage and training time.
    
    Args:
        model: Model to prepare for LoRA training
        
    Returns:
        Model prepared for LoRA training
    """
    print("\nPreparing model for LoRA training...")
    
    # Freeze all parameters (LoRA will add trainable parameters)
    for param in model.parameters():
        param.requires_grad = False
    
    print("  ✓ Frozen all model parameters")
    print("  ✓ Model ready for LoRA training (trainable parameters will be added by PEFT)")
    
    return model


def main():
    """Main function to test model loading."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load Llama 3 8B model with quantization"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help=f'Model name to load (default: {MODEL_NAME})'
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
        '--device-map',
        type=str,
        default='auto',
        help='Device mapping strategy (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Determine quantization settings
    use_4bit = args.use_4bit and not args.no_quantization and not args.use_8bit
    use_8bit = args.use_8bit and not args.no_quantization
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            device_map=args.device_map,
        )
        
        print("\n✓ Model and tokenizer loaded successfully!")
        print("\nModel is ready for fine-tuning.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

