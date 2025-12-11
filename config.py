"""
Configuration file for Llama 3 8B fine-tuning on 5G specifications.
Modify parameters here to adjust training behavior.
"""

import os

# Try to import torch, but don't fail if it's not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy torch object for device detection
    class DummyTorch:
        @staticmethod
        def cuda_is_available():
            return False
    torch = DummyTorch()

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# ============================================================================
# Training Parameters
# ============================================================================
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
MAX_SEQ_LENGTH = 512  # Maximum sequence length for tokenization
WARMUP_STEPS = 100  # Number of warmup steps for learning rate scheduler
SAVE_STEPS = 500  # Save checkpoint every N steps
LOGGING_STEPS = 50  # Log training metrics every N steps

# ============================================================================
# LoRA Configuration
# ============================================================================
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha (scaling parameter)
LORA_DROPOUT = 0.05  # LoRA dropout rate
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Target modules for LoRA

# ============================================================================
# Paths Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECS_DIR = os.path.join(BASE_DIR, "specs")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

# Create directories if they don't exist
os.makedirs(SPECS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
USE_FP16 = TORCH_AVAILABLE and torch.cuda.is_available()  # Use mixed precision training if CUDA is available
USE_8BIT = False  # Use 8-bit quantization (requires bitsandbytes)

# ============================================================================
# Data Configuration
# ============================================================================
TRAIN_SPLIT = 0.9  # Train/validation split ratio
TEST_SPLIT = 0.1
RANDOM_SEED = 42  # Random seed for reproducibility

# ============================================================================
# Inference Configuration
# ============================================================================
MAX_NEW_TOKENS = 512  # Maximum number of tokens to generate
TEMPERATURE = 0.7  # Sampling temperature
TOP_P = 0.9  # Nucleus sampling parameter
TOP_K = 50  # Top-k sampling parameter
DO_SAMPLE = True  # Whether to use sampling

# ============================================================================
# Configuration Class (Optional: for easier access)
# ============================================================================
class Config:
    """Configuration class for easy parameter access."""
    
    # Model
    model_name = MODEL_NAME
    
    # Training
    learning_rate = LEARNING_RATE
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    max_seq_length = MAX_SEQ_LENGTH
    warmup_steps = WARMUP_STEPS
    save_steps = SAVE_STEPS
    logging_steps = LOGGING_STEPS
    
    # LoRA
    lora_r = LORA_R
    lora_alpha = LORA_ALPHA
    lora_dropout = LORA_DROPOUT
    lora_target_modules = LORA_TARGET_MODULES
    
    # Paths
    base_dir = BASE_DIR
    specs_dir = SPECS_DIR
    processed_data_dir = PROCESSED_DATA_DIR
    models_dir = MODELS_DIR
    output_dir = OUTPUT_DIR
    
    # Device
    device = DEVICE
    use_fp16 = USE_FP16
    use_8bit = USE_8BIT
    
    # Data
    train_split = TRAIN_SPLIT
    test_split = TEST_SPLIT
    random_seed = RANDOM_SEED
    
    # Inference
    max_new_tokens = MAX_NEW_TOKENS
    temperature = TEMPERATURE
    top_p = TOP_P
    top_k = TOP_K
    do_sample = DO_SAMPLE
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("Configuration Parameters")
        print("=" * 60)
        print(f"Model Name: {cls.model_name}")
        print(f"\nTraining Parameters:")
        print(f"  Learning Rate: {cls.learning_rate}")
        print(f"  Epochs: {cls.epochs}")
        print(f"  Batch Size: {cls.batch_size}")
        print(f"  Gradient Accumulation Steps: {cls.gradient_accumulation_steps}")
        print(f"  Max Sequence Length: {cls.max_seq_length}")
        print(f"\nLoRA Configuration:")
        print(f"  LoRA R: {cls.lora_r}")
        print(f"  LoRA Alpha: {cls.lora_alpha}")
        print(f"  LoRA Dropout: {cls.lora_dropout}")
        print(f"\nPaths:")
        print(f"  Specs Directory: {cls.specs_dir}")
        print(f"  Processed Data Directory: {cls.processed_data_dir}")
        print(f"  Models Directory: {cls.models_dir}")
        print(f"  Output Directory: {cls.output_dir}")
        print(f"\nDevice Configuration:")
        print(f"  Device: {cls.device}")
        print(f"  Use FP16: {cls.use_fp16}")
        print(f"  Use 8-bit: {cls.use_8bit}")
        print("=" * 60)


# Example usage:
# from config import Config
# model_name = Config.model_name
# learning_rate = Config.learning_rate
# 
# Or use directly:
# from config import MODEL_NAME, LEARNING_RATE, EPOCHS

