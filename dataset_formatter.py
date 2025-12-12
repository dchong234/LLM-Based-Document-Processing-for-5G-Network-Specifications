"""
Dataset Formatter for Llama 3 8B Fine-tuning
Formats and tokenizes training data for instruction tuning.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict

try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    DatasetDict = Any  # Fallback type
    print("Warning: datasets is not installed. Install with: pip install datasets")

try:
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers is not installed. Install with: pip install transformers")

try:
    import config
    MODEL_NAME = config.MODEL_NAME
    # Use 2048 for max sequence length (user requirement)
    # config.MAX_SEQ_LENGTH is 512, but we use 2048 for tokenization
    MAX_SEQ_LENGTH = 512
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError:
    # Fallback if config.py is not available
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    MAX_SEQ_LENGTH = 512
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")


# Instruction template for formatting examples
INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""


def format_instruction_example(example: Dict[str, Any], template: str = INSTRUCTION_TEMPLATE) -> str:
    """
    Format an example using the instruction template.
    
    Args:
        example: Dictionary with 'instruction', 'context' (or 'input'), and 'response' (or 'output') keys
        template: Template string for formatting (default: INSTRUCTION_TEMPLATE)
    
    Returns:
        Formatted text string
    """
    # Extract fields with fallbacks
    instruction = example.get('instruction', example.get('input', ''))
    context = example.get('context', example.get('input', ''))
    response = example.get('response', example.get('output', ''))
    
    # Format using template
    formatted_text = template.format(
        instruction=instruction,
        context=context if context else "N/A",
        response=response
    )
    
    return formatted_text


def tokenize_function(
    examples: Dict[str, List],
    tokenizer: Any,
    max_length: int = MAX_SEQ_LENGTH,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Dict[str, List]:
    """
    Tokenize examples for training.
    
    Args:
        examples: Dictionary with 'text' key containing list of formatted texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length (default: 2048)
        padding: Whether to pad sequences (default: True)
        truncation: Whether to truncate sequences (default: True)
        return_tensors: Return tensor format (default: None, returns lists)
    
    Returns:
        Dictionary with tokenized inputs
    """
    # Tokenize the formatted text
    tokenized = tokenizer(
        examples['text'],
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    
    # Create labels (copy of input_ids for causal language modeling)
    # Labels are used to compute loss during training
    # For causal LM, we predict the next token, so labels = input_ids
    # tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def load_and_format_dataset(
    train_file: Path,
    val_file: Optional[Path] = None,
    tokenizer: Any = None,
    model_name: str = MODEL_NAME,
    max_length: int = MAX_SEQ_LENGTH,
    template: str = INSTRUCTION_TEMPLATE,
) -> Any:
    """
    Load JSONL files and format them for training.
    
    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (optional)
        tokenizer: Tokenizer to use (if None, loads from model_name)
        model_name: Model name to load tokenizer from (default: from config)
        max_length: Maximum sequence length (default: 2048)
        template: Template string for formatting (default: INSTRUCTION_TEMPLATE)
    
    Returns:
        DatasetDict with 'train' and optionally 'validation' datasets
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets is required. Install with: pip install datasets")
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. Install with: pip install transformers")
    
    # Load tokenizer if not provided
    if tokenizer is None:
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"\n{'=' * 60}")
    print("Loading and Formatting Dataset")
    print(f"{'=' * 60}")
    print(f"Training file: {train_file}")
    if val_file:
        print(f"Validation file: {val_file}")
    print(f"Max sequence length: {max_length}")
    print(f"{'=' * 60}\n")
    
    # Load datasets from JSONL files
    print("Loading JSONL files...")
    datasets = {}
    
    # Load training dataset
    train_dataset = load_dataset('json', data_files=str(train_file), split='train')
    print(f"  ✓ Loaded training dataset: {len(train_dataset)} examples")
    datasets['train'] = train_dataset
    
    # Load validation dataset if provided
    if val_file and val_file.exists():
        val_dataset = load_dataset('json', data_files=str(val_file), split='train')
        print(f"  ✓ Loaded validation dataset: {len(val_dataset)} examples")
        datasets['validation'] = val_dataset
    
    # Format examples using template
    print("\nFormatting examples...")
    
    def format_examples(batch):
        """Format a batch of examples."""
        texts = []
        batch = [{'instruction': ins, 'input': inp, 'output': out} for ins, inp, out in zip(batch['instruction'], batch['input'], batch['output'])]
        for example in batch:
            formatted = format_instruction_example(example, template)
            texts.append(formatted)
        return {'text': texts}
    
    # Apply formatting to datasets
    train_dataset = train_dataset.map(
        format_examples,
        batched=True,
        batch_size=1000,
        remove_columns=[col for col in train_dataset.column_names if col != 'text'],
    )
    print(f"  ✓ Formatted training dataset")
    
    if 'validation' in datasets:
        val_dataset = datasets['validation'].map(
            format_examples,
            batched=True,
            batch_size=1000,
            remove_columns=[col for col in datasets['validation'].column_names if col != 'text'],
        )
        print(f"  ✓ Formatted validation dataset")
        datasets['validation'] = val_dataset
    
    datasets['train'] = train_dataset
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    
    def tokenize_examples(examples):
        """Tokenize a batch of examples."""
        return tokenize_function(
            examples,
            tokenizer,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_examples,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
    )
    print(f"  ✓ Tokenized training dataset")
    
    if 'validation' in datasets:
        val_dataset = datasets['validation'].map(
            tokenize_examples,
            batched=True,
            batch_size=1000,
            remove_columns=['text'],
        )
        print(f"  ✓ Tokenized validation dataset")
        datasets['validation'] = val_dataset
    
    datasets['train'] = train_dataset
    
    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)
    
    # Print statistics
    print(f"\n{'=' * 60}")
    print("Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Training examples: {len(dataset_dict['train']):,}")
    if 'validation' in dataset_dict:
        print(f"Validation examples: {len(dataset_dict['validation']):,}")
    print(f"Features: {list(dataset_dict['train'].features.keys())}")
    print(f"{'=' * 60}\n")
    
    return dataset_dict


def create_data_collator(
    tokenizer: Any,
    mlm: bool = False,
    pad_to_multiple_of: Optional[int] = None,
) -> Any:
    """
    Create a data collator for batching.
    
    Args:
        tokenizer: Tokenizer to use
        mlm: Whether to use masked language modeling (default: False, for causal LM)
        pad_to_multiple_of: Pad sequences to multiple of this value (default: None)
            - Useful for optimizing GPU memory usage
            - Common values: 8, 16, 32
    
    Returns:
        Data collator object
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. Install with: pip install transformers")
    
    # For causal language modeling, use DataCollatorForLanguageModeling
    # with mlm=False (predicts next token, not masked tokens). It clones
    # input_ids to labels after padding so the model receives labels.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,  # False for causal LM
        pad_to_multiple_of=pad_to_multiple_of,
    )
    
    return data_collator


def format_chat_example(example: Dict[str, Any]) -> str:
    """
    Format an example from chat format (messages) to instruction format.
    
    Args:
        example: Dictionary with 'messages' key containing list of message dicts
    
    Returns:
        Formatted text string
    """
    if 'messages' not in example:
        # Fall back to instruction format
        return format_instruction_example(example)
    
    messages = example['messages']
    
    # Extract system, user, and assistant messages
    system_msg = ""
    user_msg = ""
    assistant_msg = ""
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            system_msg = content
        elif role == 'user':
            user_msg = content
        elif role == 'assistant':
            assistant_msg = content
    
    # Format using template
    formatted_text = INSTRUCTION_TEMPLATE.format(
        instruction=user_msg if user_msg else "N/A",
        context=system_msg if system_msg else "N/A",
        response=assistant_msg if assistant_msg else "N/A"
    )
    
    return formatted_text


def main():
    """Main function to test dataset formatting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Format and tokenize training datasets"
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
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help=f'Model name to load tokenizer from (default: {MODEL_NAME})'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f'Maximum sequence length (default: {MAX_SEQ_LENGTH})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save formatted datasets (optional)'
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
    
    # Check if files exist
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return 1
    
    if not val_file.exists():
        print(f"Warning: Validation file not found: {val_file}")
        val_file = None
    
    try:
        # Load and format dataset
        dataset_dict = load_and_format_dataset(
            train_file=train_file,
            val_file=val_file,
            model_name=args.model_name,
            max_length=args.max_length,
        )
        
        # Create data collator
        print("Creating data collator...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        data_collator = create_data_collator(tokenizer)
        print("  ✓ Data collator created")
        
        # Save datasets if output directory is provided
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving datasets to {output_path}...")
            dataset_dict.save_to_disk(str(output_path))
            print(f"  ✓ Datasets saved")
        
        print("\n✓ Dataset formatting completed successfully!")
        print("\nDataset is ready for training.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error formatting dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
