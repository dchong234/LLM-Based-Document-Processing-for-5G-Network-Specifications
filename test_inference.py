"""
Inference Testing Script for Fine-tuned Llama 3 8B
Tests the fine-tuned model with sample questions and interactive mode.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch is not installed. Install with: pip install torch")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or peft is not installed. Install with: pip install transformers peft")

try:
    import config
    MODEL_NAME = config.MODEL_NAME
    OUTPUT_DIR = config.OUTPUT_DIR
    MAX_NEW_TOKENS = config.MAX_NEW_TOKENS
    TEMPERATURE = config.TEMPERATURE
    TOP_P = config.TOP_P
    TOP_K = config.TOP_K
    DO_SAMPLE = config.DO_SAMPLE
except ImportError:
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    DO_SAMPLE = True


# Sample 5G test questions
SAMPLE_5G_QUESTIONS = [
    "What is 5G NR?",
    "What are the frequency ranges in 5G?",
    "Explain numerology in 5G",
    "What is the difference between 5G NSA and SA?",
    "What are the key features of 5G networks?",
    "Explain the 5G network architecture",
    "What is beamforming in 5G?",
    "What are the use cases for 5G?",
    "What is network slicing in 5G?",
    "Explain the 5G physical layer",
]


def load_fine_tuned_model(
    model_path: str,
    base_model_name: str = MODEL_NAME,
    use_4bit: bool = True,
    use_8bit: bool = False,
) -> tuple:
    """
    Load fine-tuned model with LoRA adapters.
    
    Args:
        model_path: Path to fine-tuned model directory (contains LoRA adapters)
        base_model_name: Base model name to load (default: from config)
        use_4bit: Whether to use 4-bit quantization (default: True)
        use_8bit: Whether to use 8-bit quantization (default: False)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and peft are required. Install with: pip install transformers peft")
    
    print(f"\n{'=' * 60}")
    print("Loading Fine-tuned Model")
    print(f"{'=' * 60}")
    print(f"Model path: {model_path}")
    print(f"Base model: {base_model_name}")
    print(f"{'=' * 60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad_token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✓ Tokenizer loaded")
    
    # Load base model with quantization if needed
    print("\nLoading base model...")
    from model_loader import load_model_and_tokenizer
    
    base_model, _ = load_model_and_tokenizer(
        model_name=base_model_name,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
    )
    
    print("✓ Base model loaded")
    
    # Load LoRA adapters
    print("\nLoading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✓ LoRA adapters loaded")
        
        # Merge adapters for faster inference (optional)
        # model = model.merge_and_unload()
        # print("✓ LoRA adapters merged")
        
    except Exception as e:
        print(f"⚠ Could not load LoRA adapters: {e}")
        print("  Using base model without adapters")
        model = base_model
    
    # Set to evaluation mode
    model.eval()
    print("✓ Model set to evaluation mode\n")
    
    return model, tokenizer


def format_question(question: str, context: Optional[str] = None) -> str:
    """
    Format question using the instruction template.
    
    Args:
        question: Question to ask
        context: Optional context (default: None)
    
    Returns:
        Formatted prompt string
    """
    if context:
        formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Context:
{context}

### Response:
"""
    else:
        formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Context:
N/A

### Response:
"""
    
    return formatted


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    context: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    do_sample: bool = DO_SAMPLE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate answer to a question using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        question: Question to answer
        context: Optional context (default: None)
        max_new_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        top_k: Top-k sampling parameter (default: 50)
        do_sample: Whether to use sampling (default: True)
        device: Device to run inference on (default: model device)
    
    Returns:
        Dictionary with 'answer', 'inference_time', and 'tokens_generated'
    """
    # Format the question
    prompt = format_question(question, context)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device
    if device is None:
        if TORCH_AVAILABLE:
            device = next(model.parameters()).device
        else:
            device = "cpu"
    
    if TORCH_AVAILABLE:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = inputs
    
    # Generate answer
    start_time = time.time()
    
    if TORCH_AVAILABLE:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    inference_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (everything after "### Response:")
    if "### Response:" in generated_text:
        answer = generated_text.split("### Response:")[-1].strip()
    else:
        # Fallback: extract new tokens only
        input_length = inputs['input_ids'].shape[1]
        answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    # Count tokens generated
    tokens_generated = len(outputs[0]) - inputs['input_ids'].shape[1]
    
    return {
        'answer': answer,
        'inference_time': inference_time,
        'tokens_generated': tokens_generated,
        'full_output': generated_text,
    }


def print_qa_output(question: str, result: Dict[str, Any], show_full: bool = False):
    """
    Print formatted Q&A output.
    
    Args:
        question: Question that was asked
        result: Result dictionary from generate_answer
        show_full: Whether to show full model output (default: False)
    """
    print(f"\n{'=' * 60}")
    print("Question")
    print(f"{'=' * 60}")
    print(question)
    print(f"\n{'=' * 60}")
    print("Answer")
    print(f"{'=' * 60}")
    print(result['answer'])
    print(f"\n{'=' * 60}")
    print("Inference Statistics")
    print(f"{'=' * 60}")
    print(f"Inference time: {result['inference_time']:.3f} seconds")
    print(f"Tokens generated: {result['tokens_generated']}")
    if result['tokens_generated'] > 0:
        print(f"Generation speed: {result['tokens_generated'] / result['inference_time']:.1f} tokens/second")
    print(f"{'=' * 60}\n")
    
    if show_full:
        print(f"{'=' * 60}")
        print("Full Model Output")
        print(f"{'=' * 60}")
        print(result['full_output'])
        print(f"{'=' * 60}\n")


def test_sample_questions(
    model: Any,
    tokenizer: Any,
    questions: List[str] = None,
    **generation_kwargs
):
    """
    Test model with sample questions.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        questions: List of questions to test (default: SAMPLE_5G_QUESTIONS)
        **generation_kwargs: Additional arguments for generate_answer
    """
    if questions is None:
        questions = SAMPLE_5G_QUESTIONS
    
    print(f"\n{'=' * 60}")
    print("Testing with Sample 5G Questions")
    print(f"{'=' * 60}")
    print(f"Number of questions: {len(questions)}")
    print(f"{'=' * 60}\n")
    
    total_time = 0
    total_tokens = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        
        result = generate_answer(model, tokenizer, question, **generation_kwargs)
        print_qa_output(question, result)
        
        total_time += result['inference_time']
        total_tokens += result['tokens_generated']
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"Total questions: {len(questions)}")
    print(f"Total inference time: {total_time:.3f} seconds")
    print(f"Average time per question: {total_time / len(questions):.3f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average tokens per question: {total_tokens / len(questions):.1f}")
    if total_time > 0:
        print(f"Overall generation speed: {total_tokens / total_time:.1f} tokens/second")
    print(f"{'=' * 60}\n")


def interactive_mode(model: Any, tokenizer: Any, **generation_kwargs):
    """
    Interactive mode for testing questions.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        **generation_kwargs: Additional arguments for generate_answer
    """
    print(f"\n{'=' * 60}")
    print("Interactive Testing Mode")
    print(f"{'=' * 60}")
    print("Enter questions to test the model.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to clear the screen.")
    print(f"{'=' * 60}\n")
    
    while True:
        try:
            # Get user input
            question = input("\nQuestion: ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            if question.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            
            # Generate answer
            print("\nGenerating answer...")
            result = generate_answer(model, tokenizer, question, **generation_kwargs)
            print_qa_output(question, result)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run inference testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Llama 3 8B model with inference"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to fine-tuned model directory (default: OUTPUT_DIR/final_model)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=MODEL_NAME,
        help=f'Base model name (default: {MODEL_NAME})'
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
        '--max-tokens',
        type=int,
        default=MAX_NEW_TOKENS,
        help=f'Maximum tokens to generate (default: {MAX_NEW_TOKENS})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=TEMPERATURE,
        help=f'Temperature for sampling (default: {TEMPERATURE})'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=TOP_P,
        help=f'Top-p for sampling (default: {TOP_P})'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K,
        help=f'Top-k for sampling (default: {TOP_K})'
    )
    parser.add_argument(
        '--no-sample',
        action='store_true',
        default=False,
        help='Disable sampling (use greedy decoding)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--sample-questions',
        action='store_true',
        default=False,
        help='Test with sample 5G questions'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Single question to test'
    )
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers and peft are required. Install with: pip install transformers peft")
        return 1
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        print("Please train the model first or specify a valid model path with --model-path")
        return 1
    
    # Determine quantization settings
    use_4bit = args.use_4bit and not args.no_quantization and not args.use_8bit
    use_8bit = args.use_8bit and not args.no_quantization
    
    try:
        # Load model
        model, tokenizer = load_fine_tuned_model(
            model_path=model_path,
            base_model_name=args.base_model,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
        )
        
        # Prepare generation kwargs
        generation_kwargs = {
            'max_new_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'do_sample': not args.no_sample,
        }
        
        # Run tests based on mode
        if args.interactive:
            # Interactive mode
            interactive_mode(model, tokenizer, **generation_kwargs)
        
        elif args.question:
            # Single question mode
            print(f"\nTesting single question...")
            result = generate_answer(model, tokenizer, args.question, **generation_kwargs)
            print_qa_output(args.question, result, show_full=True)
        
        elif args.sample_questions:
            # Sample questions mode
            test_sample_questions(model, tokenizer, **generation_kwargs)
        
        else:
            # Default: test with sample questions
            print("No mode specified. Testing with sample 5G questions...")
            print("Use --interactive for interactive mode or --question for single question.")
            test_sample_questions(model, tokenizer, questions=SAMPLE_5G_QUESTIONS[:3], **generation_kwargs)
        
        print("✓ Inference testing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

