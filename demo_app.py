"""
Gradio Demo App for Fine-tuned Llama 3 8B
Interactive chat interface for testing the fine-tuned model.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Warning: gradio is not installed. Install with: pip install gradio")

try:
    from test_inference import load_fine_tuned_model, generate_answer
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: test_inference.py not found")

try:
    from benchmark_questions import BENCHMARK_QUESTIONS
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    BENCHMARK_QUESTIONS = []

try:
    import config
    OUTPUT_DIR = config.OUTPUT_DIR
    MODEL_NAME = config.MODEL_NAME
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"


# Global model and tokenizer
_model = None
_tokenizer = None


def load_model(model_path: str = None, use_4bit: bool = True):
    """
    Load the fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model (default: OUTPUT_DIR/final_model)
        use_4bit: Whether to use 4-bit quantization (default: True)
    
    Returns:
        Tuple of (model, tokenizer) or (None, None) if error
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    if not INFERENCE_AVAILABLE:
        return None, None
    
    try:
        if model_path is None:
            model_path = os.path.join(OUTPUT_DIR, "final_model")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model path does not exist: {model_path}")
            return None, None
        
        print(f"Loading model from: {model_path}")
        _model, _tokenizer = load_fine_tuned_model(
            model_path=model_path,
            base_model_name=MODEL_NAME,
            use_4bit=use_4bit,
            use_8bit=False,
        )
        print("✓ Model loaded successfully")
        return _model, _tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def chat_with_model(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Generate response from the model.
    
    Args:
        message: User's question
        history: Chat history
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Tuple of (updated history, inference info)
    """
    global _model, _tokenizer
    
    if not message or not message.strip():
        return history, ""
    
    # Load model if not already loaded
    if _model is None or _tokenizer is None:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return history, "❌ Error: Model not loaded. Please check the model path."
        _model, _tokenizer = model, tokenizer
    
    try:
        # Generate answer
        start_time = time.time()
        result = generate_answer(
            _model,
            _tokenizer,
            message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )
        inference_time = time.time() - start_time
        
        answer = result['answer']
        tokens_generated = result['tokens_generated']
        
        # Update history
        history.append((message, answer))
        
        # Create inference info
        info = f"⏱️ Inference time: {inference_time:.2f}s | 📊 Tokens: {tokens_generated}"
        
        return history, info
    
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        return history, error_msg


def clear_chat() -> Tuple[List, str]:
    """
    Clear chat history.
    
    Returns:
        Empty history and empty info
    """
    return [], ""


def get_example_questions() -> List[str]:
    """
    Get example questions from benchmark questions.
    
    Returns:
        List of example questions
    """
    if BENCHMARK_AVAILABLE and BENCHMARK_QUESTIONS:
        # Get a mix of easy, medium, and hard questions
        easy = [q['question'] for q in BENCHMARK_QUESTIONS if q.get('difficulty') == 'Easy'][:2]
        medium = [q['question'] for q in BENCHMARK_QUESTIONS if q.get('difficulty') == 'Medium'][:2]
        hard = [q['question'] for q in BENCHMARK_QUESTIONS if q.get('difficulty') == 'Hard'][:2]
        return easy + medium + hard
    
    # Fallback examples
    return [
        "What is 5G NR?",
        "What are the frequency ranges in 5G?",
        "Explain numerology in 5G",
        "What is HARQ in 5G?",
        "What is beamforming in 5G?",
        "Explain network slicing in 5G",
    ]


def create_demo():
    """
    Create and launch the Gradio demo app.
    """
    if not GRADIO_AVAILABLE:
        print("Error: gradio is required. Install with: pip install gradio")
        return
    
    # Load model
    print("Initializing model...")
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("Warning: Model could not be loaded. The app will attempt to load it when first used.")
    
    # Get example questions
    example_questions = get_example_questions()
    
    # Custom CSS for professional styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    .example-button {
        margin: 5px;
        padding: 10px 15px;
        border-radius: 5px;
        border: 2px solid #667eea;
        background-color: white;
        color: #667eea;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-button:hover {
        background-color: #667eea;
        color: white;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚀 5G Specifications Assistant</h1>
            <p>Fine-tuned Llama 3 8B Model for 5G Technical Questions</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a question about 5G specifications",
                        placeholder="Type your question here...",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                # Inference info
                info_text = gr.Textbox(
                    label="Inference Information",
                    interactive=False,
                    visible=True,
                )
            
            with gr.Column(scale=1):
                # Settings
                gr.Markdown("### ⚙️ Settings")
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness (0 = deterministic, 1 = creative)",
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=256,
                    step=50,
                    label="Max Tokens",
                    info="Maximum length of generated response",
                )
                
                # Example questions
                gr.Markdown("### 📝 Example Questions")
                example_buttons = []
                for i, example in enumerate(example_questions[:6]):  # Show first 6 examples
                    btn = gr.Button(
                        example,
                        variant="outline",
                        size="sm",
                    )
                    example_buttons.append((btn, example))
                
                # About section
                gr.Markdown("### ℹ️ About")
                gr.Markdown("""
                <div class="info-box">
                    <p><strong>Model:</strong> Llama 3 8B (Fine-tuned)</p>
                    <p><strong>Domain:</strong> 5G Specifications</p>
                    <p><strong>Training Data:</strong> 3GPP Technical Specifications</p>
                    <p><strong>Fine-tuning:</strong> LoRA (Low-Rank Adaptation)</p>
                </div>
                
                <div class="info-box">
                    <p><strong>How to use:</strong></p>
                    <ul>
                        <li>Type your question in the text box</li>
                        <li>Click "Send" or press Enter</li>
                        <li>Adjust temperature and max tokens as needed</li>
                        <li>Click example questions to try them</li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <p><strong>Tips:</strong></p>
                    <ul>
                        <li>Lower temperature (0.3-0.5) for more focused answers</li>
                        <li>Higher temperature (0.7-0.9) for more creative responses</li>
                        <li>Increase max tokens for longer explanations</li>
                    </ul>
                </div>
                """)
        
        # Event handlers
        def submit_message(message, history, temp, max_tok):
            if not message.strip():
                return history, "", ""
            new_history, info = chat_with_model(message, history, temp, max_tok)
            return new_history, "", info
        
        def example_click(example_text, history, temp, max_tok):
            new_history, info = chat_with_model(example_text, history, temp, max_tok)
            return new_history, info
        
        # Submit on button click or Enter key
        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
            outputs=[chatbot, msg, info_text],
        )
        
        msg.submit(
            submit_message,
            inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
            outputs=[chatbot, msg, info_text],
        )
        
        # Clear button
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, info_text],
        )
        
        # Example question buttons
        for btn, example_text in example_buttons:
            def make_click_handler(example):
                def handler(history, temp, max_tok):
                    return example_click(example, history, temp, max_tok)
                return handler
            
            btn.click(
                make_click_handler(example_text),
                inputs=[chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, info_text],
            )
    
    # Launch the app
    print("\n" + "=" * 60)
    print("Starting Gradio Demo App")
    print("=" * 60)
    print("Access the app at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


def main():
    """Main function to run the demo app."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo app for fine-tuned model"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to fine-tuned model directory (default: OUTPUT_DIR/final_model)'
    )
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        default=True,
        help='Use 4-bit quantization (default: True)'
    )
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        default=False,
        help='Disable quantization (default: False)'
    )
    
    args = parser.parse_args()
    
    if not GRADIO_AVAILABLE:
        print("Error: gradio is required. Install with: pip install gradio")
        return 1
    
    # Set model path if provided
    if args.model_path:
        global OUTPUT_DIR
        OUTPUT_DIR = args.model_path
    
    # Set quantization
    use_4bit = args.use_4bit and not args.no_quantization
    
    try:
        # Pre-load model if path is provided
        if args.model_path:
            print(f"Loading model from: {args.model_path}")
            load_model(args.model_path, use_4bit)
        
        # Create and launch demo
        create_demo()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nDemo app stopped by user.")
        return 0
    
    except Exception as e:
        print(f"\n✗ Error launching demo app: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

