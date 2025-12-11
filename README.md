# Llama 3 8B Fine-tuning for 5G Specifications

This project fine-tunes the Llama 3 8B model on 5G specification documents to create a specialized language model for 5G-related queries and analysis.

## Overview

This project provides a complete pipeline for:
- Extracting and processing 5G specification PDFs
- Generating Q&A training pairs from specifications
- Fine-tuning Llama 3 8B using LoRA (Low-Rank Adaptation)
- Evaluating model performance on benchmark questions
- Interactive demo interface for testing the model

## Project Structure

```
.
├── specs/                      # 5G specification PDFs and documents
├── processed_data/             # Processed and tokenized training data
│   ├── training_data.jsonl     # Training dataset
│   ├── validation_data.jsonl  # Validation dataset
│   └── *.txt, *.json          # Intermediate processing files
├── models/                     # Fine-tuned model checkpoints and outputs
│   └── checkpoints/           # Model checkpoints and final model
├── tests/                      # Unit tests
│   ├── fixtures/              # Test data fixtures
│   └── test_*.py              # Test files
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
│
├── Data Processing
│   ├── pdf_extractor.py        # Extract text from PDFs
│   ├── text_cleaner.py         # Clean extracted text
│   ├── section_parser.py       # Parse sections from text
│   ├── qa_generator.py         # Generate Q&A pairs
│   ├── create_dataset.py       # Create training datasets
│   ├── dataset_formatter.py   # Format datasets for training
│   └── run_data_pipeline.py   # Run full data pipeline
│
├── Model Training
│   ├── model_loader.py         # Load model with quantization
│   ├── lora_config.py          # LoRA configuration
│   ├── training_config.py      # Training arguments configuration
│   └── train_model.py          # Main training script
│
├── Evaluation & Testing
│   ├── benchmark_questions.py  # Benchmark question definitions
│   ├── evaluation_metrics.py  # Evaluation metrics calculation
│   ├── run_benchmark.py        # Run benchmark evaluation
│   ├── test_inference.py       # Test inference on questions
│   ├── visualize_results.py    # Visualize evaluation results
│   └── integration_test.py     # Integration test suite
│
├── Utilities
│   ├── setup_check.py          # Verify environment setup
│   ├── check_status.py         # Check project status
│   └── demo_app.py             # Gradio demo interface
│
├── Pipeline
│   └── run_full_pipeline.py    # Run complete pipeline
│
├── Documentation
│   ├── README.md               # This file
│   ├── SETUP.md                # Detailed setup guide
│   └── USAGE.md                # Comprehensive usage guide
│
└── tests/                      # Test suite
    ├── conftest.py             # Test fixtures
    ├── pytest.ini              # Pytest configuration
    └── test_*.py               # Unit tests
```

## Quick Start

### 1. Setup

See [SETUP.md](SETUP.md) for detailed setup instructions.

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python3 setup_check.py

# Set Hugging Face token (required for Llama 3)
export HF_TOKEN="your_token_here"
```

### 2. Prepare Data

Place 5G specification PDFs in the `specs/` directory, then run:

```bash
# Run full data pipeline
python3 run_data_pipeline.py
```

This will:
- Extract text from PDFs
- Clean and parse sections
- Generate Q&A pairs
- Create training and validation datasets

### 3. Train Model

```bash
# Basic training
python3 train_model.py

# With custom parameters
python3 train_model.py --epochs 5 --learning-rate 1e-4
```

### 4. Evaluate

```bash
# Run benchmark evaluation
python3 run_benchmark.py

# Visualize results
python3 visualize_results.py
```

### 5. Demo

```bash
# Launch interactive demo
python3 demo_app.py
```

### Run Everything

```bash
# Run complete pipeline (data prep → training → evaluation → demo)
python3 run_full_pipeline.py
```

## Features

### Core Features

- **Fine-tuning**: Fine-tune Llama 3 8B using LoRA (Parameter-Efficient Fine-Tuning)
- **Quantization**: Support for 4-bit and 8-bit quantization with bitsandbytes
- **PDF Processing**: Extract and process 5G specification PDFs
- **Q&A Generation**: Automatically generate training Q&A pairs from specifications
- **Benchmark Evaluation**: Comprehensive evaluation on 20 benchmark questions
- **Interactive Demo**: Gradio-based web interface for testing
- **Visualization**: Generate plots and reports from evaluation results

### Advanced Features

- **Status Checking**: Check project status across all phases (`check_status.py`)
- **Integration Testing**: End-to-end integration test suite
- **Unit Tests**: Comprehensive test coverage for core components
- **Flexible Configuration**: Easy configuration via `config.py`
- **Resume Training**: Resume from checkpoints
- **Multiple Evaluation Metrics**: Exact match, semantic similarity, BLEU, F1, keyword matching

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended: 16GB+)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space

### Software

- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.8 or 12.1 (for GPU support)
- **Dependencies**: See `requirements.txt`

See [SETUP.md](SETUP.md) for detailed requirements and installation instructions.

## Documentation

- **[SETUP.md](SETUP.md)**: Detailed setup guide with troubleshooting
- **[USAGE.md](USAGE.md)**: Comprehensive usage guide covering:
  - Data preparation workflows
  - Training configuration and options
  - Benchmark evaluation
  - Visualization tools
  - Demo app usage
  - Full pipeline execution
  - Performance tips and optimization

## Usage

For detailed usage instructions, see [USAGE.md](USAGE.md).

### Common Workflows

**First-time setup:**
```bash
python3 setup_check.py          # Verify setup
python3 run_data_pipeline.py    # Prepare data
python3 train_model.py          # Train model
python3 run_benchmark.py         # Evaluate
python3 demo_app.py             # Test in demo
```

**Check project status:**
```bash
python3 check_status.py
```

**Run integration tests:**
```bash
python3 integration_test.py
```

**Run unit tests:**
```bash
pytest tests/
```

## Configuration

Key configuration parameters are in `config.py`:

- **Model**: `MODEL_NAME` - Base model to fine-tune
- **Training**: `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`
- **LoRA**: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`
- **Paths**: `SPECS_DIR`, `PROCESSED_DATA_DIR`, `OUTPUT_DIR`

See `config.py` for all available options.

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_evaluation_metrics.py

# With verbose output
pytest tests/ -v
```

### Integration Tests

```bash
# Run integration test suite
python3 integration_test.py
```

## Benchmark Questions

The project includes 20 benchmark questions covering:
- **Easy** (5 questions): Basic definitions and terminology
- **Medium** (8 questions): Technical details and concepts
- **Hard** (7 questions): Complex interactions and procedures

Categories include: Terminology, Frequency Bands, Physical Layer, Architecture, Protocols, Modulation, Beamforming, Network Slicing, and more.

View questions:
```bash
python3 benchmark_questions.py --summary
```

## Project Status

Check the status of all project phases:

```bash
python3 check_status.py
```

This checks:
- Environment setup
- Data preparation completion
- Model training status
- Evaluation results
- Demo readiness

## Troubleshooting

Common issues and solutions are documented in [SETUP.md](SETUP.md).

Quick fixes:
- **OOM errors**: Reduce batch size, use 4-bit quantization
- **Import errors**: Check virtual environment, reinstall requirements
- **CUDA not available**: Verify CUDA installation, reinstall PyTorch with CUDA
- **Hugging Face auth**: Set `HF_TOKEN` environment variable

## Contributing

1. Run tests before submitting: `pytest tests/`
2. Follow existing code style
3. Update documentation as needed

## License

_Add your license information here._

## Acknowledgments

- Meta AI for Llama 3
- Hugging Face for Transformers and PEFT
- 3GPP for 5G specifications
