# Test Suite

This directory contains unit tests for the Llama 3 8B fine-tuning project.

## Setup

Install pytest:

```bash
pip install pytest
```

Or install all requirements (which includes pytest):

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/

# Or with more verbose output
pytest tests/ -v

# Or with coverage
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_evaluation_metrics.py -v
pytest tests/test_text_cleaner.py -v
pytest tests/test_qa_generator.py -v
pytest tests/test_pdf_extractor.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_evaluation_metrics.py::TestExactMatch -v
pytest tests/test_text_cleaner.py::TestTextCleaner -v
```

### Run Specific Test

```bash
pytest tests/test_evaluation_metrics.py::TestExactMatch::test_exact_match_same -v
```

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest configuration
├── fixtures/                # Sample data for tests
│   ├── sample.txt           # Sample 3GPP-style text
│   ├── sample_cleaned.txt   # Expected cleaned text
│   └── sample_sections.json # Sample section data
├── test_pdf_extractor.py    # Tests for PDF extraction
├── test_text_cleaner.py     # Tests for text cleaning
├── test_qa_generator.py     # Tests for Q&A generation
└── test_evaluation_metrics.py # Tests for evaluation metrics
```

## Fixtures

The `conftest.py` file provides shared fixtures:
- `temp_dir`: Temporary directory for test files
- `fixtures_dir`: Path to fixtures directory
- `sample_text`: Sample 3GPP-style text
- `sample_cleaned_text`: Expected cleaned text
- `sample_section`: Sample section data
- `sample_sections`: List of sample sections

## Writing New Tests

1. Create a new test file following the naming convention: `test_*.py`
2. Import necessary modules and fixtures
3. Create test classes following the convention: `Test*`
4. Create test methods following the convention: `test_*`
5. Use fixtures from `conftest.py` for common data

Example:

```python
import pytest
from my_module import MyClass

class TestMyClass:
    def test_something(self, temp_dir):
        obj = MyClass(temp_dir)
        result = obj.do_something()
        assert result is not None
```

## Notes

- Some tests may be skipped if dependencies are not available (e.g., sentence-transformers)
- Tests use temporary directories that are automatically cleaned up
- Fixtures in `tests/fixtures/` provide sample data for testing

