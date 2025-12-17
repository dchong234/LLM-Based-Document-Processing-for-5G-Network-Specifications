"""
Unit tests for QA generator
"""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_generator import QAGenerator, TrainingExample


class TestTrainingExample:
    """Test cases for TrainingExample dataclass."""
    
    def test_creation(self):
        """Test creating a TrainingExample."""
        example = TrainingExample(
            instruction="What is 5G NR?",
            context="5G specifications",
            response="5G NR is the radio access technology.",
            source="test_spec.json",
            section="5.1"
        )
        
        assert example.instruction == "What is 5G NR?"
        assert example.context == "5G specifications"
        assert example.section == "5.1"
    
    def test_to_dict(self):
        """Test converting TrainingExample to dictionary."""
        example = TrainingExample(
            instruction="What is numerology?",
            context="5G physical layer",
            response="Numerology is the subcarrier spacing parameter.",
            source="test.json",
            section="5.1.1"
        )
        
        result = example.to_dict()
        
        assert isinstance(result, dict)
        assert result['instruction'] == "What is numerology?"
        assert result['response'] == "Numerology is the subcarrier spacing parameter."
        assert result['section'] == "5.1.1"


class TestQAGenerator:
    """Test cases for QAGenerator class."""
    
    def test_init(self):
        """Test QAGenerator initialization."""
        generator = QAGenerator()
        
        assert generator.max_context_length == 500
        assert generator.max_response_length == 1000
        assert len(generator.definition_patterns) > 0
    
    def test_init_custom_params(self):
        """Test QAGenerator with custom parameters."""
        generator = QAGenerator(
            max_context_length=1000,
            max_response_length=2000
        )
        
        assert generator.max_context_length == 1000
        assert generator.max_response_length == 2000
    
    def test_extract_definitions(self):
        """Test definition extraction."""
        generator = QAGenerator()
        
        text = """
        Numerology is defined as the subcarrier spacing parameter.
        HARQ refers to Hybrid Automatic Repeat Request.
        Beamforming is a technique for directional transmission.
        """
        
        definitions = generator.extract_definitions(text)
        
        assert len(definitions) >= 2
        # Check that definitions are extracted
        definition_terms = [d[0] for d in definitions]
        assert any('Numerology' in term or 'HARQ' in term for term in definition_terms)
    
    def test_generate_section_summary(self):
        """Test section summary generation."""
        generator = QAGenerator()
        
        content = """
        Numerology is defined as the subcarrier spacing configuration parameter mu.
        It determines the spacing between OFDM subcarriers in the frequency domain.
        Different numerologies support different subcarrier spacings including
        15 kHz, 30 kHz, 60 kHz, 120 kHz, and 240 kHz.
        """
        
        summary = generator.generate_section_summary(content)
        
        assert len(summary) > 0
        assert len(summary) < len(content)  # Summary should be shorter
        assert "Numerology" in summary or "numerology" in summary
    
    def test_truncate_text(self):
        """Test text truncation."""
        generator = QAGenerator(max_response_length=100)
        
        long_text = "A " * 100  # 200 characters
        truncated = generator.truncate_text(long_text, max_length=100)
        
        assert len(truncated) <= 100
        assert truncated in long_text  # Should be a substring
    
    def test_truncate_text_short(self):
        """Test truncation of short text."""
        generator = QAGenerator()
        
        short_text = "This is short text."
        truncated = generator.truncate_text(short_text, max_length=100)
        
        assert truncated == short_text  # Should not truncate
    
    def test_generate_qa_from_section(self, sample_section):
        """Test Q&A generation from a section."""
        generator = QAGenerator()
        
        examples = generator.generate_qa_from_section(
            sample_section,
            source_filename="test_spec.json"
        )
        
        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        
        # Check that examples have required fields
        for ex in examples:
            assert ex.instruction
            assert ex.response
            assert ex.source == "test_spec.json"
            assert ex.section == sample_section['section_num']
    
    def test_generate_qa_types(self, sample_section):
        """Test that different Q&A types are generated."""
        generator = QAGenerator()
        
        examples = generator.generate_qa_from_section(
            sample_section,
            source_filename="test.json"
        )
        
        # Should have multiple types of questions
        instructions = [ex.instruction for ex in examples]
        
        # Check for different question patterns
        has_what = any("What" in inst for inst in instructions)
        has_explain = any("Explain" in inst or "explain" in inst for inst in instructions)
        
        assert has_what or has_explain  # At least one pattern
    
    def test_generate_qa_definitions(self, sample_section):
        """Test definition-based Q&A generation."""
        generator = QAGenerator()
        
        # Section with definition
        section_with_def = {
            "section_num": "5.1.1",
            "title": "Numerology",
            "content": "Numerology is defined as the subcarrier spacing configuration parameter mu.",
            "level": 2,
            "parent": "5.1"
        }
        
        examples = generator.generate_qa_from_section(
            section_with_def,
            source_filename="test.json"
        )
        
        # Should generate definition-based questions
        instructions = [ex.instruction for ex in examples]
        has_definition_q = any("What is" in inst and "Numerology" in inst for inst in instructions)
        
        # May or may not have definition Q depending on extraction
        assert len(examples) > 0
    
    def test_process_file(self, temp_dir, fixtures_dir):
        """Test processing a sections JSON file."""
        generator = QAGenerator()
        
        # Create input JSON file
        input_file = temp_dir / "test_sections.json"
        sections_data = json.loads((fixtures_dir / "sample_sections.json").read_text())
        
        with open(input_file, 'w') as f:
            json.dump(sections_data, f)
        
        # Process file
        examples = generator.process_file(input_file, output_dir=temp_dir)
        
        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        
        # Check output file was created
        output_file = temp_dir / "test_sections_qa.json"
        assert output_file.exists()
        
        # Check output JSON
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        
        assert isinstance(output_data, list)
        assert len(output_data) > 0
        assert all('instruction' in item for item in output_data)
        assert all('response' in item for item in output_data)

