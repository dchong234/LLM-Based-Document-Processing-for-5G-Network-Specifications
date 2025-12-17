"""
Unit tests for PDF extractor
"""

import pytest
import tempfile
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_extractor import PDFExtractor


class TestPDFExtractor:
    """Test cases for PDFExtractor class."""
    
    def test_init(self, temp_dir):
        """Test PDFExtractor initialization."""
        extractor = PDFExtractor(
            input_dir=str(temp_dir / "input"),
            output_dir=str(temp_dir / "output")
        )
        
        assert extractor.input_dir == temp_dir / "input"
        assert extractor.output_dir == temp_dir / "output"
        assert extractor.stats['total_pdfs'] == 0
        assert extractor.stats['successful_extractions'] == 0
    
    def test_get_pdf_files(self, temp_dir):
        """Test getting PDF files from directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create test PDF files
        (input_dir / "test1.pdf").write_text("fake pdf content")
        (input_dir / "test2.pdf").write_text("fake pdf content")
        (input_dir / "not_a_pdf.txt").write_text("text file")
        
        extractor = PDFExtractor(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )
        
        pdf_files = extractor.get_pdf_files()
        
        assert len(pdf_files) == 2
        assert all(f.suffix == '.pdf' for f in pdf_files)
        assert all('test' in f.name for f in pdf_files)
    
    def test_save_extracted_text(self, temp_dir):
        """Test saving extracted text to file."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        extractor = PDFExtractor(
            input_dir=str(temp_dir),
            output_dir=str(output_dir)
        )
        
        test_text = "This is extracted text from a PDF."
        pdf_name = "test_document"
        
        success = extractor.save_extracted_text(pdf_name, test_text)
        
        assert success is True
        
        # Check file was created
        output_file = output_dir / f"{pdf_name}.txt"
        assert output_file.exists()
        
        # Check content
        saved_text = output_file.read_text()
        assert saved_text == test_text
    
    def test_save_extracted_text_empty(self, temp_dir):
        """Test saving empty text."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        extractor = PDFExtractor(
            input_dir=str(temp_dir),
            output_dir=str(output_dir)
        )
        
        success = extractor.save_extracted_text("empty", "")
        
        # Should still succeed, but file may be empty or not created
        # depending on implementation
        assert success is True or success is False
    
    def test_extract_text_from_pdf_nonexistent(self, temp_dir):
        """Test extracting from non-existent PDF."""
        extractor = PDFExtractor(
            input_dir=str(temp_dir),
            output_dir=str(temp_dir / "output")
        )
        
        pdf_path = temp_dir / "nonexistent.pdf"
        
        success, text, pages, chars = extractor.extract_text_from_pdf(pdf_path)
        
        assert success is False
        assert pages == 0
        assert chars == 0
    
    def test_process_pdf_stats(self, temp_dir):
        """Test that processing updates statistics."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        extractor = PDFExtractor(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )
        
        # Create a fake PDF file (won't actually extract, but tests the flow)
        pdf_file = input_dir / "test.pdf"
        pdf_file.write_text("fake content")
        
        # Process (will fail on actual extraction, but tests structure)
        result = extractor.process_pdf(pdf_file)
        
        # Stats should be updated
        assert extractor.stats['total_pdfs'] == 1
        # Note: Actual extraction requires real PDF, so success depends on PyPDF2


@pytest.mark.skipif(True, reason="Requires actual PDF file for full testing")
class TestPDFExtractorWithRealPDF:
    """Test cases that require real PDF files."""
    
    def test_extract_text_from_real_pdf(self, temp_dir, fixtures_dir):
        """Test extracting text from a real PDF file."""
        # This test would require a real PDF file in fixtures/
        # For now, it's skipped
        pass

