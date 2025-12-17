"""
Unit tests for text cleaner
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    def test_init(self, temp_dir):
        """Test TextCleaner initialization."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        assert cleaner.input_dir == temp_dir
        assert cleaner.output_dir == temp_dir
        assert len(cleaner.patterns) > 0
        assert len(cleaner.preserve_patterns) > 0
    
    def test_remove_headers_footers(self, temp_dir):
        """Test header and footer removal."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        text_with_header = """3GPP TS 38.211 V17.0.0 (2022-03)
Release 17
Technical Specification

5.1 Content here"""
        
        cleaned = cleaner.remove_headers_footers(text_with_header)
        
        assert "3GPP TS 38.211" not in cleaned
        assert "Release 17" not in cleaned
        assert "5.1 Content here" in cleaned
    
    def test_remove_page_numbers(self, temp_dir):
        """Test page number removal."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        text_with_pages = """Some content here

Page 15
Page 16

More content"""
        
        cleaned = cleaner.remove_page_numbers(text_with_pages)
        
        assert "Page 15" not in cleaned
        assert "Page 16" not in cleaned
        assert "Some content here" in cleaned
        assert "More content" in cleaned
    
    def test_remove_table_figure_artifacts(self, temp_dir):
        """Test table/figure artifact removal."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        text_with_artifacts = """Some content

Table 5.1
Figure 3.2

More content"""
        
        cleaned = cleaner.remove_table_figure_artifacts(text_with_artifacts)
        
        assert "Table 5.1" not in cleaned
        assert "Figure 3.2" not in cleaned
        assert "Some content" in cleaned
    
    def test_clean_whitespace(self, temp_dir):
        """Test whitespace cleaning."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        text_with_whitespace = """Some    content    here


Multiple    spaces    and    breaks


More   content"""
        
        cleaned = cleaner.clean_whitespace(text_with_whitespace)
        
        # Should have single spaces
        assert "    " not in cleaned  # No multiple spaces
        assert "\n\n\n" not in cleaned  # No excessive line breaks
        assert "\n\n" in cleaned or "\n" in cleaned  # But some breaks preserved
    
    def test_preserve_section_structure(self, temp_dir):
        """Test that section structure is preserved."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        text_with_sections = """5.1 Title One
Content for section one
5.1.1 Subsection
Content for subsection
5.2 Title Two
Content for section two"""
        
        cleaned = cleaner.preserve_section_structure(text_with_sections)
        
        # Should preserve section numbers with titles
        assert "5.1 Title One" in cleaned
        assert "5.1.1 Subsection" in cleaned
        assert "5.2 Title Two" in cleaned
    
    def test_clean_text_full_pipeline(self, temp_dir, sample_text):
        """Test full cleaning pipeline."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        cleaned = cleaner.clean_text(sample_text)
        
        # Should remove headers/footers
        assert "3GPP TS 38.211" not in cleaned
        assert "© 2022" not in cleaned
        assert "Page 15" not in cleaned
        
        # Should preserve content
        assert "Physical Layer Overview" in cleaned
        assert "Numerology" in cleaned
        assert "subcarrier spacing" in cleaned
    
    def test_get_text_files(self, temp_dir):
        """Test getting text files from directory."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        # Create test files
        (temp_dir / "test1.txt").write_text("content")
        (temp_dir / "test2.txt").write_text("content")
        (temp_dir / "test1_cleaned.txt").write_text("cleaned")  # Should be excluded
        (temp_dir / "other.pdf").write_text("pdf")
        
        text_files = cleaner.get_text_files()
        
        # Should only get .txt files, not _cleaned.txt
        assert len(text_files) == 2
        assert all(f.suffix == '.txt' for f in text_files)
        assert all('_cleaned' not in f.name for f in text_files)
    
    def test_process_file(self, temp_dir, fixtures_dir):
        """Test processing a single file."""
        cleaner = TextCleaner(input_dir=temp_dir)
        
        # Create test input file
        input_file = temp_dir / "test.txt"
        sample_text = (fixtures_dir / "sample.txt").read_text()
        input_file.write_text(sample_text)
        
        # Process file
        success = cleaner.process_file(input_file)
        
        assert success is True
        
        # Check output file created
        output_file = temp_dir / "test_cleaned.txt"
        assert output_file.exists()
        
        # Check cleaned content
        cleaned_content = output_file.read_text()
        assert "3GPP TS" not in cleaned_content
        assert "© 2022" not in cleaned_content
        assert "Page" not in cleaned_content
        assert "Physical Layer" in cleaned_content

