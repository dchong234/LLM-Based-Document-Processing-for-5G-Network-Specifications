"""
Text Cleaner for 3GPP Specification Documents
Cleans extracted text by removing headers, footers, page numbers, and artifacts
while preserving document structure.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import config
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError:
    # Fallback if config.py is not available
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")


class TextCleaner:
    """Clean extracted text from 3GPP PDF documents."""
    
    def __init__(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize text cleaner.
        
        Args:
            input_dir: Directory containing extracted text files (defaults to processed_data/)
            output_dir: Directory to save cleaned text files (defaults to input_dir)
        """
        self.input_dir = Path(input_dir) if input_dir else Path(PROCESSED_DATA_DIR)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'successful_cleanings': 0,
            'failed_cleanings': 0,
            'total_characters_before': 0,
            'total_characters_after': 0,
            'total_lines_before': 0,
            'total_lines_after': 0,
            'files_processed': []
        }
        
        # Compile regex patterns for 3GPP document artifacts
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for common 3GPP document artifacts."""
        
        # 3GPP header patterns (e.g., "3GPP TS 38.211 V17.0.0")
        self.patterns = {
            # 3GPP specification headers
            'header_3gpp': re.compile(
                r'^3GPP\s+TS\s+\d+\.\d{3,4}\s+V?\d+\.\d+\.\d+.*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Release information (e.g., "Release 17", "Rel-17")
            'release_info': re.compile(
                r'\b(Release|Rel-?)\s*\d+\b',
                re.IGNORECASE
            ),
            # Page numbers (standalone or in headers/footers)
            'page_number': re.compile(
                r'^\s*Page\s+\d+\s*$|^\s*\d+\s*$|^\s*\d+\s+of\s+\d+\s*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Document title patterns in headers
            'document_title_header': re.compile(
                r'^3GPP\s+(Technical\s+)?(Specification|Spec|TS).*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Footer patterns (copyright, date, etc.)
            'footer_copyright': re.compile(
                r'©\s*\d{4}\s*.*3GPP.*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Date patterns in headers/footers (YYYY-MM-DD, YYYY/MM/DD)
            'date_footer': re.compile(
                r'^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*$',
                re.MULTILINE
            ),
            # Table/figure references that are artifacts
            'table_figure_ref': re.compile(
                r'^\s*(Table|Figure|Fig\.?)\s+\d+[.\-\s]*\d*\s*:?\s*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Repeated header/footer patterns (appear multiple times)
            'repeated_header': re.compile(
                r'^.*3GPP.*TS.*\d+\.\d+.*$',
                re.IGNORECASE | re.MULTILINE
            ),
            # Section number artifacts (standalone numbers that are likely page numbers)
            'standalone_section_numbers': re.compile(
                r'^\s*\d+\.\d*\.?\d*\.?\d*\s*$',
                re.MULTILINE
            ),
            # Multiple consecutive line breaks (more than 2)
            'excessive_line_breaks': re.compile(r'\n{3,}'),
            # Multiple consecutive spaces (more than 2)
            'excessive_spaces': re.compile(r' {3,}'),
            # Tabs and other whitespace artifacts
            'tabs': re.compile(r'\t+'),
            # Leading/trailing whitespace on lines
            'line_whitespace': re.compile(r'^[\s\t]+|[\s\t]+$', re.MULTILINE),
            # Artifacts from PDF extraction (control characters, etc.)
            'control_characters': re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]'),
            # Form feed characters
            'form_feeds': re.compile(r'\f'),
            # Repeated dashes or separators
            'repeated_separators': re.compile(r'-{5,}|={5,}|\*{5,}'),
        }
        
        # Patterns to preserve (section numbers with text)
        self.preserve_patterns = {
            # Section numbers with titles (e.g., "5.2.1 Title")
            'section_with_title': re.compile(
                r'^\s*(\d+\.\d+\.?\d*\.?\d*\.?\d*)\s+([A-Z][^\n]{1,100})$',
                re.MULTILINE
            ),
            # Subsection numbers
            'subsection': re.compile(
                r'^\s*(\d+\.\d+\.?\d*\.?\d*)\s+',
                re.MULTILINE
            ),
        }
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove 3GPP headers and footers from text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text with headers/footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip header patterns
            if self.patterns['header_3gpp'].match(line.strip()):
                continue
            if self.patterns['document_title_header'].match(line.strip()):
                continue
            if self.patterns['release_info'].search(line) and len(line.strip()) < 50:
                continue
            if self.patterns['footer_copyright'].match(line.strip()):
                continue
            if self.patterns['date_footer'].match(line.strip()):
                continue
            
            # Skip page numbers (standalone)
            if self.patterns['page_number'].match(line.strip()):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_page_numbers(self, text: str) -> str:
        """
        Remove page numbers from text.
        
        Args:
            text: Text content
            
        Returns:
            Text with page numbers removed
        """
        # Remove standalone page numbers
        text = self.patterns['page_number'].sub('', text)
        
        # Remove "Page X" patterns
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_table_figure_artifacts(self, text: str) -> str:
        """
        Remove table and figure artifacts while preserving references in context.
        
        Args:
            text: Text content
            
        Returns:
            Text with table/figure artifacts removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip standalone table/figure references (likely artifacts)
            if self.patterns['table_figure_ref'].match(stripped):
                # Check if next line has content (if not, it's likely an artifact)
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_whitespace(self, text: str) -> str:
        """
        Clean excessive whitespace and line breaks.
        
        Args:
            text: Text content
            
        Returns:
            Text with cleaned whitespace
        """
        # Remove control characters
        text = self.patterns['control_characters'].sub('', text)
        
        # Remove form feeds
        text = self.patterns['form_feeds'].sub('\n', text)
        
        # Remove tabs (replace with space)
        text = self.patterns['tabs'].sub(' ', text)
        
        # Remove excessive spaces within lines (more than 2 consecutive)
        text = self.patterns['excessive_spaces'].sub(' ', text)
        
        # Remove leading/trailing whitespace on lines (but preserve empty lines)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                cleaned_lines.append(line.strip())
            else:  # Empty line - preserve but normalize
                cleaned_lines.append('')
        
        # Remove excessive line breaks (more than 2 consecutive empty lines)
        # But preserve double line breaks (paragraph breaks)
        text = '\n'.join(cleaned_lines)
        text = self.patterns['excessive_line_breaks'].sub('\n\n', text)
        
        # Remove repeated separators
        text = self.patterns['repeated_separators'].sub('', text)
        
        # Remove empty lines at the beginning
        lines = text.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove empty lines at the end
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)
    
    def preserve_section_structure(self, text: str) -> str:
        """
        Ensure section structure is preserved (numbered sections).
        
        Args:
            text: Text content
            
        Returns:
            Text with preserved section structure
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Preserve section numbers with titles (e.g., "5.2.1 Title")
            if self.preserve_patterns['section_with_title'].match(stripped):
                # Ensure section headers are on their own line with proper spacing
                if cleaned_lines and cleaned_lines[-1].strip():
                    cleaned_lines.append('')  # Add blank line before section
                cleaned_lines.append(stripped)
                cleaned_lines.append('')  # Add blank line after section header
            # Preserve subsection numbers
            elif self.preserve_patterns['subsection'].match(stripped):
                # Check if this looks like a section header (has content after number)
                if len(stripped) > 10 and not stripped.endswith('.'):
                    if cleaned_lines and cleaned_lines[-1].strip():
                        cleaned_lines.append('')
                    cleaned_lines.append(stripped)
                    cleaned_lines.append('')
                else:
                    # Just a number, likely not a section header
                    cleaned_lines.append(stripped)
            elif stripped:
                # Regular content line
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1].strip():
                # Empty line, but only add if previous line had content
                cleaned_lines.append('')
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return text
        
        # Store original stats
        original_length = len(text)
        original_lines = len(text.split('\n'))
        
        # Apply cleaning operations in order
        cleaned = text
        
        # Step 1: Remove headers and footers (before other cleaning)
        cleaned = self.remove_headers_footers(cleaned)
        
        # Step 2: Remove page numbers
        cleaned = self.remove_page_numbers(cleaned)
        
        # Step 3: Remove table/figure artifacts
        cleaned = self.remove_table_figure_artifacts(cleaned)
        
        # Step 4: Initial whitespace cleanup
        cleaned = self.clean_whitespace(cleaned)
        
        # Step 5: Preserve section structure (adds proper formatting)
        cleaned = self.preserve_section_structure(cleaned)
        
        # Step 6: Final whitespace cleanup (normalize spacing after structure preservation)
        cleaned = self.clean_whitespace(cleaned)
        
        # Update statistics
        self.stats['total_characters_before'] += original_length
        self.stats['total_characters_after'] += len(cleaned)
        self.stats['total_lines_before'] += original_lines
        self.stats['total_lines_after'] += len(cleaned.split('\n'))
        
        return cleaned
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single text file: clean and save.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if not text.strip():
                print(f"  ⚠ File is empty: {file_path.name}")
                return False
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text.strip():
                print(f"  ⚠ All text removed during cleaning: {file_path.name}")
                return False
            
            # Generate output filename
            output_filename = file_path.stem + "_cleaned.txt"
            output_path = self.output_dir / output_filename
            
            # Save cleaned text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Calculate reduction
            original_size = len(text)
            cleaned_size = len(cleaned_text)
            reduction = ((original_size - cleaned_size) / original_size * 100) if original_size > 0 else 0
            
            print(f"  ✓ Cleaned: {original_size:,} → {cleaned_size:,} chars "
                  f"({reduction:.1f}% reduction)")
            print(f"  ✓ Saved to: {output_path}")
            
            self.stats['successful_cleanings'] += 1
            self.stats['files_processed'].append(file_path.name)
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            self.stats['failed_cleanings'] += 1
            return False
    
    def get_text_files(self) -> List[Path]:
        """
        Get all text files from the input directory (excluding cleaned files).
        
        Returns:
            List of text file paths
        """
        if not self.input_dir.exists():
            return []
        
        # Get all .txt files, excluding already cleaned files
        text_files = []
        for pattern in ["*.txt"]:
            files = list(self.input_dir.glob(pattern))
            # Exclude files that are already cleaned
            text_files.extend([f for f in files if not f.name.endswith("_cleaned.txt")])
        
        return sorted(text_files)
    
    def process_all(self) -> dict:
        """
        Process all text files in the input directory.
        
        Returns:
            Dictionary with cleaning statistics
        """
        # Get all text files
        text_files = self.get_text_files()
        self.stats['total_files'] = len(text_files)
        
        if len(text_files) == 0:
            print(f"No text files found in {self.input_dir}")
            print("Note: Files ending with '_cleaned.txt' are skipped")
            return self.stats
        
        print(f"\n{'=' * 60}")
        print("Text Cleaning")
        print(f"{'=' * 60}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(text_files)} text file(s)")
        print(f"{'=' * 60}\n")
        
        # Process each file
        for idx, file_path in enumerate(text_files, 1):
            print(f"\n[{idx}/{len(text_files)}] Processing: {file_path.name}")
            try:
                self.process_file(file_path)
            except KeyboardInterrupt:
                print("\n\n⚠ Cleaning interrupted by user")
                break
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                self.stats['failed_cleanings'] += 1
                continue
        
        # Print statistics
        self.print_statistics()
        
        return self.stats
    
    def print_statistics(self):
        """Print cleaning statistics."""
        print(f"\n{'=' * 60}")
        print("Cleaning Statistics")
        print(f"{'=' * 60}")
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Successful cleanings: {self.stats['successful_cleanings']}")
        print(f"Failed cleanings: {self.stats['failed_cleanings']}")
        print(f"Total characters before: {self.stats['total_characters_before']:,}")
        print(f"Total characters after: {self.stats['total_characters_after']:,}")
        
        if self.stats['total_characters_before'] > 0:
            total_reduction = ((self.stats['total_characters_before'] - 
                              self.stats['total_characters_after']) / 
                             self.stats['total_characters_before'] * 100)
            print(f"Total reduction: {total_reduction:.1f}%")
        
        print(f"Total lines before: {self.stats['total_lines_before']:,}")
        print(f"Total lines after: {self.stats['total_lines_after']:,}")
        
        if self.stats['total_lines_before'] > 0:
            line_reduction = ((self.stats['total_lines_before'] - 
                             self.stats['total_lines_after']) / 
                            self.stats['total_lines_before'] * 100)
            print(f"Line reduction: {line_reduction:.1f}%")
        
        print(f"{'=' * 60}\n")
        
        # Calculate averages
        if self.stats['successful_cleanings'] > 0:
            avg_chars_before = self.stats['total_characters_before'] / self.stats['successful_cleanings']
            avg_chars_after = self.stats['total_characters_after'] / self.stats['successful_cleanings']
            print(f"Average characters per file (before): {avg_chars_before:,.0f}")
            print(f"Average characters per file (after): {avg_chars_after:,.0f}")
        print()


def main():
    """Main function to process all text files in the processed_data/ directory."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean extracted text from 3GPP PDF specification documents"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory containing extracted text files (default: {PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save cleaned text files (default: same as input directory)'
    )
    
    args = parser.parse_args()
    
    # Create cleaner and process all files
    cleaner = TextCleaner(input_dir=args.input_dir, output_dir=args.output_dir)
    stats = cleaner.process_all()
    
    # Exit with appropriate code
    if stats['failed_cleanings'] == 0 and stats['total_files'] > 0:
        print("✓ All files processed successfully!")
        return 0
    elif stats['successful_cleanings'] > 0:
        print("⚠ Some files failed to process. Check the output above for details.")
        return 1
    else:
        print("✗ No files were successfully processed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

