"""
PDF Text Extractor for 3GPP Specification Documents
Extracts text from PDF files and saves to individual .txt files.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 is not installed. Install with: pip install PyPDF2")

try:
    import config
    SPECS_DIR = config.SPECS_DIR
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError:
    # Fallback if config.py is not available
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPECS_DIR = os.path.join(BASE_DIR, "specs")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")


class PDFExtractor:
    """Extract text from PDF files."""
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None):
        """
        Initialize PDF extractor.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save extracted text files (defaults to processed_data/)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_pdfs': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_pages': 0,
            'total_characters': 0,
            'corrupted_files': []
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[bool, str, int, int]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (success, text, page_count, character_count)
        """
        if not PYPDF2_AVAILABLE:
            return False, "", 0, 0
        
        try:
            text_content = []
            page_count = 0
            character_count = 0
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                            character_count += len(page_text)
                    except Exception as e:
                        # Continue processing other pages if one page fails
                        print(f"    Warning: Failed to extract text from page {page_num}: {e}")
                        continue
            
            full_text = "\n\n".join(text_content)
            return True, full_text, page_count, character_count
            
        except Exception as e:
            error_msg = f"Error extracting text from {pdf_path.name}: {str(e)}"
            return False, error_msg, 0, 0
    
    def save_extracted_text(self, pdf_name: str, text: str) -> bool:
        """
        Save extracted text to a .txt file.
        
        Args:
            pdf_name: Name of the original PDF file (without extension)
            text: Extracted text content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = self.output_dir / f"{pdf_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"    Error saving text file: {e}")
            return False
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file: extract text and save to .txt file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        pdf_name = pdf_path.stem  # Filename without extension
        
        print(f"Processing: {pdf_path.name}")
        
        # Extract text
        success, text, page_count, char_count = self.extract_text_from_pdf(pdf_path)
        
        if not success:
            print(f"  ✗ Failed to extract text: {text}")
            self.stats['failed_extractions'] += 1
            self.stats['corrupted_files'].append(pdf_path.name)
            return False
        
        if page_count == 0:
            print(f"  ⚠ No pages found in PDF")
            self.stats['failed_extractions'] += 1
            self.stats['corrupted_files'].append(pdf_path.name)
            return False
        
        if char_count == 0:
            print(f"  ⚠ No text extracted from PDF (may be image-based)")
            self.stats['failed_extractions'] += 1
            self.stats['corrupted_files'].append(pdf_path.name)
            return False
        
        # Save extracted text
        if self.save_extracted_text(pdf_name, text):
            print(f"  ✓ Extracted {page_count} pages, {char_count:,} characters")
            print(f"  ✓ Saved to: {self.output_dir / f'{pdf_name}.txt'}")
            
            # Update statistics
            self.stats['successful_extractions'] += 1
            self.stats['total_pages'] += page_count
            self.stats['total_characters'] += char_count
            return True
        else:
            print(f"  ✗ Failed to save extracted text")
            self.stats['failed_extractions'] += 1
            return False
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get all PDF files from the input directory.
        
        Returns:
            List of PDF file paths
        """
        if not self.input_dir.exists():
            return []
        
        # Get all PDF files (case-insensitive)
        pdf_files = []
        for pattern in ["*.pdf", "*.PDF"]:
            pdf_files.extend(self.input_dir.glob(pattern))
        
        # Remove duplicates and sort
        pdf_files = list(set(pdf_files))
        return sorted(pdf_files)
    
    def process_all(self) -> dict:
        """
        Process all PDF files in the input directory.
        
        Returns:
            Dictionary with extraction statistics
        """
        if not PYPDF2_AVAILABLE:
            print("Error: PyPDF2 is not installed. Install with: pip install PyPDF2")
            return self.stats
        
        if not self.input_dir.exists():
            print(f"Error: Input directory does not exist: {self.input_dir}")
            return self.stats
        
        # Get all PDF files
        pdf_files = self.get_pdf_files()
        self.stats['total_pdfs'] = len(pdf_files)
        
        if len(pdf_files) == 0:
            print(f"No PDF files found in {self.input_dir}")
            return self.stats
        
        print(f"\n{'=' * 60}")
        print(f"PDF Text Extraction")
        print(f"{'=' * 60}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(pdf_files)} PDF file(s)")
        print(f"{'=' * 60}\n")
        
        # Process each PDF
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] ", end="")
            try:
                self.process_pdf(pdf_path)
            except KeyboardInterrupt:
                print("\n\n⚠ Extraction interrupted by user")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Unexpected error: {error_msg}")
                self.stats['failed_extractions'] += 1
                self.stats['corrupted_files'].append(pdf_path.name)
                # Continue with next file
                continue
        
        # Print statistics
        self.print_statistics()
        
        return self.stats
    
    def print_statistics(self):
        """Print extraction statistics."""
        print(f"\n{'=' * 60}")
        print("Extraction Statistics")
        print(f"{'=' * 60}")
        print(f"Total PDFs processed: {self.stats['total_pdfs']}")
        print(f"Successful extractions: {self.stats['successful_extractions']}")
        print(f"Failed extractions: {self.stats['failed_extractions']}")
        print(f"Total pages processed: {self.stats['total_pages']:,}")
        print(f"Total characters extracted: {self.stats['total_characters']:,}")
        
        if self.stats['corrupted_files']:
            print(f"\nCorrupted/Failed files ({len(self.stats['corrupted_files'])}):")
            for file in self.stats['corrupted_files']:
                print(f"  - {file}")
        
        print(f"{'=' * 60}\n")
        
        # Calculate success rate
        if self.stats['total_pdfs'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['total_pdfs']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # Calculate average
        if self.stats['successful_extractions'] > 0:
            avg_pages = self.stats['total_pages'] / self.stats['successful_extractions']
            avg_chars = self.stats['total_characters'] / self.stats['successful_extractions']
            print(f"Average pages per PDF: {avg_pages:.1f}")
            print(f"Average characters per PDF: {avg_chars:,.0f}")
        print()


def main():
    """Main function to process all PDFs in the specs/ directory."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract text from 3GPP PDF specification documents"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=SPECS_DIR,
        help=f'Directory containing PDF files (default: {SPECS_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory to save extracted text files (default: {PROCESSED_DATA_DIR})'
    )
    
    args = parser.parse_args()
    
    # Create extractor and process all PDFs
    extractor = PDFExtractor(input_dir=args.input_dir, output_dir=args.output_dir)
    stats = extractor.process_all()
    
    # Exit with appropriate code
    if stats['failed_extractions'] == 0 and stats['total_pdfs'] > 0:
        print("✓ All PDFs processed successfully!")
        return 0
    elif stats['successful_extractions'] > 0:
        print("⚠ Some PDFs failed to process. Check the output above for details.")
        return 1
    else:
        print("✗ No PDFs were successfully processed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

