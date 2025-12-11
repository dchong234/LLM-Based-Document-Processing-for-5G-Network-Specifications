"""
Section Parser for 3GPP Specification Documents
Parses cleaned text and extracts structured sections with numbers, titles, and content.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import config
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
except ImportError:
    # Fallback if config.py is not available
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")


@dataclass
class Section:
    """Represents a document section."""
    section_num: str
    title: str
    content: str
    level: int  # Nesting level (1, 2, 3, etc.)
    parent: Optional[str] = None  # Parent section number
    
    def to_dict(self) -> Dict:
        """Convert section to dictionary."""
        return {
            'section_num': self.section_num,
            'title': self.title,
            'content': self.content,
            'level': self.level,
            'parent': self.parent,
            'content_length': len(self.content)
        }


class SectionParser:
    """Parse document sections from cleaned text."""
    
    def __init__(self, min_section_length: int = 100):
        """
        Initialize section parser.
        
        Args:
            min_section_length: Minimum character length for a section to be included
        """
        self.min_section_length = min_section_length
        
        # Compile regex patterns for section identification
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for section identification."""
        
        # Pattern for section headers: "5.2.1 Section Title"
        # Matches: number (1-5 levels), optional space, title (starts with capital letter)
        self.section_pattern = re.compile(
            r'^(\d+(?:\.\d+){0,4})\s+(.+)$',
            re.MULTILINE
        )
        
        # Pattern to extract section number and determine level
        self.section_num_pattern = re.compile(
            r'^(\d+(?:\.\d+){0,4})$'
        )
    
    def _parse_section_number(self, section_num: str) -> Tuple[List[int], int]:
        """
        Parse section number into components and determine level.
        
        Args:
            section_num: Section number string (e.g., "5.2.1")
            
        Returns:
            Tuple of (number components, level)
        """
        parts = [int(x) for x in section_num.split('.')]
        level = len(parts)
        return parts, level
    
    def _find_parent_section(self, current_section: str, sections: List[Section]) -> Optional[str]:
        """
        Find parent section for a given section number.
        
        Args:
            current_section: Current section number (e.g., "5.2.1")
            sections: List of previously parsed sections
            
        Returns:
            Parent section number or None
        """
        current_parts, current_level = self._parse_section_number(current_section)
        
        # Find the closest parent (higher level, matching prefix)
        parent = None
        parent_level = 0
        
        for section in reversed(sections):  # Check in reverse order
            section_parts, section_level = self._parse_section_number(section.section_num)
            
            # Parent must be at a higher level (lower number)
            if section_level < current_level:
                # Check if current section is a child of this section
                if section_parts == current_parts[:section_level]:
                    # This is a potential parent
                    if section_level > parent_level:
                        parent = section.section_num
                        parent_level = section_level
        
        return parent
    
    def _extract_section_content(self, lines: List[str], start_idx: int, next_section_idx: int) -> str:
        """
        Extract content for a section between two section headers.
        
        Args:
            lines: List of text lines
            start_idx: Starting line index (after section header)
            next_section_idx: Next section header index
            
        Returns:
            Section content as a string
        """
        content_lines = []
        
        for i in range(start_idx, next_section_idx):
            line = lines[i].strip()
            if line:  # Skip empty lines at start
                content_lines.append(line)
        
        # Join lines and clean up
        content = ' '.join(content_lines)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def parse_sections(self, text: str) -> List[Section]:
        """
        Parse sections from cleaned text.
        
        Args:
            text: Cleaned text content
            
        Returns:
            List of Section objects
        """
        if not text or not text.strip():
            return []
        
        lines = text.split('\n')
        sections = []
        section_matches = []
        
        # Find all section headers
        for i, line in enumerate(lines):
            match = self.section_pattern.match(line.strip())
            if match:
                section_num = match.group(1)
                title = match.group(2).strip()
                
                # Validate section number (should start with digit)
                if section_num and title:
                    section_matches.append({
                        'line_idx': i,
                        'section_num': section_num,
                        'title': title
                    })
        
        if not section_matches:
            return []
        
        # Extract content for each section
        for i, match in enumerate(section_matches):
            section_num = match['section_num']
            title = match['title']
            start_idx = match['line_idx'] + 1  # Start after header line
            
            # Find next section (or end of document)
            if i + 1 < len(section_matches):
                end_idx = section_matches[i + 1]['line_idx']
            else:
                end_idx = len(lines)
            
            # Extract content
            content = self._extract_section_content(lines, start_idx, end_idx)
            
            # Filter out very short sections
            if len(content) < self.min_section_length:
                continue
            
            # Determine level and parent
            _, level = self._parse_section_number(section_num)
            parent = self._find_parent_section(section_num, sections)
            
            # Create section object
            section = Section(
                section_num=section_num,
                title=title,
                content=content,
                level=level,
                parent=parent
            )
            
            sections.append(section)
        
        return sections
    
    def sections_to_dict(self, sections: List[Section]) -> List[Dict]:
        """
        Convert sections to list of dictionaries.
        
        Args:
            sections: List of Section objects
            
        Returns:
            List of dictionaries
        """
        return [section.to_dict() for section in sections]
    
    def save_sections_to_json(self, sections: List[Section], output_path: Path):
        """
        Save sections to JSON file.
        
        Args:
            sections: List of Section objects
            output_path: Path to save JSON file
        """
        sections_dict = self.sections_to_dict(sections)
        
        output_data = {
            'total_sections': len(sections),
            'sections': sections_dict
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def process_file(self, file_path: Path, output_path: Optional[Path] = None) -> List[Section]:
        """
        Process a text file and extract sections.
        
        Args:
            file_path: Path to cleaned text file
            output_path: Path to save JSON file (optional)
            
        Returns:
            List of Section objects
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                print(f"  ⚠ File is empty: {file_path.name}")
                return []
            
            # Parse sections
            sections = self.parse_sections(text)
            
            if not sections:
                print(f"  ⚠ No sections found in: {file_path.name}")
                return []
            
            print(f"  ✓ Found {len(sections)} sections in {file_path.name}")
            
            # Save to JSON if output path is provided
            if output_path:
                self.save_sections_to_json(sections, output_path)
                print(f"  ✓ Saved sections to: {output_path}")
            
            return sections
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            return []
    
    def process_all_files(self, input_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, List[Section]]:
        """
        Process all cleaned text files in a directory.
        
        Args:
            input_dir: Directory containing cleaned text files
            output_dir: Directory to save JSON files (optional)
            
        Returns:
            Dictionary mapping filenames to section lists
        """
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return {}
        
        # Get all cleaned text files
        text_files = list(input_dir.glob("*_cleaned.txt"))
        
        if not text_files:
            print(f"No cleaned text files found in {input_dir}")
            print("Note: Looking for files ending with '_cleaned.txt'")
            return {}
        
        print(f"\n{'=' * 60}")
        print("Section Parsing")
        print(f"{'=' * 60}")
        print(f"Input directory: {input_dir}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print(f"Found {len(text_files)} file(s)")
        print(f"{'=' * 60}\n")
        
        all_sections = {}
        
        # Process each file
        for idx, file_path in enumerate(sorted(text_files), 1):
            print(f"\n[{idx}/{len(text_files)}] Processing: {file_path.name}")
            
            # Determine output path
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = file_path.stem.replace('_cleaned', '') + '_sections.json'
                output_path = output_dir / output_filename
            else:
                output_path = None
            
            # Process file
            sections = self.process_file(file_path, output_path)
            all_sections[file_path.name] = sections
        
        # Print summary
        total_sections = sum(len(sections) for sections in all_sections.values())
        print(f"\n{'=' * 60}")
        print("Parsing Summary")
        print(f"{'=' * 60}")
        print(f"Total files processed: {len(text_files)}")
        print(f"Total sections extracted: {total_sections}")
        print(f"Average sections per file: {total_sections / len(text_files):.1f}" if text_files else "0")
        print(f"{'=' * 60}\n")
        
        return all_sections


def create_sample_3gpp_document() -> str:
    """
    Create a sample 3GPP document structure for testing.
    
    Returns:
        Sample 3GPP document text
    """
    sample_text = """
5 Introduction
This section provides an introduction to the 3GPP specification document.
It covers the scope, purpose, and overview of the technical specification.

5.1 Scope
This section defines the scope of the specification. It outlines the objectives
and boundaries of the technical document. The scope includes various aspects
of the 5G network architecture and protocols.

5.1.1 Objectives
The objectives of this specification include defining the physical layer
procedures for 5G networks. This involves specifying channel structures,
reference signals, and modulation schemes.

5.1.2 Boundaries
The boundaries of this specification are defined by the 3GPP release
requirements and compatibility with previous releases.

5.2 Overview
This section provides an overview of the technical content covered in this
specification. It describes the key concepts and terminology used throughout
the document.

5.2.1 Channel Structure
The channel structure defines how data is organized and transmitted over
the physical layer. This includes the structure of physical channels,
transport channels, and logical channels.

5.2.1.1 Physical Channels
Physical channels are the fundamental transmission units in the 5G system.
They include the Physical Downlink Shared Channel (PDSCH), Physical Uplink
Shared Channel (PUSCH), and various control channels.

5.2.1.2 Transport Channels
Transport channels provide services to the higher layers and are mapped
to physical channels. Examples include the Downlink Shared Channel (DL-SCH)
and Uplink Shared Channel (UL-SCH).

5.2.2 Reference Signals
Reference signals are used for channel estimation, synchronization, and
measurement purposes. They are essential for reliable communication in
the 5G system.

5.2.2.1 Downlink Reference Signals
Downlink reference signals include the Cell-Specific Reference Signal (CRS)
and the Channel State Information Reference Signal (CSI-RS).

5.2.2.2 Uplink Reference Signals
Uplink reference signals include the Demodulation Reference Signal (DMRS)
and the Sounding Reference Signal (SRS).

6 Physical Layer Procedures
This section describes the physical layer procedures for 5G networks.
It covers various aspects of signal processing and transmission.

6.1 Channel Coding
Channel coding is used to improve the reliability of data transmission.
This section describes the coding schemes used in 5G systems.
"""
    return sample_text.strip()


def test_parser():
    """Test the section parser with a sample 3GPP document."""
    print("=" * 60)
    print("Testing Section Parser")
    print("=" * 60)
    
    # Create sample document
    sample_text = create_sample_3gpp_document()
    
    # Create parser
    parser = SectionParser(min_section_length=50)
    
    # Parse sections
    sections = parser.parse_sections(sample_text)
    
    # Print results
    print(f"\nFound {len(sections)} sections:\n")
    
    for section in sections:
        print(f"Section {section.section_num} (Level {section.level})")
        if section.parent:
            print(f"  Parent: {section.parent}")
        print(f"  Title: {section.title}")
        print(f"  Content length: {len(section.content)} characters")
        print(f"  Content preview: {section.content[:100]}...")
        print()
    
    # Test JSON export
    print("=" * 60)
    print("Testing JSON Export")
    print("=" * 60)
    
    sections_dict = parser.sections_to_dict(sections)
    json_output = json.dumps(sections_dict, indent=2, ensure_ascii=False)
    print(json_output[:500] + "...")  # Print first 500 chars
    
    # Test parent relationships
    print("\n" + "=" * 60)
    print("Testing Parent Relationships")
    print("=" * 60)
    
    for section in sections:
        if section.parent:
            print(f"Section {section.section_num} -> Parent: {section.parent}")
        else:
            print(f"Section {section.section_num} -> No parent (top-level)")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def main():
    """Main function to process cleaned text files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse sections from cleaned 3GPP specification documents"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory containing cleaned text files (default: {PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save JSON files (default: same as input directory)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='Minimum section length in characters (default: 100)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with sample 3GPP document'
    )
    
    args = parser.parse_args()
    
    # Run test if requested
    if args.test:
        test_parser()
        return 0
    
    # Create parser
    section_parser = SectionParser(min_section_length=args.min_length)
    
    # Determine output directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Process all files
    all_sections = section_parser.process_all_files(input_dir, output_dir)
    
    # Exit with appropriate code
    if all_sections:
        total_sections = sum(len(sections) for sections in all_sections.values())
        if total_sections > 0:
            print("✓ Section parsing completed successfully!")
            return 0
        else:
            print("⚠ No sections were extracted from the files.")
            return 1
    else:
        print("✗ No files were processed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

