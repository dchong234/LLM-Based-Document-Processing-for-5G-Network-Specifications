"""
QA Generator for 3GPP Specification Documents
Generates training examples (Q&A pairs) from parsed sections.
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
class TrainingExample:
    """Represents a training example."""
    instruction: str
    context: str
    response: str
    source: str  # Source file name
    section: str  # Section number
    
    def to_dict(self) -> Dict:
        """Convert training example to dictionary."""
        return {
            'instruction': self.instruction,
            'context': self.context,
            'response': self.response,
            'source': self.source,
            'section': self.section
        }


class QAGenerator:
    """Generate Q&A training examples from sections."""
    
    def __init__(self, max_context_length: int = 500, max_response_length: int = 1000):
        """
        Initialize QA generator.
        
        Args:
            max_context_length: Maximum length for context (characters)
            max_response_length: Maximum length for response (characters)
        """
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        
        # Compile regex patterns for definition extraction
        self._compile_patterns()
        
        # Question templates
        self._init_question_templates()
    
    def _compile_patterns(self):
        """Compile regex patterns for definition extraction."""
        
        # Patterns for extracting definitions
        self.definition_patterns = {
            # "X is defined as Y"
            'is_defined_as': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s+is\s+defined\s+as\s+(.+?)(?:\.|$)',
                re.IGNORECASE
            ),
            # "X refers to Y"
            'refers_to': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s+refers\s+to\s+(.+?)(?:\.|$)',
                re.IGNORECASE
            ),
            # "X is Y"
            'is_pattern': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]{3,}?)\s+is\s+(?:a|an|the)?\s+(.+?)(?:\.|,|$)',
                re.IGNORECASE
            ),
            # "X means Y"
            'means': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s+means?\s+(.+?)(?:\.|$)',
                re.IGNORECASE
            ),
            # "X denotes Y"
            'denotes': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s+denotes?\s+(.+?)(?:\.|$)',
                re.IGNORECASE
            ),
            # "X represents Y"
            'represents': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s+represents?\s+(.+?)(?:\.|$)',
                re.IGNORECASE
            ),
            # Acronyms: "X (Y)" or "Y (X)"
            'acronym': re.compile(
                r'([A-Z][A-Za-z0-9\s\-]+?)\s*\(([A-Z]{2,})\)|\(([A-Z]{2,})\)\s*([A-Z][A-Za-z0-9\s\-]+?)',
                re.IGNORECASE
            ),
        }
    
    def _init_question_templates(self):
        """Initialize question templates."""
        
        self.question_templates = {
            'section_description': [
                "What does section {section_num} describe?",
                "What is covered in section {section_num}?",
                "What does section {section_num} discuss?",
                "What is the content of section {section_num}?",
            ],
            'section_explanation': [
                "Explain {title}",
                "What is {title}?",
                "Can you explain {title}?",
                "Describe {title}",
                "What does {title} mean?",
            ],
            'section_summary': [
                "Summarize section {section_num}: {title}",
                "Provide a summary of section {section_num}",
                "What is the main point of section {section_num}?",
                "Give an overview of section {section_num}",
            ],
            'section_content': [
                "What is the content of section {section_num}?",
                "What information is in section {section_num}?",
                "Tell me about section {section_num}",
            ],
            'definition': [
                "What is {term}?",
                "Define {term}",
                "What does {term} mean?",
                "Explain {term}",
                "What is the definition of {term}?",
            ],
            'context_question': [
                "Based on section {section_num}, {question}",
                "According to section {section_num}, {question}",
                "In section {section_num}, {question}",
            ],
        }
    
    def extract_definitions(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract definitions from text.
        
        Args:
            text: Text content to extract definitions from
            
        Returns:
            List of (term, definition) tuples
        """
        definitions = []
        seen_terms = set()
        
        # Extract definitions using various patterns
        for pattern_name, pattern in self.definition_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                if pattern_name == 'acronym':
                    # Handle acronym patterns differently
                    if match.group(1) and match.group(2):
                        term = match.group(1).strip()
                        acronym = match.group(2).strip()
                        definition = f"{term} ({acronym})"
                        term_key = acronym.lower()
                    elif match.group(3) and match.group(4):
                        acronym = match.group(3).strip()
                        term = match.group(4).strip()
                        definition = f"{term} ({acronym})"
                        term_key = acronym.lower()
                    else:
                        continue
                else:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    term_key = term.lower()
                
                # Clean up term and definition
                term = re.sub(r'\s+', ' ', term).strip()
                definition = re.sub(r'\s+', ' ', definition).strip()
                
                # Skip if term is too short or too long
                if len(term) < 3 or len(term) > 100:
                    continue
                
                # Skip if definition is too short
                if len(definition) < 10:
                    continue
                
                # Skip if we've seen this term before
                if term_key in seen_terms:
                    continue
                
                # Skip common false positives
                if term.lower() in ['this', 'that', 'these', 'those', 'it', 'they']:
                    continue
                
                seen_terms.add(term_key)
                definitions.append((term, definition))
        
        return definitions
    
    def generate_section_summary(self, content: str) -> str:
        """
        Generate a summary of section content.
        
        Args:
            content: Section content
            
        Returns:
            Summary text
        """
        # Simple summary: take first sentence or first 200 characters
        sentences = re.split(r'[.!?]\s+', content)
        
        if sentences:
            # Use first sentence if it's reasonable length
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 50 and len(first_sentence) < 300:
                return first_sentence + '.'
            else:
                # Use first 200 characters
                summary = content[:200].strip()
                # Find last complete word
                last_space = summary.rfind(' ')
                if last_space > 100:
                    summary = summary[:last_space] + '...'
                return summary
        
        return content[:200].strip() + '...'
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length, preserving word boundaries.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we found a space reasonably close
            truncated = truncated[:last_space] + '...'
        else:
            truncated = truncated + '...'
        
        return truncated
    
    def generate_section_qa_pairs(self, section: Dict, source: str) -> List[TrainingExample]:
        """
        Generate Q&A pairs for a section.
        
        Args:
            section: Section dictionary with section_num, title, content
            source: Source file name
            
        Returns:
            List of training examples
        """
        examples = []
        section_num = section['section_num']
        title = section['title']
        content = section['content']
        
        # Generate section description questions
        for template in self.question_templates['section_description']:
            instruction = template.format(section_num=section_num)
            summary = self.generate_section_summary(content)
            response = f"Section {section_num} ({title}): {summary}"
            
            example = TrainingExample(
                instruction=instruction,
                context=title,
                response=self.truncate_text(response, self.max_response_length),
                source=source,
                section=section_num
            )
            examples.append(example)
        
        # Generate section explanation questions
        for template in self.question_templates['section_explanation']:
            instruction = template.format(title=title)
            context_content = self.truncate_text(content, self.max_context_length)
            response = self.truncate_text(content, self.max_response_length)
            
            example = TrainingExample(
                instruction=instruction,
                context=context_content,
                response=response,
                source=source,
                section=section_num
            )
            examples.append(example)
        
        # Generate section summary questions
        for template in self.question_templates['section_summary']:
            instruction = template.format(section_num=section_num, title=title)
            summary = self.generate_section_summary(content)
            response = summary
            
            example = TrainingExample(
                instruction=instruction,
                context=title,
                response=self.truncate_text(response, self.max_response_length),
                source=source,
                section=section_num
            )
            examples.append(example)
        
        # Generate section content questions
        for template in self.question_templates['section_content']:
            instruction = template.format(section_num=section_num)
            context_content = self.truncate_text(content, self.max_context_length)
            response = self.truncate_text(content, self.max_response_length)
            
            example = TrainingExample(
                instruction=instruction,
                context=context_content,
                response=response,
                source=source,
                section=section_num
            )
            examples.append(example)
        
        # Generate definition-based Q&A pairs
        definitions = self.extract_definitions(content)
        
        for term, definition in definitions:
            for template in self.question_templates['definition']:
                instruction = template.format(term=term)
                response = f"{term} is {definition}"
                
                example = TrainingExample(
                    instruction=instruction,
                    context=title,  # Use section title as context
                    response=self.truncate_text(response, self.max_response_length),
                    source=source,
                    section=section_num
                )
                examples.append(example)
        
        # Generate context-passage pairs
        # Split content into chunks for context
        content_chunks = self._split_content_into_chunks(content, chunk_size=self.max_context_length)
        
        for chunk in content_chunks:
            # Create a question about the chunk
            instruction = f"What information is provided in section {section_num} about {title}?"
            response = chunk
            
            example = TrainingExample(
                instruction=instruction,
                context=title,
                response=self.truncate_text(response, self.max_response_length),
                source=source,
                section=section_num
            )
            examples.append(example)
        
        return examples
    
    def _split_content_into_chunks(self, content: str, chunk_size: int, overlap: int = 50) -> List[str]:
        """
        Split content into overlapping chunks.
        
        Args:
            content: Content to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            if end >= len(content):
                chunks.append(content[start:].strip())
                break
            
            # Try to break at sentence boundary
            chunk = content[start:end]
            last_period = chunk.rfind('.')
            last_exclamation = chunk.rfind('!')
            last_question = chunk.rfind('?')
            
            last_sentence_end = max(last_period, last_exclamation, last_question)
            
            if last_sentence_end > chunk_size * 0.7:  # If we found a sentence end reasonably close
                end = start + last_sentence_end + 1
                chunk = content[start:end].strip()
            else:
                # Break at word boundary
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:
                    end = start + last_space
                    chunk = content[start:end].strip()
                else:
                    chunk = content[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_sections(self, sections: List[Dict], source: str) -> List[TrainingExample]:
        """
        Process sections and generate training examples.
        
        Args:
            sections: List of section dictionaries
            source: Source file name
            
        Returns:
            List of training examples
        """
        all_examples = []
        
        for section in sections:
            examples = self.generate_section_qa_pairs(section, source)
            all_examples.extend(examples)
        
        return all_examples
    
    def process_json_file(self, json_path: Path, source_name: Optional[str] = None) -> List[TrainingExample]:
        """
        Process a JSON file with sections and generate training examples.
        
        Args:
            json_path: Path to JSON file with sections
            source_name: Source file name (if None, derived from JSON path)
            
        Returns:
            List of training examples
        """
        try:
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get source name
            if source_name is None:
                source_name = json_path.stem.replace('_sections', '')
            
            # Get sections
            sections = data.get('sections', [])
            
            if not sections:
                print(f"  ⚠ No sections found in {json_path.name}")
                return []
            
            # Process sections
            examples = self.process_sections(sections, source_name)
            
            return examples
            
        except Exception as e:
            print(f"  ✗ Error processing {json_path.name}: {e}")
            return []
    
    def save_examples_to_json(self, examples: List[TrainingExample], output_path: Path):
        """
        Save training examples to JSON file.
        
        Args:
            examples: List of training examples
            output_path: Path to save JSON file
        """
        examples_dict = [example.to_dict() for example in examples]
        
        output_data = {
            'total_examples': len(examples),
            'examples': examples_dict
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def process_all_files(self, input_dir: Path, output_dir: Optional[Path] = None) -> List[TrainingExample]:
        """
        Process all JSON files with sections in a directory.
        
        Args:
            input_dir: Directory containing JSON files with sections
            output_dir: Directory to save training examples (optional)
            
        Returns:
            List of all training examples
        """
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return []
        
        # Get all JSON files with sections
        json_files = list(input_dir.glob("*_sections.json"))
        
        if not json_files:
            print(f"No section JSON files found in {input_dir}")
            print("Note: Looking for files ending with '_sections.json'")
            return []
        
        print(f"\n{'=' * 60}")
        print("QA Generation")
        print(f"{'=' * 60}")
        print(f"Input directory: {input_dir}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print(f"Found {len(json_files)} file(s)")
        print(f"{'=' * 60}\n")
        
        all_examples = []
        
        # Process each file
        for idx, json_path in enumerate(sorted(json_files), 1):
            print(f"\n[{idx}/{len(json_files)}] Processing: {json_path.name}")
            
            # Process file
            examples = self.process_json_file(json_path)
            
            if examples:
                print(f"  ✓ Generated {len(examples)} training examples")
                all_examples.extend(examples)
                
                # Save individual file if output directory is specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_filename = json_path.stem.replace('_sections', '') + '_qa.json'
                    output_path = output_dir / output_filename
                    self.save_examples_to_json(examples, output_path)
                    print(f"  ✓ Saved to: {output_path}")
            else:
                print(f"  ⚠ No examples generated")
        
        # Print summary
        print(f"\n{'=' * 60}")
        print("Generation Summary")
        print(f"{'=' * 60}")
        print(f"Total files processed: {len(json_files)}")
        print(f"Total examples generated: {len(all_examples)}")
        print(f"Average examples per file: {len(all_examples) / len(json_files):.1f}" if json_files else "0")
        print(f"{'=' * 60}\n")
        
        # Save combined file if output directory is specified
        if output_dir and all_examples:
            output_dir.mkdir(parents=True, exist_ok=True)
            combined_output_path = output_dir / "all_qa_examples.json"
            self.save_examples_to_json(all_examples, combined_output_path)
            print(f"✓ Saved combined examples to: {combined_output_path}\n")
        
        return all_examples


def test_qa_generator():
    """Test the QA generator with sample sections."""
    print("=" * 60)
    print("Testing QA Generator")
    print("=" * 60)
    
    # Create sample sections
    sample_sections = [
        {
            'section_num': '5.2.1',
            'title': 'Channel Structure',
            'content': 'The channel structure defines how data is organized and transmitted over the physical layer. Physical channels are the fundamental transmission units in the 5G system. They include the Physical Downlink Shared Channel (PDSCH), Physical Uplink Shared Channel (PUSCH), and various control channels. PDSCH is defined as the primary downlink data channel. PUSCH refers to the primary uplink data channel.',
            'level': 3,
            'parent': '5.2',
            'content_length': 400
        },
        {
            'section_num': '5.2.2',
            'title': 'Reference Signals',
            'content': 'Reference signals are used for channel estimation, synchronization, and measurement purposes. They are essential for reliable communication in the 5G system. CRS means Cell-Specific Reference Signal. DMRS denotes Demodulation Reference Signal.',
            'level': 3,
            'parent': '5.2',
            'content_length': 250
        }
    ]
    
    # Create generator
    generator = QAGenerator(max_context_length=500, max_response_length=1000)
    
    # Process sections
    examples = generator.process_sections(sample_sections, 'test_spec.txt')
    
    # Print results
    print(f"\nGenerated {len(examples)} training examples:\n")
    
    for i, example in enumerate(examples[:10], 1):  # Show first 10
        print(f"Example {i}:")
        print(f"  Instruction: {example.instruction}")
        print(f"  Context: {example.context[:100]}...")
        print(f"  Response: {example.response[:100]}...")
        print(f"  Source: {example.source}")
        print(f"  Section: {example.section}")
        print()
    
    # Test definition extraction
    print("=" * 60)
    print("Testing Definition Extraction")
    print("=" * 60)
    
    test_text = "PDSCH is defined as the primary downlink data channel. PUSCH refers to the primary uplink data channel. CRS means Cell-Specific Reference Signal."
    definitions = generator.extract_definitions(test_text)
    
    print(f"Found {len(definitions)} definitions:")
    for term, definition in definitions:
        print(f"  {term} -> {definition}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def main():
    """Main function to generate training examples from section JSON files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Q&A training examples from parsed sections"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory containing section JSON files (default: {PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save training examples (default: same as input directory)'
    )
    parser.add_argument(
        '--max-context',
        type=int,
        default=500,
        help='Maximum context length in characters (default: 500)'
    )
    parser.add_argument(
        '--max-response',
        type=int,
        default=1000,
        help='Maximum response length in characters (default: 1000)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with sample sections'
    )
    
    args = parser.parse_args()
    
    # Run test if requested
    if args.test:
        test_qa_generator()
        return 0
    
    # Create generator
    generator = QAGenerator(
        max_context_length=args.max_context,
        max_response_length=args.max_response
    )
    
    # Determine output directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Process all files
    all_examples = generator.process_all_files(input_dir, output_dir)
    
    # Exit with appropriate code
    if all_examples:
        print("✓ QA generation completed successfully!")
        return 0
    else:
        print("⚠ No training examples were generated.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

