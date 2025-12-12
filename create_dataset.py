"""
Dataset Creator for Llama 3 8B Fine-tuning
Loads Q&A pairs and formats them for training in JSONL and chat formats.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

try:
    import config
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
    RANDOM_SEED = config.RANDOM_SEED
    TRAIN_SPLIT = config.TRAIN_SPLIT
except ImportError:
    # Fallback if config.py is not available
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.9


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_examples: int = 0
    train_examples: int = 0
    val_examples: int = 0
    avg_instruction_length: float = 0.0
    avg_context_length: float = 0.0
    avg_response_length: float = 0.0
    sources: Dict[str, int] = None
    sections: Dict[str, int] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = defaultdict(int)
        if self.sections is None:
            self.sections = defaultdict(int)


class DatasetCreator:
    """Create training datasets from Q&A pairs."""
    
    # Required fields for Q&A examples
    REQUIRED_FIELDS = ['instruction', 'context', 'response', 'source', 'section']
    
    def __init__(self, train_split: float = TRAIN_SPLIT, random_seed: int = RANDOM_SEED):
        """
        Initialize dataset creator.
        
        Args:
            train_split: Fraction of data to use for training (default: 0.9)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.train_split = train_split
        self.val_split = 1.0 - train_split
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
    
    def validate_example(self, example: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate that an example has all required fields.
        
        Args:
            example: Example dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for field in self.REQUIRED_FIELDS:
            if field not in example:
                return False, f"Missing required field: {field}"
            
            if not example[field] or not str(example[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, None
    
    def load_qa_files(self, input_dir: Path) -> List[Dict]:
        """
        Load all Q&A JSON files from a directory.
        
        Args:
            input_dir: Directory containing Q&A JSON files
            
        Returns:
            List of all Q&A examples
        """
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return []
        
        # Get all Q&A JSON files
        qa_files = list(input_dir.glob("*_qa.json"))
        
        # Also check for combined file
        combined_file = input_dir / "all_qa_examples.json"
        if combined_file.exists():
            qa_files.append(combined_file)
        
        if not qa_files:
            print(f"No Q&A JSON files found in {input_dir}")
            print("Note: Looking for files ending with '_qa.json' or 'all_qa_examples.json'")
            return []
        
        print(f"\n{'=' * 60}")
        print("Loading Q&A Files")
        print(f"{'=' * 60}")
        print(f"Found {len(qa_files)} file(s)")
        print(f"{'=' * 60}\n")
        
        all_examples = []
        invalid_examples = 0
        
        # Load examples from each file
        for qa_file in sorted(qa_files):
            print(f"Loading: {qa_file.name}")
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get examples from file
                if isinstance(data, list):
                    examples = data
                elif isinstance(data, dict) and 'examples' in data:
                    examples = data['examples']
                else:
                    print(f"  ⚠ Invalid file format: {qa_file.name}")
                    continue
                
                # Validate and add examples
                file_valid = 0
                file_invalid = 0
                
                for example in examples:
                    is_valid, error_msg = self.validate_example(example)
                    if is_valid:
                        all_examples.append(example)
                        file_valid += 1
                    else:
                        file_invalid += 1
                        if file_invalid == 1:  # Print first error as example
                            print(f"  ⚠ Invalid example: {error_msg}")
                
                print(f"  ✓ Loaded {file_valid} valid examples")
                if file_invalid > 0:
                    print(f"  ⚠ Skipped {file_invalid} invalid examples")
                    invalid_examples += file_invalid
                
            except Exception as e:
                print(f"  ✗ Error loading {qa_file.name}: {e}")
                continue
        
        print(f"\nTotal examples loaded: {len(all_examples)}")
        if invalid_examples > 0:
            print(f"Total invalid examples skipped: {invalid_examples}")
        
        return all_examples
    
    def format_instruction_tuning(self, example: Dict) -> Dict:
        """
        Format example for instruction tuning (JSONL format).
        
        Args:
            example: Q&A example dictionary
            
        Returns:
            Formatted example for instruction tuning
        """
        return {
            'instruction': example['instruction'],
            'input': example.get('context', ''),
            'output': example['response'],
        }
    
    def format_chat(self, example: Dict, system_message: str = None) -> Dict:
        """
        Format example for chat format (system/user/assistant messages).
        
        Args:
            example: Q&A example dictionary
            system_message: System message (optional)
            
        Returns:
            Formatted example for chat format
        """
        if system_message is None:
            system_message = "You are a helpful assistant that answers questions about 3GPP 5G specifications based on the provided context."
        
        # Build user message
        user_message = example['instruction']
        if example.get('context'):
            user_message = f"Context: {example['context']}\n\nQuestion: {example['instruction']}"
        
        # Build assistant message
        assistant_message = example['response']
        
        return {
            'messages': [
                {
                    'role': 'system',
                    'content': system_message
                },
                {
                    'role': 'user',
                    'content': user_message
                },
                {
                    'role': 'assistant',
                    'content': assistant_message
                }
            ]
        }
    
    def calculate_statistics(self, examples: List[Dict]) -> DatasetStats:
        """
        Calculate dataset statistics.
        
        Args:
            examples: List of examples
            
        Returns:
            DatasetStats object
        """
        stats = DatasetStats()
        stats.total_examples = len(examples)
        
        if not examples:
            return stats
        
        # Calculate average lengths
        instruction_lengths = []
        context_lengths = []
        response_lengths = []
        
        for example in examples:
            instruction_lengths.append(len(str(example.get('instruction', ''))))
            context_lengths.append(len(str(example.get('context', ''))))
            response_lengths.append(len(str(example.get('response', ''))))
            
            # Count sources and sections
            if 'source' in example:
                stats.sources[example['source']] += 1
            if 'section' in example:
                stats.sections[example['section']] += 1
        
        stats.avg_instruction_length = sum(instruction_lengths) / len(instruction_lengths)
        stats.avg_context_length = sum(context_lengths) / len(context_lengths)
        stats.avg_response_length = sum(response_lengths) / len(response_lengths)
        
        return stats
    
    def split_data(self, examples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train and validation sets.
        
        Args:
            examples: List of all examples
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split index
        split_index = int(len(shuffled) * self.train_split)
        
        # Split data
        train_examples = shuffled[:split_index]
        val_examples = shuffled[split_index:]
        
        return train_examples, val_examples
    
    def save_jsonl(self, examples: List[Dict], output_path: Path, format_type: str = 'instruction'):
        """
        Save examples to JSONL file.
        
        Args:
            examples: List of examples
            output_path: Path to save JSONL file
            format_type: Format type ('instruction' or 'chat')
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                if format_type == 'instruction':
                    formatted = self.format_instruction_tuning(example)
                elif format_type == 'chat':
                    formatted = self.format_chat(example)
                else:
                    formatted = example
                
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    def print_statistics(self, stats: DatasetStats, train_stats: DatasetStats, val_stats: DatasetStats):
        """
        Print dataset statistics.
        
        Args:
            stats: Overall statistics
            train_stats: Training set statistics
            val_stats: Validation set statistics
        """
        print(f"\n{'=' * 60}")
        print("Dataset Statistics")
        print(f"{'=' * 60}")
        
        # Overall statistics
        print(f"\nOverall:")
        print(f"  Total examples: {stats.total_examples:,}")
        print(f"  Average instruction length: {stats.avg_instruction_length:.1f} characters")
        print(f"  Average context length: {stats.avg_context_length:.1f} characters")
        print(f"  Average response length: {stats.avg_response_length:.1f} characters")
        
        # Training set statistics
        print(f"\nTraining Set:")
        print(f"  Examples: {train_stats.total_examples:,} ({train_stats.total_examples/stats.total_examples*100:.1f}%)")
        print(f"  Average instruction length: {train_stats.avg_instruction_length:.1f} characters")
        print(f"  Average context length: {train_stats.avg_context_length:.1f} characters")
        print(f"  Average response length: {train_stats.avg_response_length:.1f} characters")
        
        # Validation set statistics
        print(f"\nValidation Set:")
        print(f"  Examples: {val_stats.total_examples:,} ({val_stats.total_examples/stats.total_examples*100:.1f}%)")
        print(f"  Average instruction length: {val_stats.avg_instruction_length:.1f} characters")
        print(f"  Average context length: {val_stats.avg_context_length:.1f} characters")
        print(f"  Average response length: {val_stats.avg_response_length:.1f} characters")
        
        # Source distribution
        print(f"\nSource Distribution:")
        for source, count in sorted(stats.sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source}: {count:,} examples ({count/stats.total_examples*100:.1f}%)")
        
        # Section distribution (top 10)
        print(f"\nTop 10 Sections:")
        for section, count in sorted(stats.sections.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Section {section}: {count:,} examples ({count/stats.total_examples*100:.1f}%)")
        
        print(f"\n{'=' * 60}\n")
    
    def create_dataset(self, input_dir: Path, output_dir: Path) -> bool:
        """
        Create training datasets from Q&A files.
        
        Args:
            input_dir: Directory containing Q&A JSON files
            output_dir: Directory to save dataset files
            
        Returns:
            True if successful, False otherwise
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Q&A examples
        all_examples = self.load_qa_files(input_dir)
        
        if not all_examples:
            print("✗ No valid examples found. Cannot create dataset.")
            return False
        
        # Calculate overall statistics
        stats = self.calculate_statistics(all_examples)
        
        # Split data
        print(f"\n{'=' * 60}")
        print("Splitting Data")
        print(f"{'=' * 60}")
        print(f"Train split: {self.train_split*100:.1f}%")
        print(f"Validation split: {self.val_split*100:.1f}%")
        print(f"Random seed: {self.random_seed}")
        
        train_examples, val_examples = self.split_data(all_examples)
        
        print(f"Training examples: {len(train_examples):,}")
        print(f"Validation examples: {len(val_examples):,}")
        print(f"{'=' * 60}\n")
        
        # Calculate split statistics
        train_stats = self.calculate_statistics(train_examples)
        val_stats = self.calculate_statistics(val_examples)
        
        # Save instruction tuning format (JSONL)
        print(f"\n{'=' * 60}")
        print("Saving Datasets")
        print(f"{'=' * 60}")
        
        training_instruction_path = output_dir / "training_data.jsonl"
        validation_instruction_path = output_dir / "validation_data.jsonl"
        training_chat_path = output_dir / "training_data_chat.jsonl"
        
        print(f"\nSaving instruction tuning format...")
        self.save_jsonl(train_examples, training_instruction_path, format_type='instruction')
        print(f"  ✓ Saved: {training_instruction_path} ({len(train_examples):,} examples)")
        
        self.save_jsonl(val_examples, validation_instruction_path, format_type='instruction')
        print(f"  ✓ Saved: {validation_instruction_path} ({len(val_examples):,} examples)")
        
        print(f"\nSaving chat format...")
        self.save_jsonl(train_examples, training_chat_path, format_type='chat')
        print(f"  ✓ Saved: {training_chat_path} ({len(train_examples):,} examples)")
        
        print(f"{'=' * 60}\n")
        
        # Print statistics
        self.print_statistics(stats, train_stats, val_stats)
        
        # Validate output files
        print(f"\n{'=' * 60}")
        print("Validating Output Files")
        print(f"{'=' * 60}")
        
        files_to_check = [
            training_instruction_path,
            validation_instruction_path,
            training_chat_path
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                # Count lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"  ✓ {file_path.name}: {line_count:,} examples")
            else:
                print(f"  ✗ {file_path.name}: File not found")
        
        print(f"{'=' * 60}\n")
        
        return True


def main():
    """Main function to create training datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create training datasets from Q&A pairs"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f'Directory containing Q&A JSON files (default: {PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save dataset files (default: same as input directory)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=TRAIN_SPLIT,
        help=f'Train/validation split ratio (default: {TRAIN_SPLIT})'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Create dataset creator
    creator = DatasetCreator(
        train_split=args.train_split,
        random_seed=args.random_seed
    )
    
    # Create dataset
    success = creator.create_dataset(input_dir, output_dir)
    
    # Exit with appropriate code
    if success:
        print("✓ Dataset creation completed successfully!")
        return 0
    else:
        print("✗ Dataset creation failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

