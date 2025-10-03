"""
Dataset loading and preprocessing utilities.

Handles JSONL format conversion from prompt/completion to text,
with fallback to SageMaker environment variables.
"""

import os
import json
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def convert_jsonl_format(
    input_file: str,
    output_file: str,
    prompt_key: str = "prompt",
    completion_key: str = "completion"
) -> int:
    """
    Convert JSONL with prompt/completion format to text format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        prompt_key: Key name for prompt field
        completion_key: Key name for completion field
    
    Returns:
        Number of examples converted
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If JSONL is malformed
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Extract prompt and completion
                prompt = record.get(prompt_key, "").strip()
                completion = record.get(completion_key, "").strip()
                
                # Skip if either is empty
                if not prompt or not completion:
                    logger.warning(
                        f"Skipping line {line_num}: missing prompt or completion"
                    )
                    skipped_count += 1
                    continue
                
                # Combine into single text field
                text = f"{prompt}\n{completion}"
                
                # Write converted record
                converted_record = {"text": text}
                outfile.write(json.dumps(converted_record) + "\n")
                converted_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: invalid JSON - {e}")
                skipped_count += 1
                continue
    
    logger.info(
        f"Converted {converted_count} examples from {input_file} to {output_file}"
    )
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} invalid or empty lines")
    
    return converted_count


def find_training_file(
    train_file: Optional[str] = None,
    sagemaker_channel: str = "SM_CHANNEL_TRAIN"
) -> str:
    """
    Locate training file with priority: explicit path > SageMaker env > error.
    
    Args:
        train_file: Explicit path to training file (highest priority)
        sagemaker_channel: Environment variable name for SageMaker channel
    
    Returns:
        Path to training file
    
    Raises:
        FileNotFoundError: If no valid training file found
    """
    # Priority 1: Explicit train_file argument
    if train_file and os.path.isfile(train_file):
        logger.info(f"Using explicit training file: {train_file}")
        return train_file
    
    # Priority 2: SageMaker environment variable
    train_dir = os.environ.get(sagemaker_channel)
    if train_dir and os.path.isdir(train_dir):
        logger.info(f"Using SageMaker channel directory: {train_dir}")
        
        # Find first JSONL file
        jsonl_files = glob.glob(os.path.join(train_dir, "*.jsonl"))
        if jsonl_files:
            selected_file = jsonl_files[0]
            logger.info(f"Found training file: {selected_file}")
            return selected_file
        else:
            raise FileNotFoundError(
                f"No .jsonl files found in SageMaker directory: {train_dir}"
            )
    
    # Priority 3: Fallback to default path
    if train_file:
        # If specified but doesn't exist, try treating as relative path
        fallback_path = Path(train_file)
        if fallback_path.exists():
            logger.info(f"Using training file: {train_file}")
            return str(fallback_path)
    
    raise FileNotFoundError(
        f"No training file found. Provide --train_file or set {sagemaker_channel} environment variable."
    )


def load_distillation_dataset(
    train_file: Optional[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    text_key: str = "text"
) -> Dataset:
    """
    Load and tokenize dataset for distillation training.
    
    Args:
        train_file: Path to training JSONL file (or None for env lookup)
        tokenizer: Tokenizer for preprocessing
        max_length: Maximum sequence length
        text_key: Key name for text field in JSONL
    
    Returns:
        Tokenized HuggingFace Dataset
    
    Raises:
        FileNotFoundError: If training file cannot be located
        ValueError: If dataset is empty or invalid
    """
    # Locate training file
    resolved_file = find_training_file(train_file)
    
    logger.info(f"Loading dataset from: {resolved_file}")
    
    # Load dataset
    try:
        dataset = load_dataset("json", data_files={"train": resolved_file})
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {resolved_file}: {e}")
    
    if len(dataset["train"]) == 0:
        raise ValueError(f"Dataset is empty: {resolved_file}")
    
    logger.info(f"Loaded {len(dataset['train'])} training examples")
    
    # Tokenization function
    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text examples with truncation and padding."""
        # Handle different column names
        if text_key in examples:
            texts = examples[text_key]
        elif "prompt" in examples and "completion" in examples:
            # Handle prompt/completion format on-the-fly
            texts = [
                f"{p}\n{c}" for p, c in zip(examples["prompt"], examples["completion"])
            ]
        else:
            # Fallback to first column
            first_key = list(examples.keys())[0]
            texts = examples[first_key]
            logger.warning(f"Using fallback column '{first_key}' for text")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None  # Return lists for datasets
        )
        
        # Add labels for language modeling (copy of input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Apply tokenization
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info(f"Dataset tokenized: {len(tokenized_dataset)} examples")
    
    return tokenized_dataset

