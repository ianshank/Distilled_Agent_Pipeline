"""
Unit tests for dataset loading and conversion.

Tests JSONL format conversion, file discovery, tokenization,
and error handling.
"""

import os
import json
import pytest
from pathlib import Path

from nlm.data import load_distillation_dataset, convert_jsonl_format


class TestConvertJsonlFormat:
    """Test JSONL format conversion from prompt/completion to text."""
    
    def test_basic_conversion(self, sample_jsonl_prompt_completion, temp_dir):
        """Test successful conversion of prompt/completion to text."""
        output_file = temp_dir / "converted.jsonl"
        
        count = convert_jsonl_format(
            str(sample_jsonl_prompt_completion),
            str(output_file)
        )
        
        assert count == 3
        assert output_file.exists()
        
        # Verify output format
        with open(output_file) as f:
            for line in f:
                record = json.loads(line)
                assert "text" in record
                assert "\n" in record["text"]  # Prompt and completion combined
    
    def test_missing_input_file(self, temp_dir):
        """Test error on missing input file."""
        with pytest.raises(FileNotFoundError):
            convert_jsonl_format(
                "nonexistent.jsonl",
                str(temp_dir / "output.jsonl")
            )
    
    def test_skip_empty_lines(self, temp_dir):
        """Test skipping empty lines and incomplete records."""
        input_file = temp_dir / "malformed.jsonl"
        
        with open(input_file, "w") as f:
            f.write('{"prompt": "Valid", "completion": "Response"}\n')
            f.write('\n')  # Empty line
            f.write('{"prompt": "", "completion": "Missing prompt"}\n')
            f.write('{"prompt": "Missing completion", "completion": ""}\n')
        
        output_file = temp_dir / "output.jsonl"
        count = convert_jsonl_format(str(input_file), str(output_file))
        
        assert count == 1  # Only first line valid
    
    def test_malformed_json(self, temp_dir):
        """Test handling of malformed JSON lines."""
        input_file = temp_dir / "bad_json.jsonl"
        
        with open(input_file, "w") as f:
            f.write('{"prompt": "Valid", "completion": "Response"}\n')
            f.write('not valid json\n')
            f.write('{"prompt": "Another", "completion": "Valid"}\n')
        
        output_file = temp_dir / "output.jsonl"
        count = convert_jsonl_format(str(input_file), str(output_file))
        
        assert count == 2  # Two valid lines


class TestFindTrainingFile:
    """Test training file discovery logic."""
    
    def test_explicit_file_priority(self, sample_jsonl_text):
        """Test explicit file path takes priority."""
        from nlm.data.dataset_loader import find_training_file
        
        result = find_training_file(train_file=str(sample_jsonl_text))
        assert result == str(sample_jsonl_text)
    
    def test_sagemaker_env_fallback(self, temp_dir, sample_jsonl_text, monkeypatch):
        """Test fallback to SageMaker environment variable."""
        from nlm.data.dataset_loader import find_training_file
        
        # Create SageMaker-style directory
        sm_dir = temp_dir / "sagemaker_data"
        sm_dir.mkdir()
        (sm_dir / "train.jsonl").write_text('{"text":"test"}')
        
        monkeypatch.setenv("SM_CHANNEL_TRAIN", str(sm_dir))
        
        result = find_training_file(train_file=None)
        assert str(sm_dir) in result
        assert result.endswith(".jsonl")
    
    def test_no_file_found_error(self):
        """Test error when no file can be located."""
        from nlm.data.dataset_loader import find_training_file
        
        with pytest.raises(FileNotFoundError):
            find_training_file(train_file=None)


class TestLoadDistillationDataset:
    """Test dataset loading and tokenization."""
    
    def test_load_text_format(self, sample_jsonl_text):
        """Test loading dataset with text field."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_distillation_dataset(
            train_file=str(sample_jsonl_text),
            tokenizer=tokenizer,
            max_length=64
        )
        
        assert len(dataset) == 3
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "labels" in dataset.column_names
    
    def test_load_prompt_completion_format(self, sample_jsonl_prompt_completion):
        """Test loading dataset with prompt/completion fields."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_distillation_dataset(
            train_file=str(sample_jsonl_prompt_completion),
            tokenizer=tokenizer,
            max_length=64
        )
        
        assert len(dataset) == 3
        assert "input_ids" in dataset.column_names
    
    def test_empty_dataset_error(self, temp_dir):
        """Test error on empty dataset."""
        from transformers import AutoTokenizer
        
        empty_file = temp_dir / "empty.jsonl"
        empty_file.touch()
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        with pytest.raises(ValueError, match="empty"):
            load_distillation_dataset(
                train_file=str(empty_file),
                tokenizer=tokenizer
            )
    
    def test_tokenization_length_limit(self, sample_jsonl_text):
        """Test max_length truncation in tokenization."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_distillation_dataset(
            train_file=str(sample_jsonl_text),
            tokenizer=tokenizer,
            max_length=32  # Short length
        )
        
        # All sequences should be padded to max_length
        for example in dataset:
            assert len(example["input_ids"]) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

