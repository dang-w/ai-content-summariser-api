import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Import the SummariserService from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.summariser import SummariserService

# Test with mocked model
def test_summariser_with_mock():
    # Create patches for the model and tokenizer
    with patch('app.services.summariser.AutoTokenizer') as mock_tokenizer_class, \
         patch('app.services.summariser.AutoModelForSeq2SeqLM') as mock_model_class:

        # Set up the mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "This is a test summary."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Set up the mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4]]  # Dummy token IDs
        mock_model.to.return_value = mock_model  # Handle device placement
        mock_model_class.from_pretrained.return_value = mock_model

        # Create the summarizer with our mocked dependencies
        summariser = SummariserService()

        # Test the summarize method
        text = "This is a test paragraph that should be summarized."
        summary = summariser.summarise(text, max_length=50, min_length=10)

        # Verify the result
        assert summary == "This is a test summary."

        # Verify the mocks were called correctly
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()

# Test with real model but adjusted expectations
def test_summariser():
    summariser = SummariserService()
    text = "This is a test paragraph that should be summarized. It contains multiple sentences with different information. The summarizer should extract the key points and generate a concise summary."

    # The actual model might not strictly adhere to max_length in characters
    # It uses tokens, which don't directly map to character count
    # Let's adjust our test to account for this
    summary = summariser.summarise(text, max_length=50, min_length=10)

    assert summary is not None

    # Instead of checking exact character count, let's check that it's
    # significantly shorter than the original text
    assert len(summary) < len(text) * 0.8
    assert len(summary) >= 10  # Still enforce minimum length
