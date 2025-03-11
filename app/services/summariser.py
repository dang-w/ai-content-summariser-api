import numpy as np  # Import NumPy first
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummariserService:
    def __init__(self):
        # Initialize with a smaller model for faster loading
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def summarise(self, text, max_length=150, min_length=50, do_sample=False, temperature=1.0):
        """
        Summarise the given text using the loaded model.

        Args:
            text (str): The text to summarise
            max_length (int): Maximum length of the summary in characters
            min_length (int): Minimum length of the summary in characters
            do_sample (bool): Whether to use sampling for generation
            temperature (float): Sampling temperature (higher = more random)

        Returns:
            str: The generated summary
        """
        # Convert character lengths to approximate token counts
        # A rough estimate is that 1 token ≈ 4 characters in English
        max_tokens = max(1, max_length // 4)
        min_tokens = max(1, min_length // 4)

        # Ensure text is within model's max token limit
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(self.device)

        # Set generation parameters
        generation_params = {
            "max_length": max_tokens,
            "min_length": min_tokens,
            "num_beams": 4,
            "length_penalty": 2.0,
            "early_stopping": True,
        }

        # Handle sampling and temperature
        if do_sample:
            try:
                # First attempt: try with the requested temperature
                generation_params["do_sample"] = True
                generation_params["temperature"] = temperature
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    **generation_params
                )
            except Exception as e:
                # If that fails, try with default temperature (1.0)
                print(f"Error with temperature {temperature}, falling back to default: {str(e)}")
                generation_params["temperature"] = 1.0
                try:
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        **generation_params
                    )
                except Exception:
                    # If sampling still fails, fall back to beam search without sampling
                    print("Sampling failed, falling back to beam search")
                    generation_params.pop("do_sample", None)
                    generation_params.pop("temperature", None)
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        **generation_params
                    )
        else:
            # Standard beam search without sampling
            summary_ids = self.model.generate(
                inputs["input_ids"],
                **generation_params
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # If the summary is still too long, truncate it
        if len(summary) > max_length:
            # Try to truncate at a sentence boundary
            sentences = summary.split('. ')
            truncated_summary = ''
            for sentence in sentences:
                if len(truncated_summary) + len(sentence) + 2 <= max_length:  # +2 for '. '
                    truncated_summary += sentence + '. '
                else:
                    break

            # If we couldn't even fit one sentence, just truncate at max_length
            if not truncated_summary:
                truncated_summary = summary[:max_length]

            summary = truncated_summary.strip()

        return summary
