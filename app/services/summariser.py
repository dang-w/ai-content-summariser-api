import numpy as np  # Import NumPy first
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os
import re

class SummariserService:
    def __init__(self):
        # Status tracking
        self.model_loading_status = {
            "is_loading": False,
            "step": "",
            "progress": 0
        }

        # Consider these alternative models
        model_options = {
            "general": "facebook/bart-large-cnn",
            "news": "facebook/bart-large-xsum",
            "long_form": "google/pegasus-large",
            "literary": "t5-large"
        }

        # Choose the most appropriate model
        model_name = model_options["literary"]  # Better for literary text

        # Update loading status
        self.model_loading_status["is_loading"] = True
        self.model_loading_status["step"] = "Initializing tokenizer"

        # Ensure cache directory exists and is writable
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )

            self.model_loading_status["step"] = "Loading model"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                force_download=False,
                local_files_only=False
            )

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        except Exception as e:
            # Fallback to a smaller model if the main one fails
            print(f"Error loading model {model_name}: {str(e)}")
            print("Falling back to smaller model...")

            fallback_model = "sshleifer/distilbart-cnn-6-6"  # Much smaller model

            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=cache_dir,
                local_files_only=False
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                fallback_model,
                cache_dir=cache_dir,
                force_download=False,
                local_files_only=False
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            # Update model name for metadata
            model_name = fallback_model

        self.model_loading_status["is_loading"] = False
        self.model_loading_status["progress"] = 100

        # Store the actual model name used
        self.model_name = model_name

        # Track current processing job
        self.current_job = {
            "in_progress": False,
            "start_time": None,
            "input_word_count": 0,
            "estimated_time": 0,
            "stage": "",
            "progress": 0
        }

    def clean_summary(self, summary):
        """Clean and format the summary text"""
        # Remove any leading punctuation or spaces
        summary = re.sub(r'^[,.\s]+', '', summary)

        # Ensure the first letter is capitalized
        if summary and len(summary) > 0:
            summary = summary[0].upper() + summary[1:]

        # Ensure proper ending punctuation
        if summary and not any(summary.endswith(end) for end in ['.', '!', '?']):
            last_sentence_end = max(
                summary.rfind('.'),
                summary.rfind('!'),
                summary.rfind('?')
            )
            if last_sentence_end > 0:
                summary = summary[:last_sentence_end + 1]
            else:
                summary = summary + '.'

        return summary

    def get_status(self):
        """Return the current status of the summarizer service"""
        status = {
            "model_loading": self.model_loading_status,
            "device": self.device,
            "current_job": self.current_job
        }

        # Update estimated time remaining if job in progress
        if self.current_job["in_progress"] and self.current_job["start_time"]:
            elapsed = time.time() - self.current_job["start_time"]
            estimated = self.current_job["estimated_time"]
            remaining = max(0, estimated - elapsed)
            status["current_job"]["time_remaining"] = round(remaining, 1)

            # Update progress based on time
            if estimated > 0:
                progress = min(95, (elapsed / estimated) * 100)
                status["current_job"]["progress"] = round(progress, 0)

        return status

    def summarise(self, text, max_length=250, min_length=100, do_sample=True, temperature=1.2):
        """
        Summarise the given text using the loaded model.

        Args:
            text (str): The text to summarise
            max_length (int): Maximum length of the summary in characters
            min_length (int): Minimum length of the summary in characters
            do_sample (bool): Whether to use sampling for generation
            temperature (float): Sampling temperature (higher = more random)

        Returns:
            dict: The generated summary and processing metadata
        """
        # Reset and start job tracking
        self.current_job = {
            "in_progress": True,
            "start_time": time.time(),
            "input_word_count": len(text.split()),
            "estimated_time": max(1, min(30, len(text.split()) / 500)),  # Rough estimate
            "stage": "Tokenizing input text",
            "progress": 5
        }

        result = {
            "summary": "",
            "metadata": {
                "input_word_count": self.current_job["input_word_count"],
                "estimated_time_seconds": self.current_job["estimated_time"],
                "model_used": self.model_name,
                "processing_device": self.device
            }
        }

        try:
            # Tokenization step
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            input_ids = inputs.input_ids.to(self.device)

            # Update metadata with token info
            result["metadata"]["input_token_count"] = len(input_ids[0])
            result["metadata"]["truncated"] = len(input_ids[0]) == 1024

            # Update job status
            self.current_job["stage"] = "Generating summary"
            self.current_job["progress"] = 30

            # Enhanced generation parameters
            summary_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
            )

            # Update job status
            self.current_job["stage"] = "Post-processing summary"
            self.current_job["progress"] = 90

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Clean and format the summary
            summary = self.clean_summary(summary)

            result["summary"] = summary
            result["metadata"]["output_word_count"] = len(summary.split())
            result["metadata"]["compression_ratio"] = round(len(summary.split()) / self.current_job["input_word_count"] * 100, 1)

        except Exception as e:
            # Handle errors gracefully
            print(f"Error during summarization: {str(e)}")
            result["summary"] = "An error occurred during summarization. Please try again with a shorter text or different parameters."
            result["error"] = str(e)
        finally:
            # Complete job
            self.current_job["in_progress"] = False
            self.current_job["stage"] = "Complete"
            self.current_job["progress"] = 100

        return result
