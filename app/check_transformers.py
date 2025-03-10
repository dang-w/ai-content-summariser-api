# Simple script to check if transformers is installed correctly
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("Transformers library is installed correctly!")

    # Try loading a small model to verify functionality
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    print("Model loaded successfully!")

    # Test a simple summarization
    text = "This is a test sentence to check if the summarisation model works correctly. It should be able to process this text and generate a summary."
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Test summary: {summary}")

except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please try reinstalling with: pip install transformers torch")
except Exception as e:
    print(f"Error during model loading or inference: {e}")
