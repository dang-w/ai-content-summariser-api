from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@lru_cache(maxsize=2)
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device
