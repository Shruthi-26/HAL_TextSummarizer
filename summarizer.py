import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re
import os

# -----------------------------
# FIREWALL-SAFE NLTK SETUP
# -----------------------------
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")

try:
    nltk.data.path.append(nltk_data_path)
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)


class LocalAISummarizer:
    def __init__(self):
        """Offline summarizer using locally cached BART model"""
        print("Loading LOCAL Hugging Face BART model...")

        model_path = "facebook/bart-large-cnn"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=False  # Set True after first download
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=False
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device} (1.3GB cached locally)")

    # -----------------------------
    # Text Cleaning
    # -----------------------------
    def clean_text(self, text):
        """Clean input text before sending to model"""
        text = re.sub(r"\s+", " ", text).strip()

        # BART max input is 1024 tokens (roughly 1024 words safe cut)
        if len(text) > 1024:
            text = text[:1024] + "..."

        return text

    # -----------------------------
    # Summarization
    # -----------------------------
    def summarize(self, text, summary_type="short"):
        """
        Generate summary in 3 lengths:
        - short (80 tokens)
        - medium (150 tokens)
        - large (250 tokens)
        """

        text = self.clean_text(text)

        max_len_map = {
            "short": 80,
            "medium": 150,
            "large": 250
        }

        max_len = max_len_map.get(summary_type, 80)
        min_len = max_len // 3

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return summary.strip()


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    summarizer_instance = LocalAISummarizer()
