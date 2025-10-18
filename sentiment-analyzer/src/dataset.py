import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

LABEL_MAP = {"NEG": 0, "NEU": 1, "POS": 2}

class SentimentDataset(Dataset):
    def __init__(self, excel_path, tokenizer_name, max_len):
        # Load Excel file
        df = pd.read_excel(excel_path)
        self.texts = df["Tweet"].tolist()  
        self.labels = [LABEL_MAP[label] for label in df["Final annotation"]]  # Map labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
