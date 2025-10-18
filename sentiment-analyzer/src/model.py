import torch
from torch import nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # BERT pooler
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)
