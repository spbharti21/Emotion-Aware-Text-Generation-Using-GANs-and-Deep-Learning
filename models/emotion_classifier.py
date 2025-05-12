# models/emotion_classifier.py
import torch.nn as nn
from transformers import BertForSequenceClassification
from config import NUM_EMOTIONS

class EmotionClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_EMOTIONS
        )

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
