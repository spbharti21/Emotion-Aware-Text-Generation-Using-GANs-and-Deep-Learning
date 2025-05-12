# data/preprocess.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from config import MAX_LEN, BATCH_SIZE

def get_dataloaders():
    print("Loading full GoEmotions ‘raw’ config…")
    ds = load_dataset("go_emotions", "raw", split="train")  # no .select()

    emotion_columns = [
        "admiration","amusement","anger","annoyance","approval","caring","confusion",
        "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
        "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
        "pride","realization","relief","remorse","sadness","surprise","neutral"
    ]

    def labelify(ex):
        return {"labels": [int(ex[e] > 0) for e in emotion_columns]}

    ds = ds.map(labelify, batched=False)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_fn(ex):
        return tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
    ds = ds.map(tokenize_fn, batched=True)

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(f"Dataset ready: {len(ds)} samples.")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
