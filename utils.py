# utils.py
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from config import DEVICE

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Emotion label names
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def decode_text(tok_ids):
    return tokenizer.decode(tok_ids, skip_special_tokens=True)

def get_emotion_vector(idx):
    vec = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    vec[idx] = 1.0
    return torch.tensor(vec, device=DEVICE, dtype=torch.float32)

def sample_text_autoregressive(generator, emo_idx, max_len=30, temperature=1.0, top_k=50):
    """
    Autoregressive sampling with top-k + temperature.
    """
    generator.eval()
    emo_vec = get_emotion_vector(emo_idx).unsqueeze(0)         # (1, emo_dim)
    tokens  = [tokenizer.cls_token_id]                         # start with [CLS]

    for _ in range(max_len):
        input_ids = torch.tensor(tokens, device=DEVICE).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            logits = generator(input_ids, emo_vec)                    # (1, T, V)
        next_logits = logits[0, -1, :] / temperature                  # (V,)

        # Top-k filtering
        topk = torch.topk(next_logits, top_k)
        topk_indices = topk.indices
        topk_logits  = topk.values

        probs = F.softmax(topk_logits, dim=-1)
        next_token = topk_indices[torch.multinomial(probs, 1).item()].item()

        if next_token == tokenizer.sep_token_id:  # stop on [SEP]
            break
        tokens.append(next_token)

    return tokenizer.decode(tokens, skip_special_tokens=True)
