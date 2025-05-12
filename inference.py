# inference.py
import argparse
import torch
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTIONS
from models.generator import Generator
from utils import sample_text_autoregressive, EMOTION_LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion",     required=True, help="One of: " + ", ".join(EMOTION_LABELS))
    parser.add_argument("--length",      type=int,   default=30, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k",       type=int,   default=50,  help="Top-k filtering")
    parser.add_argument("--model-path",  default="generator.pt", help="Path to generator weights")
    args = parser.parse_args()

    if args.emotion not in EMOTION_LABELS:
        print(f"Error: emotion must be one of {EMOTION_LABELS}")
        return

    emo_idx = EMOTION_LABELS.index(args.emotion)
    G = Generator(30522, EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTIONS).to(DEVICE)
    G.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    G.eval()

    output = sample_text_autoregressive(
        G, emo_idx,
        max_len=args.length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(f"\nEmotion: {args.emotion}\nGenerated Text: {output}\n")

if __name__ == "__main__":
    main()
