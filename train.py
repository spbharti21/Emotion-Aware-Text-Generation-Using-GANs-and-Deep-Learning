# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTIONS, LEARNING_RATE, EPOCHS
from data.preprocess import get_dataloaders
from models.generator import Generator
from models.discriminator import Discriminator
from models.emotion_classifier import EmotionClassifier

def train():
    print("Initializing models and data…")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id

    loader = get_dataloaders()
    vocab_size = tokenizer.vocab_size

    G = Generator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTIONS).to(DEVICE)
    D = Discriminator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTIONS).to(DEVICE)
    E = EmotionClassifier().to(DEVICE)
    E.eval()

    g_opt = optim.Adam(G.parameters(), lr=LEARNING_RATE)
    d_opt = optim.Adam(D.parameters(), lr=LEARNING_RATE)
    bce   = nn.BCELoss()
    ce    = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            emo_lbls  = batch["labels"].to(DEVICE)

            # — Discriminator update —
            d_opt.zero_grad()
            real_preds = D(input_ids, emo_lbls)
            real_tgt   = torch.ones_like(real_preds)

            with torch.no_grad():
                fake_logits = G(input_ids, emo_lbls)
                fake_ids    = fake_logits.argmax(dim=-1)
            fake_preds = D(fake_ids, emo_lbls)
            fake_tgt   = torch.zeros_like(fake_preds)

            d_loss = bce(real_preds, real_tgt) + bce(fake_preds, fake_tgt)
            d_loss.backward()
            d_opt.step()

            # — Generator update —
            g_opt.zero_grad()
            gen_logits = G(input_ids, emo_lbls)
            gen_ids    = gen_logits.argmax(dim=-1)

            # adversarial loss
            adv_preds = D(gen_ids, emo_lbls)
            adv_tgt   = torch.ones_like(adv_preds)
            adv_loss  = bce(adv_preds, adv_tgt)

            # reconstruction loss (teacher forcing)
            logits_shifted = gen_logits[:, :-1, :].reshape(-1, vocab_size)
            targets_shifted = input_ids[:, 1:].reshape(-1)
            recon_loss = ce(logits_shifted, targets_shifted)

            g_loss = adv_loss + recon_loss
            g_loss.backward()
            g_opt.step()

        print(f"  D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    torch.save(G.state_dict(), "generator.pt")
    print("Done. Saved generator.pt")

if __name__ == "__main__":
    train()
