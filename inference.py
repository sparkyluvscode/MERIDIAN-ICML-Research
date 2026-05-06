"""
Meridian: ESL-Aware AI Text Detector
Scores an essay and returns a probability that it is AI-generated.

Usage:
    python inference.py --text "Your essay text here"
    python inference.py --file essay.txt

Requirements:
    pip install torch transformers scikit-learn numpy
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
import os

# ── Model Architecture (must match training) ─────────────────────────────────

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            vocab_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden


# ── Character vocabulary (must match training) ────────────────────────────────

def build_vocab():
    chars = (
        list("abcdefghijklmnopqrstuvwxyz") +
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
        list("0123456789") +
        list(" \t\n\r.,!?;:'\"()-[]{}@#$%^&*_+=/<>\\|`~") +
        [chr(i) for i in range(128, 256)]
    )
    char_to_index = {c: i for i, c in enumerate(chars)}
    return char_to_index, len(chars)


# ── Perplexity functions ──────────────────────────────────────────────────────

def compute_esl_perplexity(text, model, char_to_index, vocab_size,
                           seq_len=60, device="cpu"):
    """Compute character-level LSTM perplexity (P_esl)."""
    model.eval()
    indices = [char_to_index.get(c, 0) for c in text]
    if len(indices) < seq_len + 1:
        return float("nan")

    total_loss, n_chunks = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for start in range(0, len(indices) - seq_len, seq_len):
            chunk = indices[start: start + seq_len + 1]
            x = torch.zeros(1, seq_len, vocab_size)
            for t, idx in enumerate(chunk[:-1]):
                x[0, t, idx] = 1.0
            x = x.to(device)
            y = torch.tensor(chunk[1:], dtype=torch.long).to(device)
            out, _ = model(x)
            loss = criterion(out[0], y)
            total_loss += loss.item()
            n_chunks += 1

    return float(np.exp(total_loss / n_chunks)) if n_chunks > 0 else float("nan")


def compute_native_perplexity(text, gpt_model, tokenizer, device="cpu",
                               max_length=512):
    """Compute DistilGPT-2 sub-word perplexity (P_native)."""
    gpt_model.eval()
    enc = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    if enc["input_ids"].shape[1] < 3:
        return float("nan")
    with torch.no_grad():
        loss = gpt_model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss).item())


# ── Logistic classifier (Phase 2 frozen weights) ─────────────────────────────
# Coefficients from Appendix C of the paper

W_ESL    = -1.8166
W_NATIVE = -0.2831
INTERCEPT = 12.0569


def classify(p_esl, p_native, threshold=0.50):
    """
    Returns probability of AI-generated and binary prediction.
    Score > threshold → AI-generated.
    """
    logit = W_ESL * p_esl + W_NATIVE * p_native + INTERCEPT
    prob_ai = 1.0 / (1.0 + np.exp(-logit))
    return prob_ai, prob_ai > threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def score_essay(text, lstm_weights_path="SUPER_esl_lstm_weights.pth",
                threshold=0.50, device=None):
    """
    Score a single essay text.

    Returns:
        dict with keys: p_esl, p_native, prob_ai, prediction
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    char_to_index, vocab_size = build_vocab()

    # Load LSTM
    lstm = CharRNN(vocab_size).to(device)
    state = torch.load(lstm_weights_path, map_location=device)
    lstm.load_state_dict(state)

    # Load DistilGPT-2
    tokenizer  = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt_model  = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    p_esl    = compute_esl_perplexity(text, lstm, char_to_index, vocab_size, device=device)
    p_native = compute_native_perplexity(text, gpt_model, tokenizer, device=device)
    prob_ai, is_ai = classify(p_esl, p_native, threshold)

    return {
        "p_esl":      round(p_esl, 3),
        "p_native":   round(p_native, 3),
        "prob_ai":    round(float(prob_ai), 4),
        "prediction": "AI-generated" if is_ai else "Human (ESL)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meridian ESL-aware AI text detector")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Essay text as a string")
    group.add_argument("--file", type=str, help="Path to a .txt file")
    parser.add_argument("--weights",    default="SUPER_esl_lstm_weights.pth",
                        help="Path to LSTM weights file")
    parser.add_argument("--threshold",  type=float, default=0.50,
                        help="Classification threshold (default 0.50; use 0.80 for lower FPR)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    print(f"\nScoring essay ({len(text)} characters)...")
    result = score_essay(text, lstm_weights_path=args.weights,
                         threshold=args.threshold)

    print(f"\n{'='*45}")
    print(f"  Meridian Result")
    print(f"{'='*45}")
    print(f"  P_esl (ESL-Expert LSTM):     {result['p_esl']}")
    print(f"  P_native (DistilGPT-2):      {result['p_native']}")
    print(f"  P(AI-generated):             {result['prob_ai']:.1%}")
    print(f"  Prediction:                  {result['prediction']}")
    print(f"{'='*45}")
    print(f"\nNote: Meridian is a screening tool, not a verdict system.")
    print(f"All flagged essays should receive human review.")
