# -*- coding: utf-8 -*-
"""
Shared Detoxify scoring for the toxicity re-analysis.

Detoxify 'unbiased' is a RoBERTa-base model trained on the Jigsaw Unintended Bias
(Civil Comments) data - the same rater framework that underlies Perspective - with
an identity-bias mitigation objective. It runs locally, so unlike Perspective it
cannot be retired and every score is reproducible from the released weights.

Scoring is batched by length (minimal padding) under torch.no_grad, in fp32. fp16
autocast is available but off by default: on the GTX 1650 used here it ran ~5x
slower than fp32 (no fast half-precision path), and it moves scores by <0.001.
"""
import numpy as np
import torch
from detoxify import Detoxify

MODEL = "unbiased"
MAX_LEN = 512          # Detoxify default; RoBERTa maximum
# Detoxify name -> Perspective attribute it corresponds to
PAIRS = {
    "toxicity": "TOXICITY",
    "severe_toxicity": "SEVERE_TOXICITY",
    "identity_attack": "IDENTITY_ATTACK",
    "insult": "INSULT",
    "threat": "THREAT",
    "obscene": "PROFANITY",
}


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = Detoxify(MODEL, device=device)
    m.model.eval()
    return m


@torch.no_grad()
def score(m, texts, batch=64, fp16=False, progress=None):
    """Return {class_name: np.array} in the order of `texts`."""
    texts = ["" if t is None else str(t) for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), len(m.class_names)), dtype=np.float32)
    use_amp = fp16 and m.model.device.type == "cuda"
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = m.tokenizer([texts[i] for i in idx], return_tensors="pt",
                          truncation=True, max_length=MAX_LEN,
                          padding=True).to(m.model.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = m.model(**enc)[0]
        # the unbiased model also emits identity-subgroup heads after the named
        # classes; Detoxify.predict reads only the first len(class_names)
        out[idx] = torch.sigmoid(logits[:, :out.shape[1]].float()).cpu().numpy()
        if progress:
            progress(len(idx))
    return {c: out[:, j] for j, c in enumerate(m.class_names)}
