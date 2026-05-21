# Encoder-Decoder with Attention — From Scratch

## Overview

This module teaches the **Sequence-to-Sequence (Seq2Seq)** architecture with **Bahdanau Attention**, the key stepping stone between Word2Vec and Transformers.

## Learning Path

```
Word2Vec (✅ done) → Encoder-Decoder + Attention (📍 you are here) → Transformers (next)
```

## Files

| File | Description |
|------|-------------|
| `encoder_decoder_scratch.py` | Complete implementation — GRU, Attention, Seq2Seq model |
| `tutorial.ipynb` | Interactive tutorial notebook — concepts, math, and hands-on code |

## Key Concepts

1. **Encoder** — Reads the input sequence word by word using a GRU, producing a hidden state at each position
2. **Decoder** — Generates the output sequence one token at a time, using attention to look back at the encoder
3. **Attention** — Instead of compressing the entire input into one vector, the decoder learns to *focus* on the most relevant parts of the input at each step
4. **Teacher Forcing** — During training, feed ground-truth tokens to the decoder (instead of its own predictions)
5. **Greedy Decoding** — At inference, pick the highest-probability word at each step

## Quick Start

```python
# Open tutorial.ipynb and run cells in order
# Or run from Python:
from encoder_decoder_attention.encoder_decoder_scratch import Seq2SeqAttention, Vocabulary

# Build vocabs, create model, train, translate — all from scratch!
```

## References

- Sutskever et al., 2014 — *Sequence to Sequence Learning with Neural Networks*
- Bahdanau et al., 2015 — *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al., 2015 — *Effective Approaches to Attention-based Neural Machine Translation*

