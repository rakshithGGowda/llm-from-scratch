"""
================================================================================
  TRANSFORMER FROM SCRATCH - A Complete Step-by-Step Tutorial
================================================================================

  Learning Path:
      Word2Vec  ->  Encoder-Decoder + Attention  ->  TRANSFORMER (you are here)

  This file contains EVERYTHING:
      - Intuition and explanations (as comments/prints)
      - Full implementation from scratch (NumPy only)
      - Training on real data
      - End-to-end working example

  Reference: "Attention Is All You Need" (Vaswani et al., 2017)

  Architecture Overview:

      Input                                    Output (shifted right)
        |                                          |
    [Embedding + Positional Encoding]     [Embedding + Positional Encoding]
        |                                          |
    [Encoder Block] x N                    [Decoder Block] x N
        |    - Self-Attention                  |    - Masked Self-Attention
        |    - Feed-Forward                    |    - Cross-Attention (to encoder)
        |    - Layer Norm + Residual           |    - Feed-Forward
        |                                      |    - Layer Norm + Residual
        |                                      |
        +----------> cross-attention <---------+
                                               |
                                        [Linear + Softmax]
                                               |
                                        Predicted next token

  WHY TRANSFORMERS?
  =================
  In the previous tutorial (Encoder-Decoder with Attention), we used an RNN
  (GRU) to process sequences. RNNs have two big problems:

  1. SEQUENTIAL BOTTLENECK: RNNs process words one at a time (h1 -> h2 -> h3).
     You can't compute h3 until h2 is done. This makes training SLOW.

  2. LONG-RANGE DEPENDENCIES: Even with attention, the RNN hidden state is a
     bottleneck. Information from the start of a long sentence gets diluted.

  The Transformer solves BOTH problems by:
  - Processing ALL positions in PARALLEL (no recurrence!)
  - Using SELF-ATTENTION so every word can directly attend to every other word
  - Adding POSITIONAL ENCODING to preserve word order (since there's no RNN)

================================================================================
"""

import numpy as np
from collections import Counter
import time

np.random.seed(42)


# ==============================================================================
# SECTION 1: ACTIVATION FUNCTIONS & UTILITIES
# ==============================================================================
# These are the same mathematical building blocks you've seen before.
# We need: softmax (for attention weights), relu (for feed-forward layers),
# and layer normalization (for training stability).
# ==============================================================================

def softmax(x, axis=-1):
    """
    Softmax: turns raw scores into probabilities that sum to 1.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Used in:
    - Attention weights (which words to focus on)
    - Output layer (which word to predict)

    The max-subtraction trick prevents numerical overflow:
    exp(1000) = inf, but exp(1000 - 1000) = exp(0) = 1
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x):
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)

    Simple but effective non-linearity for feed-forward layers.
    Intuition: "only pass through positive signals"
    """
    return np.maximum(0, x)


def layer_norm(x, gamma, beta, eps=1e-6):
    """
    Layer Normalization: normalizes each sample independently.

    WHY? Deep networks suffer from "internal covariate shift" - the
    distribution of each layer's inputs changes during training, making
    learning unstable. Layer norm fixes this by normalizing to mean=0, var=1,
    then scaling/shifting with learnable parameters gamma and beta.

    Formula:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    Args:
        x:     input array, shape (..., d_model)
        gamma: scale parameter, shape (d_model,)
        beta:  shift parameter, shape (d_model,)
        eps:   small constant to prevent division by zero

    This is different from Batch Norm:
    - Batch Norm normalizes across the BATCH dimension
    - Layer Norm normalizes across the FEATURE dimension
    - Layer Norm works better for sequences (variable length)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


print("=" * 70)
print("TRANSFORMER FROM SCRATCH - Step by Step Tutorial")
print("=" * 70)
print("\nSection 1: Utility functions loaded.")
print("  - softmax: raw scores -> probabilities")
print("  - relu: non-linear activation for feed-forward layers")
print("  - layer_norm: stabilizes training by normalizing features")


# ==============================================================================
# SECTION 2: VOCABULARY (same concept as Encoder-Decoder tutorial)
# ==============================================================================

class Vocabulary:
    """
    Maps words <-> integer indices.

    Special tokens:
        <pad> = 0   Padding (to make all sequences same length in a batch)
        <sos> = 1   Start-of-sequence (first decoder input)
        <eos> = 2   End-of-sequence (signals decoder to stop)
        <unk> = 3   Unknown word (fallback for out-of-vocabulary words)

    This is identical to what we used in the Encoder-Decoder tutorial.
    """
    SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.n_words = 0
        for token in self.SPECIAL_TOKENS:
            self._add(token)

    def _add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def build(self, sentences, min_count=1):
        counts = Counter()
        for s in sentences:
            for w in s:
                counts[w] += 1
        for w, c in counts.items():
            if c >= min_count:
                self._add(w)
        print(f"  Vocabulary built: {self.n_words} tokens "
              f"(including {len(self.SPECIAL_TOKENS)} special)")

    def encode(self, words):
        unk = self.word2idx["<unk>"]
        return [self.word2idx.get(w, unk) for w in words]

    def decode(self, indices):
        return [self.idx2word.get(i, "<unk>") for i in indices]


# ==============================================================================
# SECTION 3: POSITIONAL ENCODING
# ==============================================================================
#
# THE PROBLEM:
# Unlike RNNs, Transformers process all positions in PARALLEL.
# This means the model has NO IDEA about word order!
# "I love cats" and "cats love I" would look IDENTICAL.
#
# THE SOLUTION:
# Add a unique "position signal" to each word's embedding.
# This signal tells the model WHERE in the sequence each word is.
#
# HOW:
# We use sine and cosine functions of different frequencies:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# WHY SINE/COSINE?
# 1. Each position gets a unique pattern
# 2. The model can easily learn to attend to "relative" positions
#    because PE(pos+k) can be represented as a linear function of PE(pos)
# 3. It generalizes to sequence lengths not seen during training
# ==============================================================================

def positional_encoding(max_len, d_model):
    """
    Compute positional encoding matrix.

    Args:
        max_len: maximum sequence length
        d_model: embedding dimension

    Returns:
        PE matrix of shape (max_len, d_model)

    Each row is the positional encoding for that position.
    Even indices use sin, odd indices use cos.
    """
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            denominator = 10000 ** (i / d_model)
            PE[pos, i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denominator)
    return PE


print("\n" + "-" * 70)
print("Section 2-3: Vocabulary and Positional Encoding loaded.")
print("\nLet's visualize positional encoding:")

pe = positional_encoding(max_len=10, d_model=16)
print(f"  Shape: {pe.shape}  (10 positions x 16 dimensions)")
print(f"  Position 0: [{', '.join(f'{v:.3f}' for v in pe[0, :6])} ...]")
print(f"  Position 1: [{', '.join(f'{v:.3f}' for v in pe[1, :6])} ...]")
print(f"  Position 9: [{', '.join(f'{v:.3f}' for v in pe[9, :6])} ...]")
print("  (Each position has a unique pattern!)")


# ==============================================================================
# SECTION 4: SCALED DOT-PRODUCT ATTENTION
# ==============================================================================
#
# This is the HEART of the Transformer. Everything else is built around this.
#
# INTUITION:
# Imagine you're reading a sentence and trying to understand one word.
# You "look at" (attend to) other words in the sentence to gather context.
# For "The cat sat on the mat", to understand "sat", you'd look at:
#   - "cat" (WHO sat?) -> high attention
#   - "mat" (WHERE?) -> medium attention
#   - "the" (not very informative) -> low attention
#
# THE MECHANISM:
# For each word, we create three vectors:
#   Q (Query):  "What am I looking for?"
#   K (Key):    "What do I contain?"
#   V (Value):  "What information do I provide?"
#
# Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
#
# Step by step:
#   1. Q * K^T  -> How well does each query match each key? (relevance scores)
#   2. / sqrt(d_k) -> Scale down to prevent softmax saturation
#   3. softmax -> Convert scores to probabilities (attention weights)
#   4. * V -> Weighted sum of values (the actual output)
#
# WHY SCALE BY sqrt(d_k)?
# When d_k is large, dot products grow large, pushing softmax into regions
# with tiny gradients (near 0 or 1). Dividing by sqrt(d_k) keeps the
# variance of the dot products at 1, regardless of dimension.
# ==============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.

    Args:
        Q: queries, shape (..., seq_len_q, d_k)
        K: keys,    shape (..., seq_len_k, d_k)
        V: values,  shape (..., seq_len_k, d_v)
        mask: optional mask, shape broadcastable to (..., seq_len_q, seq_len_k)
              where True/1 means "BLOCK this position" (set to -inf before softmax)

    Returns:
        output: weighted sum of values, shape (..., seq_len_q, d_v)
        attention_weights: shape (..., seq_len_q, seq_len_k)
    """
    d_k = K.shape[-1]

    # Step 1: Compute attention scores (how relevant is each key to each query)
    scores = Q @ K.swapaxes(-2, -1)   # (..., seq_len_q, seq_len_k)

    # Step 2: Scale by sqrt(d_k) to stabilize gradients
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (if any) - set masked positions to -infinity
    # so they become 0 after softmax
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Step 4: Softmax to get attention weights (probabilities)
    weights = softmax(scores, axis=-1)

    # Step 5: Weighted sum of values
    output = weights @ V   # (..., seq_len_q, d_v)

    return output, weights


print("\n" + "-" * 70)
print("Section 4: Scaled Dot-Product Attention loaded.")
print("\nQuick demo:")

# Demo with 3 words, embedding dim = 4
demo_Q = np.random.randn(3, 4) * 0.5
demo_K = np.random.randn(3, 4) * 0.5
demo_V = np.random.randn(3, 4) * 0.5

demo_out, demo_weights = scaled_dot_product_attention(demo_Q, demo_K, demo_V)
print(f"  Q shape: {demo_Q.shape}  (3 words, dim 4)")
print(f"  Attention weights:\n{np.array2string(demo_weights, precision=3, suppress_small=True)}")
print(f"  Each row sums to: {demo_weights.sum(axis=-1)}")
print("  (Each word now has a context-aware representation!)")


# ==============================================================================
# SECTION 5: MULTI-HEAD ATTENTION
# ==============================================================================
#
# WHY MULTIPLE HEADS?
# A single attention can only focus on ONE type of relationship at a time.
# But language has MANY types of relationships simultaneously:
#   - Syntactic: subject-verb agreement ("The cats ARE")
#   - Semantic: meaning relationships ("doctor" attends to "patient")
#   - Positional: nearby words ("the" -> next word)
#
# Multi-head attention runs MULTIPLE attention operations in parallel,
# each with its own Q/K/V projections, so each "head" can learn a
# different type of relationship.
#
# Formula:
#   head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
#   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
#
# If d_model=64 and num_heads=4:
#   Each head works with d_k = d_model / num_heads = 16 dimensions
#   All heads together still produce d_model=64 dimensions
#   -> Same cost as single-head attention, but more expressive!
# ==============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention layer.

    Splits the embedding into `num_heads` heads, runs scaled dot-product
    attention on each head independently, then concatenates and projects.

    Args:
        d_model:   total embedding dimension (e.g., 64)
        num_heads: number of attention heads (e.g., 4)
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # dimension per head

        # Learnable projection matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def _split_heads(self, x):
        """
        Reshape (seq_len, d_model) -> (num_heads, seq_len, d_k)

        This splits the embedding dimension into separate heads.
        Each head gets its own d_k-dimensional "view" of the data.
        """
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)  # (seq, heads, d_k)
        return x.transpose(1, 0, 2)  # (heads, seq, d_k)

    def _combine_heads(self, x):
        """
        Reshape (num_heads, seq_len, d_k) -> (seq_len, d_model)

        Reverse of _split_heads: concatenate all heads back together.
        """
        x = x.transpose(1, 0, 2)  # (seq, heads, d_k)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)  # (seq, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            Q: query input,  shape (seq_len_q, d_model)
            K: key input,    shape (seq_len_k, d_model)
            V: value input,  shape (seq_len_k, d_model)
            mask: optional,  shape (1, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)

        Returns:
            output: shape (seq_len_q, d_model)
            attention_weights: shape (num_heads, seq_len_q, seq_len_k)
        """
        # 1. Project Q, K, V through learned weight matrices
        Q_proj = Q @ self.W_Q    # (seq_q, d_model)
        K_proj = K @ self.W_K    # (seq_k, d_model)
        V_proj = V @ self.W_V    # (seq_k, d_model)

        # 2. Split into multiple heads
        Q_heads = self._split_heads(Q_proj)   # (heads, seq_q, d_k)
        K_heads = self._split_heads(K_proj)   # (heads, seq_k, d_k)
        V_heads = self._split_heads(V_proj)   # (heads, seq_k, d_k)

        # 3. Apply scaled dot-product attention to each head
        if mask is not None and mask.ndim == 2:
            mask = mask[np.newaxis, :, :]  # (1, seq_q, seq_k) -> broadcast over heads

        attn_output, attn_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask
        )
        # attn_output: (heads, seq_q, d_k)
        # attn_weights: (heads, seq_q, seq_k)

        # 4. Concatenate heads and project back to d_model
        concat = self._combine_heads(attn_output)  # (seq_q, d_model)
        output = concat @ self.W_O                  # (seq_q, d_model)

        return output, attn_weights


print("\n" + "-" * 70)
print("Section 5: Multi-Head Attention loaded.")
print("\nDemo with 4 heads:")

mha = MultiHeadAttention(d_model=16, num_heads=4)
x_demo = np.random.randn(5, 16) * 0.1   # 5 words, 16 dims
out_demo, w_demo = mha.forward(x_demo, x_demo, x_demo)
print(f"  Input shape:  {x_demo.shape}  (5 words, 16 dims)")
print(f"  Output shape: {out_demo.shape}  (same! attention is shape-preserving)")
print(f"  Weights shape: {w_demo.shape}  (4 heads, each 5x5 attention matrix)")
print("  Each head learns a DIFFERENT attention pattern!")


# ==============================================================================
# SECTION 6: FEED-FORWARD NETWORK (FFN)
# ==============================================================================
#
# After attention, each position passes through a simple 2-layer neural network.
# This is applied INDEPENDENTLY to each position (like a 1x1 convolution).
#
# WHY?
# Attention captures RELATIONSHIPS between words, but the FFN adds
# NON-LINEAR TRANSFORMATIONS at each position. It's where the model
# learns complex features from the attended representations.
#
# Formula:
#   FFN(x) = max(0, x * W1 + b1) * W2 + b2
#            ^^^^^
#            ReLU activation
#
# The hidden dimension (d_ff) is typically 4x the model dimension.
# E.g., d_model=64 -> d_ff=256. This "expand then compress" pattern
# lets the network learn richer representations in the larger space.
# ==============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    Two linear layers with ReLU in between.
    Applied to each position independently.

    Args:
        d_model: input/output dimension
        d_ff:    hidden dimension (typically 4 * d_model)
    """

    def __init__(self, d_model, d_ff):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        Args:
            x: shape (seq_len, d_model)
        Returns:
            shape (seq_len, d_model)
        """
        hidden = relu(x @ self.W1 + self.b1)   # (seq, d_ff) - expand
        output = hidden @ self.W2 + self.b2     # (seq, d_model) - compress back
        return output


print("\n" + "-" * 70)
print("Section 6: Feed-Forward Network loaded.")
ff_demo = FeedForward(d_model=16, d_ff=64)
ff_out = ff_demo.forward(x_demo)
print(f"  FFN input:  {x_demo.shape}")
print(f"  FFN output: {ff_out.shape}  (same shape - position-wise transform)")


# ==============================================================================
# SECTION 7: ENCODER LAYER
# ==============================================================================
#
# One encoder layer = Self-Attention + Feed-Forward, each wrapped with:
#   1. Residual connection:  output = sublayer(x) + x
#   2. Layer normalization:  output = LayerNorm(output)
#
# WHY RESIDUAL CONNECTIONS?
# As networks get deeper, gradients can vanish or explode.
# Residual connections create "shortcuts" that let gradients flow
# directly through the network. They also make it easy for the
# network to learn the identity function (just pass through).
#
# The full flow for one encoder layer:
#
#   x ----+
#         |
#   [Self-Attention]
#         |
#   + <---+  (residual: add the original x)
#         |
#   [Layer Norm]
#         |
#   ------+
#         |
#   [Feed-Forward]
#         |
#   + <---+  (residual)
#         |
#   [Layer Norm]
#         |
#   output
# ==============================================================================

class EncoderLayer:
    """
    One Transformer Encoder layer.

    Components:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (x2)
    - Residual connections (x2)

    Args:
        d_model:   embedding dimension
        num_heads: number of attention heads
        d_ff:      feed-forward hidden dimension
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer norm parameters (learnable scale and shift)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: input, shape (seq_len, d_model)
            mask: optional padding mask

        Returns:
            output: shape (seq_len, d_model)
            attn_weights: shape (num_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Multi-head self-attention + residual + layer norm
        attn_out, attn_weights = self.self_attention.forward(x, x, x, mask)
        x = layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)

        # Sub-layer 2: Feed-forward + residual + layer norm
        ff_out = self.feed_forward.forward(x)
        x = layer_norm(x + ff_out, self.ln2_gamma, self.ln2_beta)

        return x, attn_weights


# ==============================================================================
# SECTION 8: DECODER LAYER
# ==============================================================================
#
# The decoder layer has THREE sub-layers (encoder had two):
#
#   1. MASKED Self-Attention:
#      Same as encoder self-attention, BUT with a causal mask that prevents
#      each position from attending to FUTURE positions.
#      WHY? During generation, we produce tokens left-to-right. Token at
#      position 3 should NOT see tokens at positions 4, 5, 6...
#      That would be "cheating" (looking at the answer).
#
#   2. Cross-Attention (Encoder-Decoder Attention):
#      The decoder attends to the ENCODER's output.
#      Q comes from the decoder, K and V come from the encoder.
#      This is where the decoder "reads" the source sequence.
#      (Similar to Bahdanau attention from our previous tutorial!)
#
#   3. Feed-Forward:
#      Same as the encoder's FFN.
#
# Each sub-layer has residual connections + layer normalization.
# ==============================================================================

class DecoderLayer:
    """
    One Transformer Decoder layer.

    Components:
    - Masked multi-head self-attention
    - Multi-head cross-attention (to encoder output)
    - Feed-forward network
    - Layer normalization (x3)
    - Residual connections (x3)

    Args:
        d_model:   embedding dimension
        num_heads: number of attention heads
        d_ff:      feed-forward hidden dimension
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer norms for each sub-layer
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
        self.ln3_gamma = np.ones(d_model)
        self.ln3_beta = np.zeros(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x:              decoder input, shape (tgt_len, d_model)
            encoder_output: encoder output, shape (src_len, d_model)
            src_mask:       mask for encoder output (padding)
            tgt_mask:       causal mask for decoder (prevent future peeking)

        Returns:
            output: shape (tgt_len, d_model)
            self_attn_weights: shape (num_heads, tgt_len, tgt_len)
            cross_attn_weights: shape (num_heads, tgt_len, src_len)
        """
        # Sub-layer 1: Masked self-attention (can't look at future tokens)
        self_attn_out, self_attn_w = self.masked_self_attention.forward(
            x, x, x, mask=tgt_mask
        )
        x = layer_norm(x + self_attn_out, self.ln1_gamma, self.ln1_beta)

        # Sub-layer 2: Cross-attention (decoder attends to encoder output)
        # Q = decoder states, K = encoder output, V = encoder output
        cross_attn_out, cross_attn_w = self.cross_attention.forward(
            x, encoder_output, encoder_output, mask=src_mask
        )
        x = layer_norm(x + cross_attn_out, self.ln2_gamma, self.ln2_beta)

        # Sub-layer 3: Feed-forward
        ff_out = self.feed_forward.forward(x)
        x = layer_norm(x + ff_out, self.ln3_gamma, self.ln3_beta)

        return x, self_attn_w, cross_attn_w


print("\n" + "-" * 70)
print("Section 7-8: Encoder and Decoder layers loaded.")
print("\nDemo:")
enc_layer = EncoderLayer(d_model=16, num_heads=4, d_ff=64)
dec_layer = DecoderLayer(d_model=16, num_heads=4, d_ff=64)

src_demo = np.random.randn(5, 16) * 0.1  # 5 source words
tgt_demo = np.random.randn(3, 16) * 0.1  # 3 target words

enc_out, enc_w = enc_layer.forward(src_demo)
print(f"  Encoder: input {src_demo.shape} -> output {enc_out.shape}")

# Create causal mask for decoder: upper triangle = True (blocked)
causal_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
dec_out, dec_sw, dec_cw = dec_layer.forward(tgt_demo, enc_out, tgt_mask=causal_mask)
print(f"  Decoder: input {tgt_demo.shape} -> output {dec_out.shape}")
print(f"  Causal mask (True = blocked):")
print(f"    {causal_mask[0]}   <- position 0 can see only itself")
print(f"    {causal_mask[1]}   <- position 1 can see 0 and 1")
print(f"    {causal_mask[2]}   <- position 2 can see 0, 1, and 2")


# ==============================================================================
# SECTION 9: THE FULL TRANSFORMER MODEL
# ==============================================================================
#
# Now we assemble everything into the complete Transformer:
#
#   ENCODER SIDE:
#   1. Embed source tokens
#   2. Add positional encoding
#   3. Pass through N encoder layers
#
#   DECODER SIDE:
#   1. Embed target tokens
#   2. Add positional encoding
#   3. Pass through N decoder layers (each attending to encoder output)
#   4. Project to vocabulary size -> softmax -> predicted word
#
# For training, we use TEACHER FORCING (same as Encoder-Decoder tutorial):
# feed the correct previous token at each decoder step.
#
# For inference, we use GREEDY DECODING:
# feed the model's own prediction back as the next input.
# ==============================================================================

class Transformer:
    """
    Full Transformer model (Encoder-Decoder architecture).

    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        d_model:        embedding dimension (default: 64)
        num_heads:      number of attention heads (default: 4)
        num_layers:     number of encoder/decoder layers (default: 2)
        d_ff:           feed-forward hidden dimension (default: 256)
        max_len:        maximum sequence length (default: 100)
        learning_rate:  SGD learning rate (default: 0.001)
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=64, num_heads=4, num_layers=2,
                 d_ff=256, max_len=100, learning_rate=0.001):

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.tgt_vocab_size = tgt_vocab_size

        # --- Embedding layers ---
        scale = np.sqrt(2.0 / d_model)
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * scale
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * scale

        # --- Positional encoding (fixed, not learned) ---
        self.pe = positional_encoding(max_len, d_model)

        # --- Encoder layers ---
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        # --- Decoder layers ---
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        # --- Output projection (decoder output -> vocab logits) ---
        self.output_projection = np.random.randn(d_model, tgt_vocab_size) * scale
        self.output_bias = np.zeros(tgt_vocab_size)

    def _embed_src(self, src_indices):
        """Embed source tokens and add positional encoding."""
        seq_len = len(src_indices)
        # Look up embeddings
        embedded = self.src_embedding[src_indices]  # (seq_len, d_model)
        # Scale embeddings (as in the paper) and add positional encoding
        embedded = embedded * np.sqrt(self.d_model)
        embedded = embedded + self.pe[:seq_len]
        return embedded

    def _embed_tgt(self, tgt_indices):
        """Embed target tokens and add positional encoding."""
        seq_len = len(tgt_indices)
        embedded = self.tgt_embedding[tgt_indices]  # (seq_len, d_model)
        embedded = embedded * np.sqrt(self.d_model)
        embedded = embedded + self.pe[:seq_len]
        return embedded

    def _causal_mask(self, size):
        """
        Create a causal (look-ahead) mask.

        Returns a boolean matrix where True means "BLOCK this position".
        Upper triangle is True -> each position can only see itself and before.

             pos 0  pos 1  pos 2
        pos 0 [False  True  True ]   <- can only see pos 0
        pos 1 [False False  True ]   <- can see pos 0, 1
        pos 2 [False False False]    <- can see pos 0, 1, 2
        """
        return np.triu(np.ones((size, size), dtype=bool), k=1)

    def encode(self, src_indices):
        """
        Encode the source sequence.

        Args:
            src_indices: list of source token ids

        Returns:
            encoder_output: shape (src_len, d_model)
        """
        x = self._embed_src(src_indices)

        for layer in self.encoder_layers:
            x, _ = layer.forward(x)

        return x

    def decode(self, tgt_indices, encoder_output):
        """
        Decode the target sequence (with teacher forcing).

        Args:
            tgt_indices:    list of target token ids
            encoder_output: shape (src_len, d_model)

        Returns:
            logits: shape (tgt_len, tgt_vocab_size)
        """
        x = self._embed_tgt(tgt_indices)
        tgt_mask = self._causal_mask(len(tgt_indices))

        for layer in self.decoder_layers:
            x, _, _ = layer.forward(x, encoder_output, tgt_mask=tgt_mask)

        # Project to vocabulary
        logits = x @ self.output_projection + self.output_bias  # (tgt_len, vocab)
        return logits

    def forward(self, src_indices, tgt_indices):
        """
        Full forward pass: encode source, then decode target.

        Args:
            src_indices: list of int (source token ids)
            tgt_indices: list of int (target token ids, starting with <sos>)

        Returns:
            logits: shape (tgt_len, tgt_vocab_size) - raw scores for each position
        """
        encoder_output = self.encode(src_indices)
        logits = self.decode(tgt_indices, encoder_output)
        return logits

    def compute_loss(self, logits, target_indices):
        """
        Cross-entropy loss.

        For each decoder position, we have logits over the vocabulary.
        The target is the NEXT token (shifted by 1).

        Args:
            logits:         shape (tgt_len, vocab_size)
            target_indices: list of int (includes <sos> and <eos>)

        Returns:
            float: average cross-entropy loss
        """
        total_loss = 0.0
        count = 0
        # logits[t] should predict target_indices[t+1]
        for t in range(logits.shape[0] - 1):
            probs = softmax(logits[t])
            label = target_indices[t + 1]
            total_loss += -np.log(probs[label] + 1e-12)
            count += 1
        return total_loss / max(count, 1)

    def _collect_params(self):
        """Collect all learnable parameter arrays for gradient computation."""
        params = []

        # Embeddings
        params.append(self.src_embedding)
        params.append(self.tgt_embedding)

        # Encoder layers
        for layer in self.encoder_layers:
            params.extend([
                layer.self_attention.W_Q, layer.self_attention.W_K,
                layer.self_attention.W_V, layer.self_attention.W_O,
                layer.feed_forward.W1, layer.feed_forward.b1,
                layer.feed_forward.W2, layer.feed_forward.b2,
                layer.ln1_gamma, layer.ln1_beta,
                layer.ln2_gamma, layer.ln2_beta,
            ])

        # Decoder layers
        for layer in self.decoder_layers:
            params.extend([
                layer.masked_self_attention.W_Q, layer.masked_self_attention.W_K,
                layer.masked_self_attention.W_V, layer.masked_self_attention.W_O,
                layer.cross_attention.W_Q, layer.cross_attention.W_K,
                layer.cross_attention.W_V, layer.cross_attention.W_O,
                layer.feed_forward.W1, layer.feed_forward.b1,
                layer.feed_forward.W2, layer.feed_forward.b2,
                layer.ln1_gamma, layer.ln1_beta,
                layer.ln2_gamma, layer.ln2_beta,
                layer.ln3_gamma, layer.ln3_beta,
            ])

        # Output projection
        params.append(self.output_projection)
        params.append(self.output_bias)

        return params

    def train_step(self, src_indices, tgt_indices):
        """
        One training step with ANALYTICAL gradients.

        We compute full forward pass, then manually backpropagate through:
          1. Output projection (Linear layer)
          2. Decoder layers (cross-attn, self-attn, FFN) — simplified
          3. Encoder layers (self-attn, FFN) — simplified

        This is the same algorithm PyTorch/TensorFlow use automatically,
        but we write it out explicitly for learning.

        SIMPLIFIED BACKPROP STRATEGY:
        We compute exact gradients for the output projection + decoder
        embeddings (which matter most for learning), and use a lightweight
        finite-difference update for the internal attention/FFN weights.
        This gives us good training speed while keeping the code readable.

        Full backprop through multi-head attention is ~200 lines of chain-rule
        code. For learning the architecture, the simplified approach here
        teaches the same concepts.
        """
        # ---------- Forward pass (save intermediates) ----------
        encoder_output = self.encode(src_indices)

        tgt_emb = self._embed_tgt(tgt_indices)
        tgt_mask = self._causal_mask(len(tgt_indices))

        # Run decoder layers
        dec_x = tgt_emb.copy()
        for layer in self.decoder_layers:
            dec_x, _, _ = layer.forward(dec_x, encoder_output, tgt_mask=tgt_mask)

        # dec_x shape: (tgt_len, d_model)
        # Output projection: logits = dec_x @ W_out + b_out
        logits = dec_x @ self.output_projection + self.output_bias  # (tgt_len, vocab)

        # ---------- Compute loss ----------
        total_loss = 0.0
        count = 0
        # Gradient of loss w.r.t. logits
        d_logits = np.zeros_like(logits)  # (tgt_len, vocab)

        for t in range(logits.shape[0] - 1):
            probs = softmax(logits[t])
            label = tgt_indices[t + 1]
            total_loss += -np.log(probs[label] + 1e-12)
            count += 1

            # Gradient of cross-entropy w.r.t. logits:
            # d_loss/d_logits = probs - one_hot(label)
            d_logits[t] = probs.copy()
            d_logits[t, label] -= 1.0

        loss = total_loss / max(count, 1)
        d_logits /= max(count, 1)

        # ---------- Backprop through output projection ----------
        # logits = dec_x @ W_out + b_out
        # d_W_out = dec_x.T @ d_logits
        # d_b_out = sum(d_logits)
        # d_dec_x = d_logits @ W_out.T

        d_W_out = dec_x.T @ d_logits
        d_b_out = np.sum(d_logits, axis=0)
        d_dec_x = d_logits @ self.output_projection.T

        self.output_projection -= self.learning_rate * d_W_out
        self.output_bias -= self.learning_rate * d_b_out

        # ---------- Backprop through target embeddings ----------
        # The target embedding lookup is like: tgt_emb[t] = E[tgt_indices[t]] * sqrt(d) + PE[t]
        # Gradient flows back into the embedding rows that were selected
        # d_dec_x propagated through residual connections reaches tgt_emb approximately
        scale = np.sqrt(self.d_model)
        for t, idx in enumerate(tgt_indices):
            self.tgt_embedding[idx] -= self.learning_rate * d_dec_x[t] * scale

        # ---------- Backprop through encoder/decoder internal weights ----------
        # For the attention & FFN weights, we use a FAST finite-difference
        # approach on a small random subset of parameters.
        # This is a practical compromise: exact backprop through multi-head
        # attention with split/combine heads is complex. The key learning
        # point is the architecture, not the gradient plumbing.
        internal_params = []
        for layer in self.encoder_layers:
            internal_params.extend([
                layer.self_attention.W_Q, layer.self_attention.W_K,
                layer.self_attention.W_V, layer.self_attention.W_O,
                layer.feed_forward.W1, layer.feed_forward.W2,
            ])
        for layer in self.decoder_layers:
            internal_params.extend([
                layer.masked_self_attention.W_Q, layer.masked_self_attention.W_K,
                layer.masked_self_attention.W_V, layer.masked_self_attention.W_O,
                layer.cross_attention.W_Q, layer.cross_attention.W_K,
                layer.cross_attention.W_V, layer.cross_attention.W_O,
                layer.feed_forward.W1, layer.feed_forward.W2,
            ])

        # Source embeddings too
        internal_params.append(self.src_embedding)

        epsilon = 1e-4
        n_samples = 20  # update 20 random elements per parameter matrix per step

        for p in internal_params:
            indices = [tuple(np.random.randint(0, s) for s in p.shape)
                       for _ in range(min(n_samples, p.size))]
            for ix in indices:
                orig = p[ix]

                p[ix] = orig + epsilon
                logits_p = self.forward(src_indices, tgt_indices)
                loss_p = self.compute_loss(logits_p, tgt_indices)

                p[ix] = orig - epsilon
                logits_m = self.forward(src_indices, tgt_indices)
                loss_m = self.compute_loss(logits_m, tgt_indices)

                grad = (loss_p - loss_m) / (2 * epsilon)
                p[ix] = orig - self.learning_rate * grad

        return loss

    def translate(self, src_indices, max_len=20, sos_idx=1, eos_idx=2):
        """
        Translate using greedy decoding.

        Start with <sos>, predict one token at a time,
        feed it back, repeat until <eos> or max_len.

        Args:
            src_indices: list of source token ids
            max_len:     maximum output length
            sos_idx:     <sos> token index
            eos_idx:     <eos> token index

        Returns:
            output_tokens: list of predicted token ids
            cross_attention_weights: attention weights from last decoder layer
        """
        encoder_output = self.encode(src_indices)

        output_tokens = [sos_idx]
        all_cross_attn = []

        for _ in range(max_len):
            tgt_so_far = output_tokens
            x = self._embed_tgt(tgt_so_far)
            tgt_mask = self._causal_mask(len(tgt_so_far))

            cross_attn_w = None
            for layer in self.decoder_layers:
                x, _, cross_attn_w = layer.forward(
                    x, encoder_output, tgt_mask=tgt_mask
                )

            # Get logits for the LAST position only
            last_logits = x[-1] @ self.output_projection + self.output_bias
            probs = softmax(last_logits)
            next_token = int(np.argmax(probs))

            output_tokens.append(next_token)
            if cross_attn_w is not None:
                # Average over heads, take the last time-step
                all_cross_attn.append(cross_attn_w.mean(axis=0)[-1])

            if next_token == eos_idx:
                break

        return output_tokens[1:], all_cross_attn  # remove <sos> from output


print("\n" + "-" * 70)
print("Section 9: Full Transformer model loaded!")
print("  Transformer class has: encode, decode, forward, train_step, translate")


# ==============================================================================
# SECTION 10: VISUALIZING ATTENTION
# ==============================================================================

def print_attention_heatmap(weights, src_tokens, tgt_tokens, head=None):
    """Print attention weights as a text table."""
    title = "Attention Heatmap"
    if head is not None:
        title += f" (Head {head})"
    print(f"\n  {title}")
    header = "           " + "  ".join(f"{t:>8s}" for t in src_tokens)
    print(header)
    print("           " + "-" * (len(src_tokens) * 10))
    for i, tgt_tok in enumerate(tgt_tokens):
        if i < len(weights):
            row = "  ".join(f"{w:8.3f}" for w in weights[i])
            print(f"  {tgt_tok:>8s} | {row}")
    print()


# ==============================================================================
# SECTION 11: TRAINING ON REAL DATA
# ==============================================================================
#
# Let's train the Transformer on a small English -> French translation task.
# We keep it small so numerical gradients finish in reasonable time.
# ==============================================================================

print("\n" + "=" * 70)
print("SECTION 11: TRAINING THE TRANSFORMER")
print("=" * 70)

# ---- Dataset ----
training_pairs = [
    (["i", "am", "happy"],           ["je", "suis", "content"]),
    (["you", "are", "smart"],         ["tu", "es", "intelligent"]),
    (["he", "is", "tall"],            ["il", "est", "grand"]),
    (["she", "is", "kind"],           ["elle", "est", "gentille"]),
    (["we", "are", "strong"],         ["nous", "sommes", "forts"]),
    (["i", "am", "tall"],             ["je", "suis", "grand"]),
    (["you", "are", "kind"],          ["tu", "es", "gentille"]),
    (["he", "is", "happy"],           ["il", "est", "content"]),
]

print(f"\nDataset: {len(training_pairs)} English -> French translation pairs")
for en, fr in training_pairs[:3]:
    print(f"  {' '.join(en):20s} -> {' '.join(fr)}")
print("  ...")

# ---- Build vocabularies ----
print("\nBuilding vocabularies...")
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

src_sents = [p[0] for p in training_pairs]
tgt_sents = [p[1] for p in training_pairs]

print("  Source (English):", end=" ")
src_vocab.build(src_sents)
print("  Target (French):", end=" ")
tgt_vocab.build(tgt_sents)

# ---- Prepare data (add <sos> and <eos> to target) ----
sos_idx = tgt_vocab.word2idx["<sos>"]
eos_idx = tgt_vocab.word2idx["<eos>"]

data = []
for src_sent, tgt_sent in training_pairs:
    src_ids = src_vocab.encode(src_sent)
    tgt_ids = [sos_idx] + tgt_vocab.encode(tgt_sent) + [eos_idx]
    data.append((src_ids, tgt_ids))

print(f"\nPrepared {len(data)} training examples:")
for i, (s, t) in enumerate(data[:3]):
    print(f"  {src_vocab.decode(s)} -> {tgt_vocab.decode(t)}")
print("  ...")

# ---- Create the Transformer ----
print("\nCreating Transformer model...")
np.random.seed(42)

# Using TINY dimensions so numerical-gradient training finishes in reasonable time.
# In real Transformers: d_model=512, num_heads=8, num_layers=6, d_ff=2048
# Here we use small values purely for learning — the ARCHITECTURE is identical.
model = Transformer(
    src_vocab_size=src_vocab.n_words,
    tgt_vocab_size=tgt_vocab.n_words,
    d_model=16,       # embedding dimension (tiny for speed)
    num_heads=2,      # 2 attention heads of dim 8 each
    num_layers=1,     # 1 encoder + 1 decoder layer
    d_ff=32,          # feed-forward hidden dim
    max_len=50,
    learning_rate=0.01
)

n_params = sum(p.size for p in model._collect_params())
print(f"  d_model={model.d_model}, heads={model.num_heads}, layers={model.num_layers}")
print(f"  Total parameters: {n_params:,}")

# ---- Test forward pass before training ----
print("\n--- Before Training ---")
for en, fr in training_pairs[:2]:
    src_ids = src_vocab.encode(en)
    out, _ = model.translate(src_ids, sos_idx=sos_idx, eos_idx=eos_idx)
    out_words = tgt_vocab.decode(out)
    out_clean = [w for w in out_words if w not in ("<eos>", "<pad>")]
    print(f"  {' '.join(en):20s} -> {' '.join(out_clean):25s} (expected: {' '.join(fr)})")
print("  (Random garbage before training - expected!)")

# ---- Training loop ----
print("\n--- Training ---")
print("Using analytical gradients (output layer) + sampled finite-difference")
print("(internal layers). Each epoch trains on ALL examples.\n")

num_epochs = 30

for epoch in range(num_epochs):
    t0 = time.time()
    total_loss = 0

    for src_ids, tgt_ids in data:
        loss = model.train_step(src_ids, tgt_ids)
        total_loss += loss

    elapsed = time.time() - t0
    avg_loss = total_loss / len(data)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:>2d}/{num_epochs}  |  Loss: {avg_loss:.4f}  |  Time: {elapsed:.1f}s")

print("\nTraining complete!")


# ---- Test translations after training ----
print("\n" + "=" * 70)
print("TRANSLATION RESULTS (after training)")
print("=" * 70)

for en, fr in training_pairs:
    src_ids = src_vocab.encode(en)
    out, cross_attn = model.translate(src_ids, sos_idx=sos_idx, eos_idx=eos_idx)
    out_words = tgt_vocab.decode(out)
    out_clean = [w for w in out_words if w not in ("<eos>", "<pad>")]

    match = "[OK]" if out_clean == fr else "[X]"
    print(f"  {match} {' '.join(en):20s} -> {' '.join(out_clean):25s} (expected: {' '.join(fr)})")


# ---- Visualize attention ----
print("\n" + "=" * 70)
print("ATTENTION VISUALIZATION")
print("=" * 70)
print("\nLet's see which source words the decoder attends to:")

test_src = training_pairs[0][0]  # ["i", "am", "happy"]
src_ids = src_vocab.encode(test_src)
out_ids, cross_attn_list = model.translate(src_ids, sos_idx=sos_idx, eos_idx=eos_idx)
out_words = tgt_vocab.decode(out_ids)
out_display = [w for w in out_words if w not in ("<eos>", "<pad>")]

print(f"\n  Source:  {test_src}")
print(f"  Output:  {out_display}")

if cross_attn_list:
    attn_mat = np.array(cross_attn_list)
    if len(out_words) > 0 and attn_mat.shape[0] > 0:
        print_attention_heatmap(attn_mat, test_src, out_words)


# ==============================================================================
# SECTION 12: STEP-BY-STEP FORWARD PASS WALKTHROUGH
# ==============================================================================

print("\n" + "=" * 70)
print("STEP-BY-STEP FORWARD PASS WALKTHROUGH")
print("=" * 70)

test_src = ["i", "am", "happy"]
test_tgt = ["je", "suis", "content"]

src_ids = src_vocab.encode(test_src)
tgt_ids = [sos_idx] + tgt_vocab.encode(test_tgt) + [eos_idx]

print(f"\n  Source: {test_src}  ->  indices: {src_ids}")
print(f"  Target: ['<sos>'] + {test_tgt} + ['<eos>']  ->  indices: {tgt_ids}")

# Step 1: Source embedding + positional encoding
src_emb = model._embed_src(src_ids)
print(f"\n  1. Source Embedding + Positional Encoding:")
print(f"     Shape: {src_emb.shape}  ({len(src_ids)} words x {model.d_model} dims)")
print(f"     Word 'i':     [{src_emb[0, 0]:.4f}, {src_emb[0, 1]:.4f}, ...]")
print(f"     Word 'am':    [{src_emb[1, 0]:.4f}, {src_emb[1, 1]:.4f}, ...]")
print(f"     Word 'happy': [{src_emb[2, 0]:.4f}, {src_emb[2, 1]:.4f}, ...]")

# Step 2: Encoder
enc_out = model.encode(src_ids)
print(f"\n  2. Encoder Output:")
print(f"     Shape: {enc_out.shape}")
print(f"     Each position now contains context from ALL source words")
print(f"     (via self-attention)")

# Step 3: Target embedding
tgt_emb = model._embed_tgt(tgt_ids)
print(f"\n  3. Target Embedding + Positional Encoding:")
print(f"     Shape: {tgt_emb.shape}  ({len(tgt_ids)} words x {model.d_model} dims)")

# Step 4: Causal mask
mask = model._causal_mask(len(tgt_ids))
print(f"\n  4. Causal Mask (prevents future peeking):")
for i in range(len(tgt_ids)):
    word = tgt_vocab.idx2word[tgt_ids[i]]
    visible = [tgt_vocab.idx2word[tgt_ids[j]] for j in range(i + 1)]
    print(f"     '{word}' can see: {visible}")

# Step 5: Decoder forward
logits = model.forward(src_ids, tgt_ids)
print(f"\n  5. Decoder Output (logits):")
print(f"     Shape: {logits.shape}  ({len(tgt_ids)} positions x {model.tgt_vocab_size} vocab)")
for t in range(logits.shape[0] - 1):
    probs = softmax(logits[t])
    pred_idx = int(np.argmax(probs))
    pred_word = tgt_vocab.idx2word[pred_idx]
    target_word = tgt_vocab.idx2word[tgt_ids[t + 1]]
    prob = probs[pred_idx]
    print(f"     Position {t}: predicted '{pred_word}' (p={prob:.3f}), target '{target_word}'"
          f"  {'[OK]' if pred_idx == tgt_ids[t+1] else '[X]'}")


# ==============================================================================
# SECTION 13: SUMMARY & KEY TAKEAWAYS
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY & KEY TAKEAWAYS")
print("=" * 70)

print("""
What we built:
  1. Scaled Dot-Product Attention  - The core mechanism (Q, K, V)
  2. Multi-Head Attention          - Multiple attention patterns in parallel
  3. Positional Encoding           - Inject position info (since no RNN)
  4. Feed-Forward Network          - Non-linear transform at each position
  5. Encoder Layer                 - Self-attention + FFN + residual + LayerNorm
  6. Decoder Layer                 - Masked self-attn + cross-attn + FFN
  7. Full Transformer              - Stack of encoder/decoder layers

Key differences from Encoder-Decoder with RNN (previous tutorial):
  +----------------------------+------------------+--------------------+
  | Feature                    | RNN + Attention  | Transformer        |
  +----------------------------+------------------+--------------------+
  | Sequence processing        | Sequential       | Parallel           |
  | Position information       | Built into RNN   | Positional encoding|
  | Attention type             | Additive         | Scaled dot-product |
  | Number of attention heads  | 1                | Multiple (4-16+)   |
  | Training speed             | Slow (sequential)| Fast (parallel)    |
  +----------------------------+------------------+--------------------+

The road ahead:
  Word2Vec -> Enc-Dec + Attention -> Transformer (done!) -> GPT / BERT
                                                            ^^^^^^^^^^^^
                                                         Your next step!

  - GPT = Transformer DECODER only (auto-regressive language model)
  - BERT = Transformer ENCODER only (bidirectional, masked language model)
""")

print("Tutorial complete! You now understand the Transformer architecture")
print("from the ground up.")

