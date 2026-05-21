"""
Encoder-Decoder with Attention — Implementation from Scratch

This module implements a Sequence-to-Sequence (Seq2Seq) model with the
Bahdanau (additive) attention mechanism, built entirely with NumPy.
Designed for learning purposes so you can understand every moving part.

Key Concepts:
- Encoder:   Reads the input sequence and compresses it into hidden states
- Decoder:   Generates the output sequence one token at a time
- Attention: Lets the decoder "look back" at ALL encoder hidden states
             instead of relying on a single fixed-size context vector
- GRU Cell:  A gated recurrent unit (simpler cousin of LSTM)

Architecture Overview:
    Input tokens  →  [Embedding]  →  [Encoder GRU]  →  encoder hidden states
                                                              ↓
    Output tokens →  [Embedding]  →  [Decoder GRU + Attention]  →  predictions

Reference Papers:
    - Sutskever et al., 2014: "Sequence to Sequence Learning with Neural Networks"
    - Bahdanau et al., 2015: "Neural Machine Translation by Jointly Learning to Align and Translate"
"""

import numpy as np
from collections import Counter


# =============================================================================
# BUILDING BLOCK 1: Activation Functions
# =============================================================================

def sigmoid(x):
    """
    Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    
    Used inside GRU gates (update & reset) to squash values to [0, 1].
    A gate value of 0 means "completely ignore", 1 means "completely use".
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """
    Tanh activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Used to produce candidate hidden states in [-1, 1].
    """
    return np.tanh(x)


def softmax(x):
    """
    Softmax: converts raw scores (logits) into probabilities that sum to 1.
    
    softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    Used in:
    - Attention weights (which encoder states to focus on)
    - Output layer  (which word to predict next)
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for numerical stability
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# =============================================================================
# BUILDING BLOCK 2: Vocabulary
# =============================================================================

class Vocabulary:
    """
    Maps words ↔ integer indices.
    
    Special tokens:
        <pad> = 0   Padding for batching (not used much here, but good practice)
        <sos> = 1   Start-of-sequence — fed as the first decoder input
        <eos> = 2   End-of-sequence — signals the decoder to stop
        <unk> = 3   Unknown word (out-of-vocabulary fallback)
    """

    SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.n_words = 0

        # Add special tokens
        for token in self.SPECIAL_TOKENS:
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def build_from_sentences(self, sentences, min_count=1):
        """Count words, then keep only those with freq >= min_count."""
        for sent in sentences:
            for word in sent:
                self.word_count[word] += 1

        for word, count in self.word_count.items():
            if count >= min_count:
                self._add_word(word)

        print(f"Vocabulary built: {self.n_words} tokens "
              f"(including {len(self.SPECIAL_TOKENS)} special tokens)")

    def encode(self, sentence):
        """Convert a list of words → list of indices."""
        unk_idx = self.word2idx["<unk>"]
        return [self.word2idx.get(w, unk_idx) for w in sentence]

    def decode(self, indices):
        """Convert a list of indices → list of words."""
        return [self.idx2word.get(i, "<unk>") for i in indices]


# =============================================================================
# BUILDING BLOCK 3: GRU Cell  (Gated Recurrent Unit)
# =============================================================================

class GRUCell:
    """
    A single GRU cell that processes ONE time-step.

    GRU has two gates:
        - Update gate (z): how much of the old hidden state to keep
        - Reset gate  (r): how much of the old hidden state to forget
                           before computing the candidate

    Equations (for one time-step):
        z = σ(W_z · [h_{t-1}, x_t])          ← update gate
        r = σ(W_r · [h_{t-1}, x_t])          ← reset gate
        h̃ = tanh(W_h · [r ⊙ h_{t-1}, x_t])  ← candidate hidden state
        h_t = (1 - z) ⊙ h̃ + z ⊙ h_{t-1}    ← final hidden state

    Here ⊙ means element-wise multiplication.

    Args:
        input_dim  (int): Dimension of the input vector x_t
        hidden_dim (int): Dimension of the hidden state h_t
    """

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # Update gate weights
        self.W_z = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_z = np.zeros((hidden_dim, 1))

        # Reset gate weights
        self.W_r = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_r = np.zeros((hidden_dim, 1))

        # Candidate hidden state weights
        self.W_h = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_h = np.zeros((hidden_dim, 1))

    def forward(self, x, h_prev):
        """
        One forward step of the GRU.

        Args:
            x      (np.ndarray): Input vector, shape (input_dim, 1)
            h_prev (np.ndarray): Previous hidden state, shape (hidden_dim, 1)

        Returns:
            h_next (np.ndarray): New hidden state, shape (hidden_dim, 1)
        """
        # Concatenate input and previous hidden state: [h_{t-1}; x_t]
        combined = np.vstack([h_prev, x])  # shape: (input_dim + hidden_dim, 1)

        # Update gate: decides how much of the past to keep
        z = sigmoid(self.W_z @ combined + self.b_z)

        # Reset gate: decides how much of the past to forget
        r = sigmoid(self.W_r @ combined + self.b_r)

        # Candidate hidden state: new information to potentially add
        combined_r = np.vstack([r * h_prev, x])
        h_candidate = tanh(self.W_h @ combined_r + self.b_h)

        # Final hidden state: blend old and new
        h_next = (1 - z) * h_candidate + z * h_prev

        return h_next


# =============================================================================
# BUILDING BLOCK 4: Embedding Layer
# =============================================================================

class EmbeddingLayer:
    """
    Converts word indices → dense vectors.

    This is just a lookup table. If word index = 5, we return row 5
    of the embedding matrix. The matrix values are learned during training.

    Args:
        vocab_size    (int): Total number of words
        embedding_dim (int): Size of each embedding vector
    """

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize with small random values
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1

    def forward(self, idx):
        """
        Look up the embedding for a single word index.

        Args:
            idx (int): Word index

        Returns:
            np.ndarray of shape (embedding_dim, 1)
        """
        return self.weights[idx].reshape(-1, 1)


# =============================================================================
# BUILDING BLOCK 5: Bahdanau (Additive) Attention
# =============================================================================

class BahdanauAttention:
    """
    Additive attention mechanism (Bahdanau et al., 2015).

    *** THE KEY IDEA ***
    Instead of forcing the entire input sequence into one fixed vector,
    attention lets the decoder LOOK AT EVERY encoder hidden state and
    decide which ones are most relevant at each decoding step.

    Score function (additive / concat):
        score(s_t, h_j) = v^T · tanh(W_a · s_t + U_a · h_j)

    Where:
        s_t  = decoder hidden state at time t       (what I'm trying to produce)
        h_j  = encoder hidden state at position j   (what I can look at)
        W_a, U_a, v = learnable parameters

    Attention weights (how much to focus on each encoder state):
        α_{t,j} = softmax_j( score(s_t, h_j) )

    Context vector (weighted sum of encoder states):
        c_t = Σ_j  α_{t,j} · h_j

    This context vector c_t is then fed into the decoder along with the
    current input, giving the decoder access to the most relevant parts
    of the source sequence.

    Args:
        encoder_dim (int): Dimension of encoder hidden states
        decoder_dim (int): Dimension of decoder hidden state
        attention_dim (int): Internal dimension of the attention network
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        self.attention_dim = attention_dim

        scale = np.sqrt(2.0 / attention_dim)

        # W_a maps decoder state → attention space
        self.W_a = np.random.randn(attention_dim, decoder_dim) * scale
        # U_a maps each encoder state → attention space
        self.U_a = np.random.randn(attention_dim, encoder_dim) * scale
        # v projects to a scalar score
        self.v = np.random.randn(1, attention_dim) * scale

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden  (np.ndarray): shape (decoder_dim, 1)
            encoder_outputs (list of np.ndarray): each shape (encoder_dim, 1),
                            one per input time-step

        Returns:
            context (np.ndarray): weighted sum, shape (encoder_dim, 1)
            weights (np.ndarray): attention weights, shape (src_len,)
        """
        src_len = len(encoder_outputs)
        scores = np.zeros(src_len)

        for j in range(src_len):
            h_j = encoder_outputs[j]  # (encoder_dim, 1)

            # score = v^T · tanh(W_a · s_t  +  U_a · h_j)
            energy = tanh(self.W_a @ decoder_hidden + self.U_a @ h_j)  # (attn_dim, 1)
            scores[j] = (self.v @ energy).item()  # scalar

        # Normalize scores → probabilities
        weights = softmax(scores)  # (src_len,)

        # Context = weighted sum of encoder outputs
        context = np.zeros_like(encoder_outputs[0])
        for j in range(src_len):
            context += weights[j] * encoder_outputs[j]

        return context, weights


# =============================================================================
# BUILDING BLOCK 6: Output (Linear) Layer
# =============================================================================

class LinearLayer:
    """
    Simple linear transformation:  y = W · x + b

    Maps the decoder's output vector → logits over the vocabulary,
    so we can apply softmax to get a probability for each word.

    Args:
        in_dim  (int): Input dimension
        out_dim (int): Output dimension (usually vocab size)
    """

    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((out_dim, 1))

    def forward(self, x):
        """
        Args:
            x (np.ndarray): shape (in_dim, 1)
        Returns:
            np.ndarray: shape (out_dim, 1)
        """
        return self.W @ x + self.b


# =============================================================================
# THE FULL MODEL: Encoder–Decoder with Attention
# =============================================================================

class Seq2SeqAttention:
    """
    Complete Sequence-to-Sequence model with Bahdanau attention.

    Data flow at training time for one example:

        SOURCE: ["i", "love", "cats"]    TARGET: ["j'aime", "les", "chats"]

        1. Encode source  →  encoder hidden states  h_1, h_2, h_3
        2. Decoder step 1:
              input  = <sos> embedding
              attend to h_1..h_3 → context vector c_1
              GRU([ <sos>_emb ; c_1 ], prev_hidden) → new_hidden
              Linear(new_hidden) → logits → predict "j'aime"
        3. Decoder step 2:  (teacher forcing: feed ground-truth "j'aime")
              attend → c_2,  GRU → hidden,  predict "les"
        4. Decoder step 3:
              attend → c_3,  GRU → hidden,  predict "chats"
        5. Decoder step 4:
              predict <eos> → STOP

    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        embedding_dim  (int): Embedding vector dimension
        hidden_dim     (int): GRU hidden state dimension
        attention_dim  (int): Internal attention dimension
        learning_rate  (float): SGD learning rate
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embedding_dim=64, hidden_dim=128, attention_dim=64,
                 learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # --- Encoder components ---
        self.enc_embedding = EmbeddingLayer(src_vocab_size, embedding_dim)
        self.enc_gru = GRUCell(embedding_dim, hidden_dim)

        # --- Decoder components ---
        self.dec_embedding = EmbeddingLayer(tgt_vocab_size, embedding_dim)
        # Decoder GRU input = embedding + context vector  →  dim = embedding_dim + hidden_dim
        self.dec_gru = GRUCell(embedding_dim + hidden_dim, hidden_dim)

        # --- Attention ---
        self.attention = BahdanauAttention(
            encoder_dim=hidden_dim,
            decoder_dim=hidden_dim,
            attention_dim=attention_dim
        )

        # --- Output projection ---
        self.output_layer = LinearLayer(hidden_dim, tgt_vocab_size)

    # -----------------------------------------------------------------
    # ENCODER
    # -----------------------------------------------------------------
    def encode(self, src_indices):
        """
        Run the encoder over the source sequence.

        Args:
            src_indices (list[int]): Source token indices

        Returns:
            encoder_outputs (list[np.ndarray]): Hidden state at each position
            h (np.ndarray): Final encoder hidden state (used to init decoder)
        """
        h = np.zeros((self.hidden_dim, 1))  # initial hidden state
        encoder_outputs = []

        for idx in src_indices:
            x = self.enc_embedding.forward(idx)   # (emb_dim, 1)
            h = self.enc_gru.forward(x, h)        # (hidden_dim, 1)
            encoder_outputs.append(h.copy())

        return encoder_outputs, h

    # -----------------------------------------------------------------
    # DECODER  (one step)
    # -----------------------------------------------------------------
    def decode_step(self, token_idx, h_prev, encoder_outputs):
        """
        One step of decoding with attention.

        Args:
            token_idx (int): Current input token index
            h_prev (np.ndarray): Previous decoder hidden state (hidden_dim, 1)
            encoder_outputs (list[np.ndarray]): All encoder hidden states

        Returns:
            logits  (np.ndarray): Raw scores over target vocab (vocab_size, 1)
            h_next  (np.ndarray): Updated decoder hidden state (hidden_dim, 1)
            attn_weights (np.ndarray): Attention weights (src_len,)
        """
        # 1. Embed the current target token
        emb = self.dec_embedding.forward(token_idx)  # (emb_dim, 1)

        # 2. Compute attention context from encoder outputs
        context, attn_weights = self.attention.forward(h_prev, encoder_outputs)

        # 3. Concatenate embedding + context → GRU input
        gru_input = np.vstack([emb, context])  # (emb_dim + hidden_dim, 1)

        # 4. GRU step
        h_next = self.dec_gru.forward(gru_input, h_prev)  # (hidden_dim, 1)

        # 5. Project to vocabulary → logits
        logits = self.output_layer.forward(h_next)  # (tgt_vocab_size, 1)

        return logits, h_next, attn_weights

    # -----------------------------------------------------------------
    # FORWARD PASS  (full sequence, with teacher forcing)
    # -----------------------------------------------------------------
    def forward(self, src_indices, tgt_indices):
        """
        Full forward pass: encode source, then decode target with teacher forcing.

        Teacher forcing means we feed the GROUND-TRUTH previous token to the
        decoder at each step (instead of its own prediction). This is standard
        during training because it stabilises learning.

        Args:
            src_indices (list[int]): Source token indices
            tgt_indices (list[int]): Target token indices (includes <sos> at start)

        Returns:
            all_logits   (list[np.ndarray]): Logits at each decoder step
            all_attn     (list[np.ndarray]): Attention weights at each step
        """
        # Encode
        encoder_outputs, h = self.encode(src_indices)

        all_logits = []
        all_attn = []

        # Decode step by step (feed ground-truth tokens = teacher forcing)
        for t in range(len(tgt_indices) - 1):
            input_token = tgt_indices[t]
            logits, h, attn_w = self.decode_step(input_token, h, encoder_outputs)
            all_logits.append(logits)
            all_attn.append(attn_w)

        return all_logits, all_attn

    # -----------------------------------------------------------------
    # LOSS: Cross-Entropy
    # -----------------------------------------------------------------
    @staticmethod
    def compute_loss(all_logits, tgt_indices):
        """
        Cross-entropy loss averaged over all decoder steps.

        L = - (1/T) Σ_t log P(y_t)

        Where P(y_t) is the softmax probability the model assigns to the
        correct target word at step t.

        Args:
            all_logits  (list[np.ndarray]): Each shape (vocab_size, 1)
            tgt_indices (list[int]):        Target tokens (shifted by 1 — 
                                            tgt_indices[1:] are the labels)

        Returns:
            float: Average cross-entropy loss
        """
        total_loss = 0.0
        for t, logits in enumerate(all_logits):
            probs = softmax(logits.flatten())
            target_idx = tgt_indices[t + 1]  # label is the NEXT token
            total_loss += -np.log(probs[target_idx] + 1e-12)

        return total_loss / len(all_logits)

    # -----------------------------------------------------------------
    # NUMERICAL-GRADIENT TRAINING (simple but correct)
    # -----------------------------------------------------------------
    def _numerical_gradient(self, param, src, tgt, epsilon=1e-5):
        """
        Estimate gradient of loss w.r.t. every element of `param`
        using the two-sided finite difference:

            ∂L/∂θ_i ≈ (L(θ_i + ε) - L(θ_i - ε)) / (2ε)

        This is SLOW but correct — great for learning and debugging.
        """
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            orig = param[ix]

            param[ix] = orig + epsilon
            logits_p, _ = self.forward(src, tgt)
            loss_p = self.compute_loss(logits_p, tgt)

            param[ix] = orig - epsilon
            logits_m, _ = self.forward(src, tgt)
            loss_m = self.compute_loss(logits_m, tgt)

            grad[ix] = (loss_p - loss_m) / (2 * epsilon)
            param[ix] = orig  # restore
            it.iternext()
        return grad

    def train_step(self, src_indices, tgt_indices):
        """
        One training step using numerical gradients (SGD update).

        For every learnable parameter tensor in the model we:
            1. Compute gradients via finite differences
            2. Update:  θ ← θ - lr · ∇L

        Returns:
            float: Loss before the update
        """
        # Current loss
        logits, attn = self.forward(src_indices, tgt_indices)
        loss = self.compute_loss(logits, tgt_indices)

        # List of all parameter arrays to update
        params = [
            self.enc_embedding.weights,
            self.enc_gru.W_z, self.enc_gru.b_z,
            self.enc_gru.W_r, self.enc_gru.b_r,
            self.enc_gru.W_h, self.enc_gru.b_h,
            self.dec_embedding.weights,
            self.dec_gru.W_z, self.dec_gru.b_z,
            self.dec_gru.W_r, self.dec_gru.b_r,
            self.dec_gru.W_h, self.dec_gru.b_h,
            self.attention.W_a, self.attention.U_a, self.attention.v,
            self.output_layer.W, self.output_layer.b,
        ]

        for p in params:
            grad = self._numerical_gradient(p, src_indices, tgt_indices)
            p -= self.learning_rate * grad

        return loss

    # -----------------------------------------------------------------
    # GREEDY DECODING (inference time)
    # -----------------------------------------------------------------
    def translate(self, src_indices, max_len=20, sos_idx=1, eos_idx=2):
        """
        Translate a source sequence using greedy decoding.

        At each step, pick the word with the highest probability.
        Stop when <eos> is produced or max_len is reached.

        Args:
            src_indices (list[int]): Source token ids
            max_len (int): Maximum output length
            sos_idx (int): Index of <sos> token
            eos_idx (int): Index of <eos> token

        Returns:
            output_indices (list[int]): Predicted token ids
            attention_matrix (list[np.ndarray]): Attention weights per step
        """
        encoder_outputs, h = self.encode(src_indices)

        input_token = sos_idx
        output_indices = []
        attention_matrix = []

        for _ in range(max_len):
            logits, h, attn_w = self.decode_step(input_token, h, encoder_outputs)
            probs = softmax(logits.flatten())
            predicted_idx = int(np.argmax(probs))

            output_indices.append(predicted_idx)
            attention_matrix.append(attn_w)

            if predicted_idx == eos_idx:
                break

            input_token = predicted_idx  # auto-regressive: feed own prediction

        return output_indices, attention_matrix


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_attention_matrix(attention_matrix, src_tokens, tgt_tokens):
    """
    Display an attention heatmap showing which source tokens the decoder
    focused on when producing each target token.

    Args:
        attention_matrix: list of np.ndarray, each shape (src_len,)
        src_tokens: list of source words
        tgt_tokens: list of target words (predictions)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    mat = np.array(attention_matrix)  # (tgt_len, src_len)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(mat, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='left')
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("Source (encoder)")
    ax.set_ylabel("Target (decoder)")
    ax.set_title("Attention Weights")

    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def print_attention_text(attention_matrix, src_tokens, tgt_tokens):
    """
    Print attention weights as a simple text table (no matplotlib needed).
    """
    print("\n  Attention Heatmap (rows=target, cols=source)")
    header = "           " + "  ".join(f"{t:>8s}" for t in src_tokens)
    print(header)
    print("           " + "-" * (len(src_tokens) * 10))
    for i, tgt_tok in enumerate(tgt_tokens):
        weights = attention_matrix[i]
        row = "  ".join(f"{w:8.3f}" for w in weights)
        print(f"  {tgt_tok:>8s} | {row}")
    print()

