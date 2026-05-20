"""
Word2Vec Implementation from Scratch

This module implements the Word2Vec algorithm, specifically the Skip-gram model
with negative sampling. This implementation is designed for learning purposes
to understand how word embeddings work.

Key Concepts:
- Word embeddings: Dense vector representations of words
- Skip-gram model: Given a target word, predict context words
- Negative sampling: Training optimization technique
- Context window: How many words around a target we consider
"""

import numpy as np
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class Word2VecSkipGram:
    """
    Skip-gram model implementation from scratch.

    The Skip-gram model learns word embeddings by:
    1. Taking a target word and its context window
    2. Learning to predict context words from the target word
    3. Using negative sampling to make training efficient

    Args:
        embedding_dim (int): Dimension of word embeddings (default: 100)
        window_size (int): Context window size on each side of target word (default: 5)
        negative_samples (int): Number of negative samples per positive (default: 5)
        learning_rate (float): Learning rate for gradient descent (default: 0.01)
        min_count (int): Minimum word frequency to include (default: 2)
    """

    def __init__(self, embedding_dim=100, window_size=5, negative_samples=5,
                 learning_rate=0.01, min_count=2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count

        # Dictionary to map words to indices
        self.word2idx = {}
        self.idx2word = {}

        # Embedding matrices
        # W: target word embeddings (what we mainly care about)
        # W_context: context word embeddings (auxiliary during training)
        self.W = None
        self.W_context = None

        # For negative sampling
        self.word_freq = None
        self.word_table = None

        # Training statistics
        self.vocab_size = 0

    def _build_vocab(self, sentences):
        """
        Build vocabulary from sentences.

        Args:
            sentences (list): List of word lists (tokenized sentences)
        """
        word_freq = Counter()

        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                word_freq[word] += 1

        # Filter words by minimum count
        self.word_freq = {word: freq for word, freq in word_freq.items()
                         if freq >= self.min_count}

        # Create mappings
        for idx, word in enumerate(sorted(self.word_freq.keys())):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size}")

    def _build_negative_sampling_table(self):
        """
        Build a frequency table for negative sampling.

        Words are sampled according to their frequency raised to the power of 0.75
        (this is an empirical choice that works well in practice).

        The table is built as a large list where each word appears a number of times
        proportional to its sampling probability.
        """
        power = 0.75
        freq_sum = 0

        # Normalize frequencies by power
        norm_freq = {}
        for word, freq in self.word_freq.items():
            norm_freq[word] = freq ** power
            freq_sum += norm_freq[word]

        # Build word table for efficient sampling
        self.word_table = []
        for word, norm_f in norm_freq.items():
            # Add word index multiple times based on its frequency
            count = int((norm_f / freq_sum) * 1e6)  # Scale to 1 million
            self.word_table.extend([self.word2idx[word]] * count)

        print(f"Negative sampling table size: {len(self.word_table)}")

    def _init_embeddings(self):
        """
        Initialize embedding matrices with small random values.

        Proper initialization is important for training stability.
        We use uniform distribution in range [-0.5/embedding_dim, 0.5/embedding_dim]
        """
        bound = 0.5 / self.embedding_dim

        # Target word embeddings
        self.W = np.random.uniform(
            -bound, bound,
            size=(self.vocab_size, self.embedding_dim)
        )

        # Context word embeddings
        self.W_context = np.random.uniform(
            -bound, bound,
            size=(self.vocab_size, self.embedding_dim)
        )

    def _sigmoid(self, x):
        """
        Sigmoid activation function: 1 / (1 + exp(-x))

        Used for binary classification (positive vs negative samples).
        Clip values to prevent numerical overflow.
        """
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-x))

    def _get_negative_samples(self, target_idx, num_samples):
        """
        Sample negative examples (words that are NOT in the context).

        Args:
            target_idx (int): Index of target word to exclude
            num_samples (int): Number of negative samples to draw

        Returns:
            list: Indices of negative samples
        """
        negative_samples = []
        while len(negative_samples) < num_samples:
            # Sample random index from word table
            neg_idx = self.word_table[np.random.randint(len(self.word_table))]
            # Avoid sampling the target word itself
            if neg_idx != target_idx:
                negative_samples.append(neg_idx)
        return negative_samples

    def _train_pair(self, target_idx, context_idx):
        """
        Train on a single target-context word pair using negative sampling.

        This is the core learning step. For each positive (target, context) pair:
        1. We want to maximize the similarity between W[target] and W_context[context]
        2. We sample random negative words and minimize their similarity

        Args:
            target_idx (int): Index of target word
            context_idx (int): Index of context word

        Returns:
            float: Loss for this pair
        """
        # Get target word embedding
        target_vec = self.W[target_idx]  # Shape: (embedding_dim,)

        # Positive sample - we want to maximize this
        # This is equivalent to minimizing negative log probability
        pos_vec = self.W_context[context_idx]
        pos_score = np.dot(target_vec, pos_vec)
        pos_pred = self._sigmoid(pos_score)

        # Gradient for positive sample
        # We want pos_pred to be close to 1
        pos_grad = (pos_pred - 1.0) * pos_vec

        # Accumulate loss
        loss = -np.log(pos_pred + 1e-8)

        # Negative samples - we want to minimize these
        neg_samples = self._get_negative_samples(target_idx, self.negative_samples)
        neg_grad = np.zeros_like(target_vec)

        for neg_idx in neg_samples:
            neg_vec = self.W_context[neg_idx]
            neg_score = np.dot(target_vec, neg_vec)
            neg_pred = self._sigmoid(neg_score)

            # Gradient for negative sample
            # We want neg_pred to be close to 0
            neg_grad += neg_pred * neg_vec

            # Accumulate loss
            loss += -np.log(1.0 - neg_pred + 1e-8)

        # Update target word embedding
        total_grad = pos_grad + neg_grad
        self.W[target_idx] -= self.learning_rate * total_grad

        return loss

    def train(self, sentences, epochs=5, verbose=True):
        """
        Train the Word2Vec model on sentences.

        Args:
            sentences (list): List of tokenized sentences (list of word lists)
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
        """
        # Build vocabulary
        self._build_vocab(sentences)

        # Initialize embeddings
        self._init_embeddings()

        # Build negative sampling table
        self._build_negative_sampling_table()

        total_pairs = 0

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_pairs = 0

            # Process each sentence
            for sentence in sentences:
                # Convert words to indices
                indexed_sentence = [self.word2idx[word] for word in sentence
                                   if word in self.word2idx]

                # Skip if sentence is too short
                if len(indexed_sentence) <= 1:
                    continue

                # For each word in the sentence
                for target_pos, target_idx in enumerate(indexed_sentence):
                    # Define context window (randomly sample window size)
                    window = np.random.randint(1, self.window_size + 1)

                    context_start = max(0, target_pos - window)
                    context_end = min(len(indexed_sentence), target_pos + window + 1)

                    # Train on each context word
                    for context_pos in range(context_start, context_end):
                        if context_pos != target_pos:
                            context_idx = indexed_sentence[context_pos]
                            loss = self._train_pair(target_idx, context_idx)
                            epoch_loss += loss
                            epoch_pairs += 1

            total_pairs += epoch_pairs
            if verbose:
                avg_loss = epoch_loss / (epoch_pairs + 1e-8)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Pairs trained: {epoch_pairs}")

        print(f"\nTraining complete! Total pairs trained: {total_pairs}")

    def get_vector(self, word):
        """
        Get the embedding vector for a word.

        Args:
            word (str): The word

        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.W[self.word2idx[word]]

    def most_similar(self, word, topn=5):
        """
        Find most similar words to the given word using cosine similarity.

        The intuition: Words that appear in similar contexts will have similar embeddings.

        Args:
            word (str): Query word
            topn (int): Return top N similar words

        Returns:
            list: List of (word, similarity_score) tuples
        """
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")

        # Get query word embedding
        word_vec = self.get_vector(word)

        # Normalize query vector
        word_vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-8)

        # Compute similarity with all words
        similarities = {}
        for vocab_word, idx in self.word2idx.items():
            if vocab_word != word:
                vocab_vec = self.W[idx]
                vocab_vec_norm = vocab_vec / (np.linalg.norm(vocab_vec) + 1e-8)

                # Cosine similarity
                similarity = np.dot(word_vec_norm, vocab_vec_norm)
                similarities[vocab_word] = similarity

        # Sort by similarity and return top N
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:topn]

    def analogy(self, positive_words, negative_words, topn=5):
        """
        Solve word analogies using vector arithmetic.

        Example: king - man + woman ≈ queen

        The intuition: Embeddings capture semantic relationships.
        If we have: embedding[king] - embedding[man] + embedding[woman]
        The result should be close to embedding[queen]

        Args:
            positive_words (list): Words to add (e.g., ['king', 'woman'])
            negative_words (list): Words to subtract (e.g., ['man'])
            topn (int): Return top N similar words

        Returns:
            list: List of (word, similarity_score) tuples
        """
        # Compute analogy vector
        analogy_vec = np.zeros(self.embedding_dim)

        for word in positive_words:
            if word not in self.word2idx:
                raise ValueError(f"Word '{word}' not in vocabulary")
            analogy_vec += self.get_vector(word)

        for word in negative_words:
            if word not in self.word2idx:
                raise ValueError(f"Word '{word}' not in vocabulary")
            analogy_vec -= self.get_vector(word)

        # Normalize
        analogy_vec_norm = analogy_vec / (np.linalg.norm(analogy_vec) + 1e-8)

        # Find most similar words
        excluded_words = set(positive_words + negative_words)
        similarities = {}

        for vocab_word, idx in self.word2idx.items():
            if vocab_word not in excluded_words:
                vocab_vec = self.W[idx]
                vocab_vec_norm = vocab_vec / (np.linalg.norm(vocab_vec) + 1e-8)

                similarity = np.dot(analogy_vec_norm, vocab_vec_norm)
                similarities[vocab_word] = similarity

        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:topn]

