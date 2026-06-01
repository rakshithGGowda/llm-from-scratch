"""
Tokenizer implementations from scratch.
Demonstrating various text tokenization approaches.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from collections import Counter, defaultdict


class TokenizerBase(ABC):
    """Abstract base class for tokenizers."""

    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize tokenizer.

        Args:
            vocab: Optional vocabulary dictionary mapping tokens to indices
        """
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()} if vocab else {}

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into token indices."""
        pass

    @abstractmethod
    def decode(self, indices: List[int]) -> str:
        """Decode token indices back to text."""
        pass

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from texts."""
        pass


class CharacterTokenizer(TokenizerBase):
    """Tokenizer that splits text into individual characters."""

    def __init__(self, vocab: Dict[str, int] = None):
        """Initialize character tokenizer."""
        super().__init__(vocab)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters.

        Args:
            text: Input text

        Returns:
            List of character tokens
        """
        # Convert to lowercase and split into characters
        return list(text.lower())

    def encode(self, text: str) -> List[int]:
        """
        Encode text into character indices.

        Args:
            text: Input text

        Returns:
            List of character indices
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, 0) for token in tokens]

    def decode(self, indices: List[int]) -> str:
        """
        Decode character indices back to text.

        Args:
            indices: List of character indices

        Returns:
            Decoded text
        """
        return "".join([self.inv_vocab.get(idx, "") for idx in indices])

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of input texts
            min_freq: Minimum frequency for a character to be included
        """
        char_freq = Counter()
        for text in texts:
            char_freq.update(text.lower())

        # Add special tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2

        for char, freq in char_freq.most_common():
            if freq >= min_freq:
                self.vocab[char] = idx
                idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}


class WordTokenizer(TokenizerBase):
    """Tokenizer that splits text into words."""

    def __init__(self, vocab: Dict[str, int] = None, lowercase: bool = True):
        """
        Initialize word tokenizer.

        Args:
            vocab: Optional vocabulary dictionary
            lowercase: Whether to convert text to lowercase
        """
        super().__init__(vocab)
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        if self.lowercase:
            text = text.lower()

        # Remove special characters and split by whitespace
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def encode(self, text: str) -> List[int]:
        """
        Encode text into word indices.

        Args:
            text: Input text

        Returns:
            List of word indices
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get("<unk>", 0)) for token in tokens]

    def decode(self, indices: List[int]) -> str:
        """
        Decode word indices back to text.

        Args:
            indices: List of word indices

        Returns:
            Decoded text
        """
        words = [self.inv_vocab.get(idx, "") for idx in indices]
        return " ".join(words)

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of input texts
            min_freq: Minimum frequency for a word to be included
        """
        word_freq = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_freq.update(tokens)

        # Add special tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2

        for word, freq in word_freq.most_common():
            if freq >= min_freq:
                self.vocab[word] = idx
                idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}


class SubwordTokenizer(TokenizerBase):
    """
    Simple subword tokenizer using Byte Pair Encoding (BPE) principles.
    """

    def __init__(self, vocab: Dict[str, int] = None, num_merges: int = 100):
        """
        Initialize subword tokenizer.

        Args:
            vocab: Optional vocabulary dictionary
            num_merges: Number of BPE merge operations
        """
        super().__init__(vocab)
        self.num_merges = num_merges
        self.merges = {}

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
        """Get frequency of adjacent token pairs."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[symbols[i:i+2]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str],
                     v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge a pair of tokens in vocabulary."""
        v_out = {}
        bigram = pair
        replacement = "".join(pair)

        for word in v_in:
            new_word = tuple(replacement if word[i:i+2] == bigram
                           else word[i]
                           for i in range(len(word) - 1)) + (word[-1],)
            v_out[new_word] = v_in[word]

        return v_out

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.

        Args:
            text: Input text

        Returns:
            List of subword tokens
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # Character-level tokenization as base
        words = text.split()
        return [char for word in words for char in word]

    def encode(self, text: str) -> List[int]:
        """
        Encode text into subword token indices.

        Args:
            text: Input text

        Returns:
            List of subword token indices
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get("<unk>", 0)) for token in tokens]

    def decode(self, indices: List[int]) -> str:
        """
        Decode subword token indices back to text.

        Args:
            indices: List of subword token indices

        Returns:
            Decoded text
        """
        tokens = [self.inv_vocab.get(idx, "") for idx in indices]
        return "".join(tokens).replace("##", "")

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary using BPE-like approach.

        Args:
            texts: List of input texts
            min_freq: Minimum frequency threshold
        """
        # Initialize with character-level vocabulary
        word_freqs = defaultdict(int)
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            for word in text.split():
                word_freqs[" ".join(list(word)) + " </w>"] += 1

        # Add special tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2

        # Add initial character vocabulary
        chars = set()
        for word in word_freqs:
            chars.update(word.split())

        for char in sorted(chars):
            self.vocab[char] = idx
            idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

