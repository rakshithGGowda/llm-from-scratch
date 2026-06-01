"""
Comprehensive tutorial for tokenization concepts and implementation.
Demonstrates the importance of tokenization in NLP and explores
different tokenization strategies from scratch.
"""

from tokenizer_scratch import (
    CharacterTokenizer,
    WordTokenizer,
    SubwordTokenizer,
)


class TokenizerTutorial:
    """Tutorial demonstrating tokenization concepts."""

    @staticmethod
    def introduction():
        """Introduction to tokenization."""
        print("""
TOKENIZATION TUTORIAL
=====================

What is Tokenization?
---------------------
Tokenization is the process of breaking down text into smaller units called tokens.
These tokens are the building blocks for natural language processing models.

Why is Tokenization Important?
--------------------------------
1. Models work with discrete units (tokens), not continuous text
2. Affects model performance and efficiency
3. Influences vocabulary size and memory requirements
4. Different tasks may require different tokenization strategies

Types of Tokenization:
----------------------
1. Character-level: Break text into individual characters
2. Word-level: Break text into words
3. Subword-level: Break text into meaningful subword units (BPE, WordPiece, etc.)
        """)

    @staticmethod
    def character_level_explanation():
        """Detailed explanation of character-level tokenization."""
        print("\n" + "=" * 70)
        print("1. CHARACTER-LEVEL TOKENIZATION")
        print("=" * 70)
        print("""
Pros:
- Handles misspellings naturally
- No out-of-vocabulary (OOV) problem for characters
- Small vocabulary size
- Can generate any text

Cons:
- Longer sequences mean slower training
- Harder for model to learn long-range dependencies
- May require deeper networks to capture word semantics

Use Cases:
- Language modeling with character boundaries
- Spelling correction systems
- Character-based neural machine translation
        """)

        # Example
        print("\nCHARACTER-LEVEL EXAMPLE:")
        print("-" * 70)

        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello world", "machine learning", "deep learning"])

        text = "hello"
        tokens = tokenizer.tokenize(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Encoding: {tokenizer.encode(text)}")

    @staticmethod
    def word_level_explanation():
        """Detailed explanation of word-level tokenization."""
        print("\n" + "=" * 70)
        print("2. WORD-LEVEL TOKENIZATION")
        print("=" * 70)
        print("""
Pros:
- Shorter sequences (fewer tokens per sentence)
- Faster training and inference
- Natural semantic units
- Matches human intuition

Cons:
- Large vocabulary size for large datasets
- Out-of-vocabulary (OOV) problem for rare/unseen words
- Difficulty handling compound words or inflections
- Language-specific (requires language knowledge for good tokenization)

Use Cases:
- Most traditional NLP tasks
- Machine translation
- Text classification
- Named entity recognition

Challenges:
- How to handle contractions? (don't → do + n't or don + t)
- Punctuation? (New York. vs New York)
- Multi-word expressions? (New York City)
        """)

        # Example
        print("\nWORD-LEVEL EXAMPLE:")
        print("-" * 70)

        tokenizer = WordTokenizer()
        texts = ["hello world", "machine learning is great", "deep learning models"]
        tokenizer.build_vocab(texts)

        text = "machine learning"
        tokens = tokenizer.tokenize(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        print(f"Encoding: {tokenizer.encode(text)}")

    @staticmethod
    def subword_level_explanation():
        """Detailed explanation of subword-level tokenization."""
        print("\n" + "=" * 70)
        print("3. SUBWORD-LEVEL TOKENIZATION (BPE)")
        print("=" * 70)
        print("""
Subword Tokenization Algorithms:
- Byte Pair Encoding (BPE)
- WordPiece (used in BERT)
- SentencePiece
- Unigram Language Model

Byte Pair Encoding (BPE):
1. Start with character-level vocabulary
2. Iteratively merge most frequent adjacent token pairs
3. Repeat until vocabulary reaches target size
4. Result: Balance between word and character tokens

Pros:
- Handles OOV words by breaking into known subwords
- Smaller vocabulary than word-level (typically 30k-50k)
- Captures both word semantics and morphology
- Works for multiple languages

Cons:
- More complex than word or character tokenization
- May split words in unintuitive ways
- Requires storing merge operations for encoding new text

Use Cases:
- Modern transformer models (GPT, BERT)
- Multilingual models
- Domain-specific models without full vocabulary coverage

How BPE Handles Unknown Words:
Input: "playing" (if not in vocabulary)
If "play" and "ing" are in vocabulary:
Output: ["play", "ing"]
        """)

        # Example
        print("\nSUBWORD-LEVEL EXAMPLE:")
        print("-" * 70)

        tokenizer = SubwordTokenizer(num_merges=50)
        texts = ["playing games", "learning models", "hello world"]
        tokenizer.build_vocab(texts)

        text = "hello"
        tokens = tokenizer.tokenize(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        print(f"Encoding: {tokenizer.encode(text)}")

    @staticmethod
    def practical_comparison():
        """Provide practical comparison of tokenizers."""
        print("\n" + "=" * 70)
        print("PRACTICAL COMPARISON")
        print("=" * 70)

        text = "preprocessing tokenization"
        texts = [text, "machine learning", "deep learning", "natural language processing"]

        print(f"\nText to tokenize: '{text}'")
        print(f"Vocabulary texts: {texts}\n")

        # Character tokenization
        char_tok = CharacterTokenizer()
        char_tok.build_vocab(texts)
        char_tokens = char_tok.tokenize(text)
        print(f"CHARACTER TOKENIZATION ({len(char_tokens)} tokens):")
        print(f"  Tokens: {char_tokens}")
        print(f"  Vocabulary size: {len(char_tok.vocab)}\n")

        # Word tokenization
        word_tok = WordTokenizer()
        word_tok.build_vocab(texts)
        word_tokens = word_tok.tokenize(text)
        print(f"WORD TOKENIZATION ({len(word_tokens)} tokens):")
        print(f"  Tokens: {word_tokens}")
        print(f"  Vocabulary size: {len(word_tok.vocab)}\n")

        # Subword tokenization
        subword_tok = SubwordTokenizer(num_merges=50)
        subword_tok.build_vocab(texts)
        subword_tokens = subword_tok.tokenize(text)
        print(f"SUBWORD TOKENIZATION ({len(subword_tokens)} tokens):")
        print(f"  Tokens: {subword_tokens}")
        print(f"  Vocabulary size: {len(subword_tok.vocab)}\n")

        print("Key Observations:")
        print(f"- Character tokenization requires {len(char_tok.vocab)} tokens to encode text")
        print(f"- Word tokenization requires {len(word_tok.vocab)} tokens to encode text")
        print(f"- Subword tokenization requires {len(subword_tok.vocab)} tokens to encode text")

    @staticmethod
    def run_all():
        """Run the complete tutorial."""
        TokenizerTutorial.introduction()
        TokenizerTutorial.character_level_explanation()
        TokenizerTutorial.word_level_explanation()
        TokenizerTutorial.subword_level_explanation()
        TokenizerTutorial.practical_comparison()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
Choosing the Right Tokenizer:

1. CHARACTER-LEVEL: Use when:
   - Working with morphologically rich languages
   - Need to handle lots of misspellings
   - Model capacity is high enough for longer sequences

2. WORD-LEVEL: Use when:
   - Working with well-formed text
   - Vocabulary is manageable (< 100k words)
   - Speed is critical and vocabulary is stable

3. SUBWORD-LEVEL: Use when:
   - Building modern NLP models (transformers)
   - Need to handle diverse vocabulary efficiently
   - Working with multiple languages
   - Want to balance vocabulary size and sequence length

For modern deep learning NLP:
→ Subword tokenization is typically the best choice!
        """)


if __name__ == "__main__":
    TokenizerTutorial.run_all()

