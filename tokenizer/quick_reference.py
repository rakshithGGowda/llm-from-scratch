"""
Quick reference guide and examples for using different tokenizers.
"""

from tokenizer_scratch import (
    CharacterTokenizer,
    WordTokenizer,
    SubwordTokenizer,
)


def example_character_tokenizer():
    """Demonstrate character-level tokenization."""
    print("=" * 60)
    print("CHARACTER TOKENIZER")
    print("=" * 60)

    # Texts for building vocabulary
    texts = [
        "Hello world",
        "Tokenization is important",
        "The quick brown fox",
    ]

    # Create and build tokenizer
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.build_vocab(texts)

    print(f"Vocabulary size: {len(char_tokenizer.vocab)}")
    print(f"Sample vocab: {dict(list(char_tokenizer.vocab.items())[:10])}\n")

    # Tokenize example
    text = "hello world"
    tokens = char_tokenizer.tokenize(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}\n")

    # Encode/decode
    encoded = char_tokenizer.encode(text)
    print(f"Encoded: {encoded}")

    decoded = char_tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'\n")


def example_word_tokenizer():
    """Demonstrate word-level tokenization."""
    print("=" * 60)
    print("WORD TOKENIZER")
    print("=" * 60)

    texts = [
        "Hello world",
        "Tokenization is important",
        "The quick brown fox",
    ]

    # Create and build tokenizer
    word_tokenizer = WordTokenizer()
    word_tokenizer.build_vocab(texts)

    print(f"Vocabulary size: {len(word_tokenizer.vocab)}")
    print(f"Vocabulary: {word_tokenizer.vocab}\n")

    # Tokenize example
    text = "hello world tokenization"
    tokens = word_tokenizer.tokenize(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}\n")

    # Encode/decode
    encoded = word_tokenizer.encode(text)
    print(f"Encoded: {encoded}")

    decoded = word_tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'\n")


def example_subword_tokenizer():
    """Demonstrate subword tokenization."""
    print("=" * 60)
    print("SUBWORD TOKENIZER (BPE-like)")
    print("=" * 60)

    texts = [
        "Hello world",
        "Tokenization is important",
        "The quick brown fox",
    ]

    # Create and build tokenizer
    subword_tokenizer = SubwordTokenizer(num_merges=50)
    subword_tokenizer.build_vocab(texts)

    print(f"Vocabulary size: {len(subword_tokenizer.vocab)}")
    print(f"Sample vocab: {dict(list(subword_tokenizer.vocab.items())[:15])}\n")

    # Tokenize example
    text = "hello world"
    tokens = subword_tokenizer.tokenize(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}\n")

    # Encode/decode
    encoded = subword_tokenizer.encode(text)
    print(f"Encoded: {encoded}")

    decoded = subword_tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'\n")


def comparison_example():
    """Compare different tokenizers on the same text."""
    print("=" * 60)
    print("COMPARISON: Different Tokenization Strategies")
    print("=" * 60)

    text = "The quick brown fox jumps over the lazy dog"
    texts = [text, "Hello world", "Natural language processing"]

    # Character tokenizer
    char_tok = CharacterTokenizer()
    char_tok.build_vocab(texts)
    char_tokens = char_tok.tokenize(text)
    print(f"Character Tokens ({len(char_tokens)} tokens):")
    print(f"  {char_tokens}\n")

    # Word tokenizer
    word_tok = WordTokenizer()
    word_tok.build_vocab(texts)
    word_tokens = word_tok.tokenize(text)
    print(f"Word Tokens ({len(word_tokens)} tokens):")
    print(f"  {word_tokens}\n")

    # Subword tokenizer
    subword_tok = SubwordTokenizer()
    subword_tok.build_vocab(texts)
    subword_tokens = subword_tok.tokenize(text)
    print(f"Subword Tokens ({len(subword_tokens)} tokens):")
    print(f"  {subword_tokens}\n")


if __name__ == "__main__":
    example_character_tokenizer()
    print("\n")
    example_word_tokenizer()
    print("\n")
    example_subword_tokenizer()
    print("\n")
    comparison_example()

