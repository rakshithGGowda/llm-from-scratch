# Tokenizer - Building Text Tokenizers from Scratch

Welcome to the Tokenizer project! This project demonstrates how to build different types of text tokenizers from scratch, including character-level, word-level, and subword-level tokenization approaches.

## Overview

Tokenization is a fundamental step in natural language processing (NLP). It breaks down text into smaller, meaningful units called tokens. Different tokenization strategies are suited for different applications, and this project explores three main approaches:

### 1. Character-Level Tokenization
- Breaks text into individual characters
- **Pros**: Handles misspellings, no OOV problem, small vocabulary
- **Cons**: Longer sequences, harder for model to learn semantics
- **Best for**: Morphologically rich languages, character-based tasks

### 2. Word-Level Tokenization
- Breaks text into words (whitespace-separated tokens)
- **Pros**: Shorter sequences, faster processing, human-intuitive
- **Cons**: Large vocabulary, OOV problem, language-specific
- **Best for**: Traditional NLP tasks, well-formed text

### 3. Subword-Level Tokenization (BPE)
- Uses Byte Pair Encoding to break words into meaningful subword units
- **Pros**: Balances vocabulary size and sequence length, handles OOV
- **Cons**: More complex, may split words counterintuitively
- **Best for**: Modern transformers (GPT, BERT), multilingual models

## Project Structure

```
tokenizer/
├── __init__.py                    # Package initialization
├── tokenizer_scratch.py           # Core tokenizer implementations
├── quick_reference.py             # Quick examples and reference
├── tutorial.py                    # Comprehensive tutorial
└── docs/
    └── README.md                  # This file
```

## Installation

No external dependencies required! This project uses only Python standard library.

```bash
# Clone or download the project
cd tokenizer

# Run examples
python quick_reference.py

# Run comprehensive tutorial
python tutorial.py
```

## Usage Examples

### Basic Usage

```python
from tokenizer import WordTokenizer

# Create tokenizer
tokenizer = WordTokenizer()

# Build vocabulary from texts
texts = ["hello world", "machine learning", "deep learning"]
tokenizer.build_vocab(texts)

# Tokenize text
text = "hello world"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', 'world']

# Encode text (convert to indices)
encoded = tokenizer.encode(text)
print(encoded)  # [2, 3] (indices in vocabulary)

# Decode indices back to text
decoded = tokenizer.decode(encoded)
print(decoded)  # 'hello world'
```

### Character-Level Tokenization

```python
from tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer()
tokenizer.build_vocab(["hello", "world"])

tokens = tokenizer.tokenize("hello")
print(tokens)  # ['h', 'e', 'l', 'l', 'o']
```

### Subword Tokenization

```python
from tokenizer import SubwordTokenizer

tokenizer = SubwordTokenizer(num_merges=100)
tokenizer.build_vocab(["tokenization", "learning", "models"])

tokens = tokenizer.tokenize("tokenization")
print(tokens)  # Subword breakdown of 'tokenization'
```

## Key Concepts

### Vocabulary
A mapping from tokens (strings) to indices (integers). Used to convert text into numerical representations.

```python
tokenizer.vocab
# Example: {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, ...}
```

### Encoding
Converting text (strings) into token indices using the vocabulary.

### Decoding
Converting token indices back into text using the inverse vocabulary.

### Out-of-Vocabulary (OOV)
Tokens that don't exist in the vocabulary. Subword tokenizers handle this better by breaking unknown words into known subwords.

## Tokenizer Comparison

| Aspect | Character | Word | Subword |
|--------|-----------|------|---------|
| Vocabulary Size | Small (100s) | Large (10k-100k+) | Medium (1k-50k) |
| Sequence Length | Very Long | Short | Medium |
| OOV Problem | None | Severe | Minimal |
| Training Speed | Slow | Fast | Medium |
| Semantic Compression | Poor | Excellent | Very Good |
| Handles Morphology | Well | Poorly | Very Well |

## When to Use Which Tokenizer

### Character-Level:
- Spelling correction systems
- Character-aware language models
- Tasks where character patterns are important

### Word-Level:
- Simple NLP tasks with fixed vocabulary
- When vocabulary is available and manageable
- Fast processing is critical

### Subword-Level:
- Modern deep learning models (Transformers)
- Multilingual NLP
- Open-ended text generation
- When vocabulary coverage is uncertain

## Algorithm Details

### Byte Pair Encoding (BPE) Process

1. **Initialization**: Start with character-level tokens
2. **Count Frequencies**: Find all adjacent token pairs and their frequencies
3. **Merge Most Frequent**: Merge the most frequent pair into a new token
4. **Repeat**: Steps 2-3 until reaching desired vocabulary size

### Example BPE Execution

```
Initial:  l o w </w>  l o w e r </w>
Step 1:   lo w </w>   lo w e r </w>          (merge 'l o')
Step 2:   low </w>    low e r </w>          (merge 'w </w>')
Step 3:   low </w>    low er </w>           (merge 'e r')
Final:    low </w>    lower </w>            (3 tokens instead of 7)
```

## Implementation Details

### Base Class: `TokenizerBase`
Abstract base class with common interface:
- `tokenize(text)`: Break text into tokens
- `encode(text)`: Convert text to indices
- `decode(indices)`: Convert indices to text
- `build_vocab(texts)`: Build vocabulary from texts

### Features
- Automatic vocabulary building from text corpora
- Support for special tokens (`<pad>`, `<unk>`)
- Case normalization options
- Minimum frequency filtering for rare tokens

## Performance Considerations

### Memory Usage
- Character-level: O(1) for vocabulary, O(n) for encoding (n = text length)
- Word-level: O(v) for vocabulary (v = vocabulary size), O(w) for encoding (w = word count)
- Subword-level: O(v) for vocabulary, O(m) for encoding (m = subword count)

### Speed
- Character-level: Slower due to longer sequences
- Word-level: Fastest, but limited coverage
- Subword-level: Balance between speed and coverage

## Common Issues and Solutions

### Problem: OOV (Out-of-Vocabulary) Tokens
- **Word-Level**: Map to `<unk>` token
- **Subword-Level**: Automatically handled by breaking into subwords

### Problem: Language Dependencies
- **Character-Level**: Language-agnostic
- **Word-Level**: Requires language-specific rules; different for English, Chinese, etc.
- **Subword-Level**: Mostly language-agnostic; works across languages

### Problem: Rare Words
- **Solution**: Set minimum frequency threshold when building vocabulary
- Rare words get mapped to `<unk>` token

## Running the Examples

### Quick Reference
See basic examples of each tokenizer:
```bash
python quick_reference.py
```

### Comprehensive Tutorial
Detailed explanations and guided learning:
```bash
python tutorial.py
```

## Learning Objectives

After working with this project, you should understand:

1. ✅ Fundamental concepts of tokenization
2. ✅ Differences between tokenization strategies
3. ✅ How to implement tokenizers from scratch
4. ✅ Trade-offs between vocabulary size and sequence length
5. ✅ How to handle out-of-vocabulary words
6. ✅ Practical applications of different tokenization methods
7. ✅ Why modern NLP uses subword tokenization

## Further Reading

- **Byte Pair Encoding**: [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)
- **WordPiece**: Used in BERT and other transformers
- **SentencePiece**: Language-independent alternative to BPE
- **Subword NMT**: Neural machine translation with subword units

## Real-World Applications

- **BERT**: Uses WordPiece tokenization with 30k vocabulary
- **GPT-2/3**: Uses Byte Pair Encoding with 50k vocabulary
- **Multilingual Models**: Use language-agnostic subword tokenization
- **Domain-Specific Models**: Use custom subword vocabularies

## Contributing

This project is for educational purposes. Feel free to:
- Extend with new tokenization algorithms
- Add support for different languages
- Implement additional BPE features
- Add benchmarking capabilities

## License

Educational project - Use freely for learning purposes.

---

Happy tokenizing! Start with `quick_reference.py` for quick examples or `tutorial.py` for deeper understanding.

