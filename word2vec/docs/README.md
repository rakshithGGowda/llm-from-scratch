# Word2Vec from Scratch - Learning Guide

A comprehensive implementation and tutorial for understanding **Word2Vec** (Skip-gram model) from first principles.

## 📚 What is Word2Vec?

Word2Vec is an algorithm that learns **dense vector representations (embeddings) of words**. Instead of treating words as discrete symbols, it maps them to continuous vectors where **semantically similar words have similar vectors**.

### Key Insight: Distributional Hypothesis
> "Words are known by the company they keep" - J.R. Firth (1957)

Words that appear in similar contexts have similar meanings. If "dog" and "puppy" appear near the same words, their embeddings will be similar.

## 🎯 Why Learn Word2Vec?

1. **Foundation for NLP**: Word2Vec is fundamental to understanding modern NLP
2. **Practical Applications**: Used in search, recommendation systems, and language understanding
3. **Deep Learning**: Embeddings are the input layer for neural networks
4. **Machine Learning**: Better feature representation than one-hot encoding
5. **Transfer Learning**: Pre-trained embeddings can be used across tasks

## 📋 Project Structure

```
llm-from-scratch/
├── word2vec_scratch.py       # Main implementation
├── tutorial.py               # Beginner's guide with examples
├── advanced_examples.py      # Real-world examples
├── visualization_utils.py    # Analysis and visualization tools
└── README.md                 # This file
```

## 🚀 Getting Started

### Installation

No special dependencies needed for the basic implementation! Uses only NumPy.

```bash
pip install numpy
```

**Optional** (for visualization and dimensionality reduction):
```bash
pip install scikit-learn matplotlib
```

### Quick Start

1. **Start with the tutorial**:
```bash
python tutorial.py
```

This will train a simple Word2Vec model and show you:
- How embeddings work
- How to find similar words
- How word analogies work

2. **Try advanced examples**:
```bash
python advanced_examples.py
```

This demonstrates:
- Training on realistic data
- Custom training loops
- Hyperparameter effects
- Visualization

## 🧠 Understanding the Algorithm

### Skip-gram Model

The Skip-gram model learns to predict **context words from a target word**.

```
Input: target word
↓
Neural Network (with embedding layer)
↓
Output: probability of context words
```

**Example**:
- Sentence: "The cat sat on the mat"
- Target word: "cat" (position 2)
- Context words: "the", "sat" (within window)

The model learns: cat → likely to see "sat" nearby

### Training Process

1. **Initialize** embeddings randomly
2. **For each word pair** (target, context):
   - Forward pass: compute similarity
   - Loss computation: how well do we predict context?
   - Backward pass: update embeddings via gradient descent
3. **Repeat** over many epochs

### Negative Sampling

Training on all word pairs is expensive. **Negative sampling** is a trick:

- For each positive pair (target, context), sample random "negative" words
- Learn to distinguish positive from negative
- Much faster training!

```
Positive sample:  (target, context) → label=1
Negative samples: (target, random)  → label=0
```

## 📖 Class Reference

### Word2VecSkipGram

Main class for training Word2Vec models.

```python
from word2vec.word2vec_scratch import Word2VecSkipGram

# Create model
model = Word2VecSkipGram(
    embedding_dim=100,  # Dimension of word vectors
    window_size=5,  # Context window size
    negative_samples=5,  # Negative samples per positive
    learning_rate=0.01,  # Gradient descent step size
    min_count=2  # Minimum word frequency
)

# Train
sentences = [
    ['the', 'cat', 'sat'],
    ['the', 'dog', 'ran'],
]
model.train(sentences, epochs=5)

# Use embeddings
vector = model.get_vector('cat')  # Get word vector
similar = model.most_similar('cat', topn=5)  # Find similar words
analogy = model.analogy(['king', 'woman'],  # Solve analogies
                        ['man'])
```

### Parameters Explained

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `embedding_dim` | 100 | Dimensions of word vectors (higher = more expressiveness) |
| `window_size` | 5 | How many words left/right to consider as context |
| `negative_samples` | 5 | Number of negative samples per positive pair |
| `learning_rate` | 0.01 | Step size for gradient descent (higher = faster learning, but less stable) |
| `min_count` | 2 | Ignore words appearing less than this many times |

### Methods

#### `train(sentences, epochs=5, verbose=True)`
Train the model on sentences.

```python
sentences = [
    ['hello', 'world'],
    ['goodbye', 'world'],
]
model.train(sentences, epochs=10)
```

#### `get_vector(word)`
Get the embedding vector for a word.

```python
vec = model.get_vector('cat')  # Returns numpy array of shape (embedding_dim,)
```

#### `most_similar(word, topn=5)`
Find the most similar words using cosine similarity.

```python
similar_words = model.most_similar('cat')
# Returns: [('dog', 0.8234), ('kitten', 0.7892), ...]
```

#### `analogy(positive_words, negative_words, topn=5)`
Solve word analogies using vector arithmetic.

```python
# Find: king - man + woman ≈ ?
result = model.analogy(['king', 'woman'], ['man'])
# The result should be close to 'queen'
```

## 🔧 Customization Examples

### Example 1: Training on Your Own Text

```python
from word2vec.word2vec_scratch import Word2VecSkipGram

# Load your text
with open('mytext.txt', 'r') as f:
    text = f.read()

# Tokenize (simple version)
sentences = [line.lower().split() for line in text.split('\n')]

# Create and train model
model = Word2VecSkipGram(embedding_dim=200, window_size=5)
model.train(sentences, epochs=50)

# Explore results
print(model.most_similar('machine'))
```

### Example 2: Training on Specific Domain

```python
# Medical text
medical_text = [
    "patient has symptoms of fever",
    "doctor prescribes antibiotics",
    "hospital provides medical care",
]

sentences = [t.split() for t in medical_text]
model = Word2VecSkipGram(embedding_dim=100)
model.train(sentences, epochs=100)

# Now embeddings capture medical domain relationships
```

### Example 3: Analysis and Visualization

```python
from word2vec.visualization_utils import (
    get_embedding_stats, analyze_relationships,
    visualize_embeddings_2d, plot_embeddings_text
)

# Get statistics
get_embedding_stats(model)

# Analyze specific words
analyze_relationships(model, ['cat', 'dog', 'animal'])

# Visualize in 2D (requires scikit-learn)
coords = visualize_embeddings_2d(model, method='pca')
plot_embeddings_text(model, coords)
```

## 📊 Understanding the Results

### What Good Embeddings Look Like

After training, you should see:

1. **Semantic clusters**: Similar words have similar vectors
   ```
   dog ≈ puppy (close similarity)
   dog ≠ car (low similarity)
   ```

2. **Meaningful relationships**:
   ```
   king - man + woman ≈ queen
   Paris - France + Germany ≈ Berlin
   ```

3. **Context-based similarity**:
   Words in similar contexts become similar
   ```
   "The cat sat on the mat" → cat, sat, mat should be somewhat similar
   "The dog sat on the log" → dog should be similar to cat
   ```

### Loss During Training

- **Decreasing loss** = model is learning
- Loss should decrease over epochs initially
- Only training on small data? Loss may fluctuate

```
Epoch 1, Loss: 2.4532
Epoch 2, Loss: 2.3212
Epoch 3, Loss: 2.1845
...
```

## 🎓 Learning Path

### Beginner (Start Here)
1. Run `tutorial.py`
2. Understand the Skip-gram concept
3. Experiment with `most_similar()`
4. Modify example sentences and retrain

### Intermediate
1. Read through `word2vec_scratch.py` code
2. Understand negative sampling in `_train_pair()`
3. Try different hyperparameters
4. Use `advanced_examples.py`

### Advanced
1. Implement modifications:
   - CBOW model (instead of Skip-gram)
   - Hierarchical softmax (instead of negative sampling)
   - Different negative sampling strategies
2. Train on large datasets
3. Compare with pre-trained embeddings (GloVe, FastText)
4. Integrate with neural networks

## 💡 Common Mistakes & Solutions

### Problem: Loss not decreasing
- **Solution**: Increase learning rate (0.025-0.05), use more epochs

### Problem: "Word not in vocabulary"
- **Solution**: Decrease `min_count` parameter, use more training data

### Problem: Similar words don't seem related
- **Solutions**:
  - Train for more epochs
  - Use more data
  - Increase embedding dimension
  - Larger window size for broader context

### Problem: Code running too slow
- **Solution**: Use smaller `embedding_dim`, reduce sentences, increase `min_count`

## 🔬 Hyperparameter Tuning Guide

| What to adjust | Effect |
|---|---|
| Increase `embedding_dim` | Richer representations but slower training |
| Increase `window_size` | Captures broader context but slower training |
| Increase `negative_samples` | Better gradients but slower training |
| Increase `learning_rate` | Faster convergence but less stable |
| Increase `epochs` | Better convergence but slower |
| Decrease `min_count` | More words in vocabulary but noisier |

**Golden rule**: Start with defaults, then change one at a time!

## 📚 Further Reading

- [Original Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Efficient Estimation of Word2Vec](https://arxiv.org/abs/1310.4546)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [FastText (Facebook's improvement)](https://fasttext.cc/)

## 🤔 How This Differs from Production Word2Vec

Our implementation is **educational and simplified**:

| Aspect | Ours | Production |
|--------|------|-----------|
| Architecture | Skip-gram only | Skip-gram + CBOW |
| Optimization | Negative sampling | Multiple options |
| Performance | Pure Python (slow) | C++/optimized |
| Features | Core algorithm | Many tuning options |
| Use case | Learning | Production systems |

Production systems (like Gensim's Word2Vec) are highly optimized for speed and incorporate years of improvements.

## 🎯 Next Challenges

1. **Implement CBOW** (Continuous Bag of Words) model
2. **Add hierarchical softmax** as alternative to negative sampling
3. **Train on Wikipedia dump** for better results
4. **Compare with pre-trained embeddings**
5. **Integrate into a downstream NLP task** (sentiment analysis, clustering)

## ❓ Troubleshooting

### Import errors?
```bash
pip install numpy
```

### Scikit-learn not found?
```bash
pip install scikit-learn matplotlib
```

### Still having issues?
Check that you're running Python 3.6+:
```bash
python --version
```

## 📝 License

This educational implementation is provided as-is for learning purposes.

## 🙏 Credits

Implementation created for understanding Word2Vec from first principles.

---

**Happy Learning! 🚀**

Start with `python tutorial.py` to begin your Word2Vec journey!

