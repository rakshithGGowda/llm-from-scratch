"""
Word2Vec Quick Reference Guide

Copy-paste snippets and common operations.
"""

# ============================================================================
# QUICK START (5 minutes)
# ============================================================================

from word2vec_scratch import Word2VecSkipGram

# 1. Prepare sentences
sentences = [
    ['hello', 'world'],
    ['goodbye', 'cruel', 'world'],
    ['hello', 'friend'],
]

# 2. Create and train model
model = Word2VecSkipGram(embedding_dim=50, window_size=1)
model.train(sentences, epochs=100)

# 3. Use model
print(model.most_similar('world'))


# ============================================================================
# BASIC OPERATIONS
# ============================================================================

# Get embedding for a word
embedding = model.get_vector('world')
print(f"Shape: {embedding.shape}")
print(f"First 5 dims: {embedding[:5]}")

# Find similar words
similar = model.most_similar('world', topn=5)
for word, score in similar:
    print(f"{word}: {score:.4f}")

# Calculate similarity between two words
import numpy as np
vec1 = model.get_vector('world')
vec2 = model.get_vector('hello')
vec1_norm = vec1 / np.linalg.norm(vec1)
vec2_norm = vec2 / np.linalg.norm(vec2)
similarity = np.dot(vec1_norm, vec2_norm)
print(f"Similarity: {similarity:.4f}")


# ============================================================================
# TRAINING WITH REAL DATA
# ============================================================================

import os

# Option 1: From file
def train_from_file(filepath, window_size=5, embedding_dim=100, epochs=50):
    """Train on text file."""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.lower().split()
            if len(words) > 1:
                sentences.append(words)

    model = Word2VecSkipGram(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=2
    )
    model.train(sentences, epochs=epochs)
    return model

# usage:
# model = train_from_file('mytext.txt', embedding_dim=200, epochs=100)


# Option 2: From list of documents
def train_from_texts(texts, min_count=2):
    """Train on list of text strings."""
    sentences = [text.lower().split() for text in texts]

    model = Word2VecSkipGram(
        embedding_dim=100,
        window_size=5,
        min_count=min_count
    )
    model.train(sentences, epochs=50)
    return model

# usage:
# texts = ["your text here", "more text here"]
# model = train_from_texts(texts)


# ============================================================================
# ADVANCED OPERATIONS
# ============================================================================

# Word analogies (skip if words not in vocabulary)
# Example: king - man + woman = queen
try:
    analogy = model.analogy(
        positive_words=['learning', 'neural'],
        negative_words=['networks'],
        topn=5
    )
except:
    pass

# Export embeddings (word2vec format)
def export_embeddings(model, filepath):
    """Export embeddings for use with other tools."""
    with open(filepath, 'w') as f:
        f.write(f"{model.vocab_size} {model.embedding_dim}\n")
        for word, idx in model.word2idx.items():
            vec = model.W[idx]
            vec_str = ' '.join([f"{v:.6f}" for v in vec])
            f.write(f"{word} {vec_str}\n")

# usage:
# export_embeddings(model, 'embeddings.txt')


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# More training data? Use smaller embedding_dim and larger learning rate
model = Word2VecSkipGram(
    embedding_dim=50,        # Smaller
    window_size=2,
    negative_samples=3,      # Fewer negative samples
    learning_rate=0.05       # Higher
)

# Limited training data? Use larger embedding_dim and smaller learning rate
model = Word2VecSkipGram(
    embedding_dim=300,       # Larger
    window_size=10,
    negative_samples=15,     # More negative samples
    learning_rate=0.01       # Lower
)

# GPU simulation (not really GPU but organized training)
def large_scale_training(sentences, batch_size=1000):
    """Train in batches (for memory efficiency)."""
    model = Word2VecSkipGram(embedding_dim=200, window_size=5)

    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        model.train(batch, epochs=10, verbose=False)

    return model


# ============================================================================
# ANALYSIS & DEBUGGING
# ============================================================================

from visualization_utils import (
    get_embedding_stats,
    analyze_relationships,
    compare_embeddings
)

# Get model statistics
get_embedding_stats(model)

# Analyze specific words
analyze_relationships(model, ['world', 'hello', 'friend'])

# Compare two words
compare_embeddings(model, 'world', 'hello')


# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Pattern 1: Find all words similar to a topic
def get_topic_words(model, seed_word, depth=3):
    """Find related words recursively."""
    found = {seed_word}
    to_explore = [seed_word]

    for _ in range(depth):
        new_words = []
        for word in to_explore:
            try:
                similar = model.most_similar(word, topn=5)
                for w, _ in similar:
                    if w not in found:
                        found.add(w)
                        new_words.append(w)
            except:
                pass
        to_explore = new_words

    return found

# usage:
# religion_words = get_topic_words(model, 'prayer', depth=2)


# Pattern 2: Filter embeddings by similarity threshold
def filter_by_similarity(model, word, threshold=0.7):
    """Get all words above similarity threshold."""
    similar = model.most_similar(word, topn=model.vocab_size)
    filtered = [w for w, score in similar if score >= threshold]
    return filtered

# usage:
# highly_similar = filter_by_similarity(model, 'cat', 0.8)


# Pattern 3: Create word clusters
def cluster_words(model, words, num_clusters=3):
    """Simple clustering using embeddings."""
    from sklearn.cluster import KMeans

    vectors = np.array([model.get_vector(w) for w in words])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    clusters = {}
    for word, label in zip(words, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)

    return clusters

# usage:
# clusters = cluster_words(model, ['cat', 'dog', 'run', 'walk'])


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_model_quality(model):
    """Quick test to check model quality."""
    tests_passed = 0
    tests_total = 0

    # Test 1: Embeddings have correct shape
    tests_total += 1
    for word in list(model.word2idx.keys())[:5]:
        vec = model.get_vector(word)
        if vec.shape == (model.embedding_dim,):
            tests_passed += 1
        else:
            print(f"❌ Wrong shape for {word}")

    # Test 2: Similar words have high similarity
    tests_total += 1
    try:
        similar = model.most_similar(list(model.word2idx.keys())[0], topn=1)
        if similar and similar[0][1] > 0.3:
            tests_passed += 1
            print(f"✓ Similar words found: {similar}")
    except:
        print("❌ Could not find similar words")

    print(f"\nTests passed: {tests_passed}/{tests_total}")


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
Speed optimization:
1. Reduce embedding_dim (50-100 for quick tests)
2. Reduce window_size (2-3 for faster training)
3. Increase min_count (filter rare words)
4. Use fewer epochs initially (10-50)
5. Pre-process text to remove rare words

Memory optimization:
1. Use smaller embedding_dim
2. Process data in batches
3. Use float32 instead of float64

Better results:
1. More training data
2. More epochs (100-500)
3. Larger embedding_dim (200+)
4. Larger window_size
5. More negative samples
"""


# ============================================================================
# REAL-WORLD EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Quick Reference Examples\n")

    # Create sample data
    sample_data = [
        "machine learning is powerful",
        "deep learning uses neural networks",
        "neural networks learn from data",
        "artificial intelligence is the future",
        "learning algorithms improve with data",
    ]

    sentences = [text.lower().split() for text in sample_data]

    # Train model
    print("Training model...")
    model = Word2VecSkipGram(
        embedding_dim=50,
        window_size=2,
        negative_samples=5,
        learning_rate=0.025
    )

    model.train(sentences, epochs=100, verbose=False)
    print("✓ Training complete\n")

    # Test operations
    print("Testing model operations:")
    print(f"Vocabulary size: {model.vocab_size}\n")

    print("Similar to 'learning':")
    similar = model.most_similar('learning', topn=3)
    for word, score in similar:
        print(f"  {word}: {score:.4f}")

    print("\nTest complete!")


