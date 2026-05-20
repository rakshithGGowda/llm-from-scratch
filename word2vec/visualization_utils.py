"""
Visualization and Analysis Utilities for Word2Vec

This module provides utilities to visualize and analyze word embeddings.
"""

import numpy as np
import os

def get_embedding_stats(model):
    """
    Print statistics about the learned embeddings.

    Args:
        model: Trained Word2VecSkipGram model
    """
    print("\n--- EMBEDDING STATISTICS ---")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Embedding dimension: {model.embedding_dim}")

    # Compute statistics for each dimension
    all_embeddings = model.W

    print(f"\nEmbedding matrix shape: {all_embeddings.shape}")
    print(f"Mean embedding values: min={all_embeddings.min():.4f}, "
          f"max={all_embeddings.max():.4f}, mean={all_embeddings.mean():.4f}")

    # Vector lengths
    norms = np.linalg.norm(all_embeddings, axis=1)
    print(f"Word vector lengths: min={norms.min():.4f}, "
          f"max={norms.max():.4f}, mean={norms.mean():.4f}")

    # Similarity statistics
    similarities = []
    sample_size = min(100, model.vocab_size)
    for i in range(sample_size):
        vec_i = all_embeddings[i] / (np.linalg.norm(all_embeddings[i]) + 1e-8)
        for j in range(i+1, min(i+10, model.vocab_size)):
            vec_j = all_embeddings[j] / (np.linalg.norm(all_embeddings[j]) + 1e-8)
            sim = np.dot(vec_i, vec_j)
            similarities.append(sim)

    similarities = np.array(similarities)
    print(f"Similarity statistics (cosine): min={similarities.min():.4f}, "
          f"max={similarities.max():.4f}, mean={similarities.mean():.4f}")


def visualize_embeddings_2d(model, method='pca'):
    """
    Reduce embeddings to 2D and return coordinates for visualization.

    Args:
        model: Trained Word2VecSkipGram model
        method (str): 'pca' or 'tsne' (requires scikit-learn)

    Returns:
        dict: word -> (x, y) coordinates
    """
    embeddings = model.W
    words = list(model.idx2word.values())

    try:
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(embeddings)
            print(f"PCA explained variance: {pca.explained_variance_ratio_}")

        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
            coords_2d = tsne.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create mapping
        word_coords = {words[i]: (coords_2d[i, 0], coords_2d[i, 1])
                       for i in range(len(words))}

        return word_coords

    except ImportError:
        print(f"Error: {method} requires scikit-learn. Install with: pip install scikit-learn")
        return None


def plot_embeddings_text(model, word_coords, output_file='embeddings_2d.txt'):
    """
    Save 2D embedding visualization as ASCII text (for when matplotlib isn't available).

    Args:
        model: Trained Word2VecSkipGram model
        word_coords: dict of word -> (x, y) coordinates
        output_file: path to save the plot
    """
    if word_coords is None:
        print("No coordinates provided")
        return

    # Normalize coordinates to a grid
    x_coords = [coord[0] for coord in word_coords.values()]
    y_coords = [coord[1] for coord in word_coords.values()]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Map to grid (80x20)
    width, height = 120, 30
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    for word, (x, y) in word_coords.items():
        # Normalize to grid
        grid_x = int((x - x_min) / (x_max - x_min + 1e-8) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min + 1e-8) * (height - 1))

        # Clip to bounds
        grid_x = max(0, min(width - len(word), grid_x))
        grid_y = max(0, min(height - 1, grid_y))

        # Place word
        if len(word) <= width - grid_x:
            for i, char in enumerate(word):
                if grid_x + i < width:
                    grid[grid_y][grid_x + i] = char

    # Print and save
    output = '\n'.join(''.join(row) for row in grid)
    print("\n2D Embedding Visualization (ASCII):")
    print(output)

    with open(output_file, 'w') as f:
        f.write(output)
    print(f"\nVisualization saved to: {output_file}")


def analyze_relationships(model, words_to_analyze):
    """
    Analyze semantic relationships between words.

    Args:
        model: Trained Word2VecSkipGram model
        words_to_analyze: list of words to analyze
    """
    print("\n--- SEMANTIC RELATIONSHIP ANALYSIS ---")

    for word in words_to_analyze:
        if word not in model.word2idx:
            print(f"'{word}' not in vocabulary")
            continue

        print(f"\n{word.upper()}:")

        # Get vector
        vec = model.get_vector(word)
        print(f"  Vector length: {np.linalg.norm(vec):.4f}")

        # Most similar words
        similar = model.most_similar(word, topn=5)
        print(f"  Most similar words:")
        for sim_word, score in similar:
            print(f"    · {sim_word}: {score:.4f}")


def compare_embeddings(model, word1, word2):
    """
    Compare two word embeddings in detail.

    Args:
        model: Trained Word2VecSkipGram model
        word1, word2: words to compare
    """
    print(f"\n--- COMPARING '{word1}' vs '{word2}' ---")

    if word1 not in model.word2idx or word2 not in model.word2idx:
        print("One or both words not in vocabulary")
        return

    vec1 = model.get_vector(word1)
    vec2 = model.get_vector(word2)

    # Normalize
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    # Cosine similarity
    cos_sim = np.dot(vec1_norm, vec2_norm)
    print(f"Cosine similarity: {cos_sim:.4f}")

    # Euclidean distance
    euclidean = np.linalg.norm(vec1 - vec2)
    print(f"Euclidean distance: {euclidean:.4f}")

    # Dot product
    dot_product = np.dot(vec1, vec2)
    print(f"Dot product: {dot_product:.4f}")

    # Common similar words
    sim1 = set([w for w, _ in model.most_similar(word1, topn=10)])
    sim2 = set([w for w, _ in model.most_similar(word2, topn=10)])
    common = sim1 & sim2
    print(f"Common similar words: {common if common else 'None'}")


def export_embeddings(model, output_file='embeddings.txt'):
    """
    Export embeddings in word2vec format for use with other tools.

    Format:
    <word> <value1> <value2> ... <valueN>

    Args:
        model: Trained Word2VecSkipGram model
        output_file: path to save embeddings
    """
    print(f"Exporting embeddings to {output_file}...")

    with open(output_file, 'w') as f:
        f.write(f"{model.vocab_size} {model.embedding_dim}\n")

        for word, idx in model.word2idx.items():
            vec = model.W[idx]
            vec_str = ' '.join([f"{v:.6f}" for v in vec])
            f.write(f"{word} {vec_str}\n")

    print(f"Exported {model.vocab_size} embeddings!")


def load_embeddings(model, input_file):
    """
    Load pre-saved embeddings (inverse of export_embeddings).

    Args:
        model: Word2VecSkipGram model instance
        input_file: path to embedding file
    """
    print(f"Loading embeddings from {input_file}...")

    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        vocab_size, embedding_dim = map(int, first_line.split())

        model.vocab_size = vocab_size
        model.embedding_dim = embedding_dim
        model.W = np.zeros((vocab_size, embedding_dim))
        model.word2idx = {}
        model.idx2word = {}

        for idx, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])

            model.word2idx[word] = idx
            model.idx2word[idx] = word
            model.W[idx] = vec

    print(f"Loaded {vocab_size} embeddings!")


# ============================================================================
# VISUAL HIV TEST - Check if matplotlib is available for better visualizations
# ============================================================================

def try_import_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        return False


if try_import_matplotlib():
    import matplotlib.pyplot as plt

    def plot_embeddings_matplotlib(model, word_coords, top_words=30,
                                   output_file='embeddings_2d.png'):
        """
        Create a better visualization using matplotlib.

        Args:
            model: Trained Word2VecSkipGram model
            word_coords: dict of word -> (x, y) coordinates
            top_words: number of high-frequency words to label
            output_file: path to save the plot
        """
        if word_coords is None:
            print("No coordinates provided")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        x_coords = np.array([coord[0] for coord in word_coords.values()])
        y_coords = np.array([coord[1] for coord in word_coords.values()])

        # Plot all points
        ax.scatter(x_coords, y_coords, alpha=0.5, s=20, c='blue')

        # Label top words
        words = list(word_coords.keys())
        top_word_indices = np.argsort([model.word_freq.get(w, 0) for w in words])[-top_words:]

        for idx in top_word_indices:
            word = words[idx]
            x, y = word_coords[word]
            ax.annotate(word, (x, y), fontsize=9, alpha=0.8)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Word2Vec Embeddings (2D Visualization)')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        plt.close()

