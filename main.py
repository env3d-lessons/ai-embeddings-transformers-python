"""
main.py

This script demonstrates how to generate and visualize sentence embeddings using transformer models.
It includes:
- Loading a sentence-transformer model for feature extraction.
- Reducing high-dimensional word embeddings to 2D using PCA or t-SNE for visualization.
- Plotting selected word embeddings in 2D space.
- Calculating cosine similarity between embeddings.
- An interactive UI to compare user input with reference sentences.

Intended for teaching and exploring the structure of language model embeddings.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

def reduce_to_2d(embeddings, method='tsne', labels=None, save_path='embeddings_2d.png', show_labels=None, normalize_output=False):
    """
    Reduce high-dimensional embeddings to 2D using PCA or t-SNE and plot the result.
    
    Args:
        embeddings: list of lists of floats, shape (n_samples, n_features)
        method: 'pca' or 'tsne'
        labels: list of str, optional, labels for each point
        save_path: str, path to save the plot
        show_labels: list of str, optional, only these labels will be plotted and annotated
        normalize_output: bool, if True, normalize 2D output to unit vectors
    Returns:
        embeddings_2d: numpy array of shape (n_samples, 2)
    """
    n_samples = len(embeddings)
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'tsne':
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    if normalize_output:
        embeddings_2d = normalize(embeddings_2d, norm='l2')

    # Filter points to plot if show_labels is specified
    if labels is not None and show_labels is not None:
        indices = [i for i, label in enumerate(labels) if label in show_labels]
        plot_embeddings = embeddings_2d[indices]
        plot_labels = [labels[i] for i in indices]
    else:
        plot_embeddings = embeddings_2d
        plot_labels = labels if labels is not None else None

    # Plotting
    plt.figure(figsize=(8, 6))
    if plot_labels is not None:
        x, y = plot_embeddings[:, 0], plot_embeddings[:, 1]
        plt.scatter(x, y, c='blue')
        for i, label in enumerate(plot_labels):
            plt.annotate(label, (x[i], y[i]), fontsize=12, ha='right')
    else:
        x, y = plot_embeddings[:, 0], plot_embeddings[:, 1]
        plt.scatter(x, y, c='blue')
    plt.title(f'2D Embeddings ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return embeddings_2d

# Load the model and setup the embedder
print("Loading the model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = pipeline("feature-extraction", model=model_name)
print(f"Model '{model_name}' loaded successfully.")

# Test run: Create embeddings for each of the following words
labels = [
    'Dog', 'Cat', 'Mouse', 'King', 'Queen', 'Prince',
    'Lion', 'Tiger', 'Wolf', 'Bear', 'Fox', 'Rabbit',
    'Man', 'Woman', 'Boy', 'Girl', 'Father', 'Mother',
    'Car', 'Bus', 'Train', 'Plane', 'Ship', 'Bicycle',
    'Apple', 'Banana', 'Orange', 'Grape', 'Lemon', 'Peach',
    'Table', 'Chair', 'Sofa', 'Bed', 'Desk', 'Lamp',
    'Sun', 'Moon', 'Star', 'Sky', 'Cloud', 'Rain',
    'Doctor', 'Nurse', 'Teacher', 'Student', 'Lawyer', 'Engineer'
]

# Visualize the embeddings of the words in 2D space just so you can see it.
# NOTE: This is known as dimensionality reduction, and is a common technique to visualize high-dimensional data.
#       But keep in mind that the 2D representation may not capture all the nuances of the original high-dimensional embeddings.
filename = 'embeddings_2d.png'
reduce_to_2d(
    np.array([ embedder(word)[0][0] for word in labels ]),
    labels=labels,
    method='tsne',
    normalize_output=False,
    show_labels=['Dog','Cat','King','Queen','Apple','Banana'],
    save_path='embeddings_2d.png'
)
print(f"Creating visual of simple embeddings... {','.join(labels)}.  Saved to {filename}")

def cosine_similarity(vec1, vec2):
    # Normalize the vectors
    vec1_normalized = vec1 / ( np.linalg.norm(vec1) + 1e-16 )
    vec2_normalized = vec2 / ( np.linalg.norm(vec2) + 1e-16 )
    
    # Compute and return the cosine similarity (dot product of normalized vectors)
    return np.dot(vec1_normalized, vec2_normalized)

# Function to calculate normalized Euclidean distance
def distance(vec1, vec2):
    # Normalize the vectors
    vec1_normalized = vec1 / ( np.linalg.norm(vec1) + 1e-16 )
    vec2_normalized = vec2 / ( np.linalg.norm(vec2) + 1e-16 )
     
    # Calculate Euclidean distance on normalized vectors
    #return np.linalg.norm(vec1_normalized - vec2_normalized)
    return cosine_similarity(vec1, vec2)

def prepare_dataset():
    # Predefined reference strings
    reference_strings = [
        "Hello, world!",
        "This is a test.",
        "I love AI and embeddings.",
        "How can I assist you today?",
        "Goodbye, see you soon!"
    ]
    
    # Compute embeddings for reference strings
    print("\nComputing embeddings for reference strings...")
    reference_embeddings = [embedder(text)[0][0] for text in reference_strings]  # Use the first embedding directly
    
    # Create a DataFrame to store reference strings and their embeddings
    df = pd.DataFrame({
        "Reference String": reference_strings,
        "Embedding": reference_embeddings
    })
    print("Reference embeddings ready.\n")
    return df

def compare(input_text, dataset):
    # Generate embedding for the input text
    input_embedding = embedder(input_text)[0][0]  # Use the first embedding directly
            
    # Calculate similarities
    dataset["Similarity"] = dataset.apply( lambda row: distance(input_embedding, row['Embedding']), axis=1)
            
    # Sort by similarity
    df_sorted = dataset.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
    return df_sorted

def ui():
    dataset = prepare_dataset()
    print("Enter text to compare (type 'exit' to quit):")
    while True:
        try:
            # Read input from stdin
            text = input("Embedder > ")
            if text.lower() == "exit":
                print("Exiting...")
                break

            df = compare(text, dataset)
            
            # Display results
            print("\nSimilarities:")
            print(df[["Reference String", "Similarity"]])
            print(f"\nMost similar: '{df.iloc[0]['Reference String']}' with distance {df.iloc[0]['Similarity']:.4f}\n")
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    ui()
