import pandas as pd
import numpy as np
from transformers import pipeline

# Load the model and setup the embedder
print("Loading the model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = pipeline("feature-extraction", model=model_name)
print(f"Model '{model_name}' loaded successfully.")

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
    df_sorted = dataset.sort_values(by="Similarity")
    return df_sorted

def ui():
    dataset = prepare_dataset()
    print("Enter text to compare (type 'exit' to quit):")
    while True:
        try:
            # Read input from stdin
            text = input(">>> ")
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
