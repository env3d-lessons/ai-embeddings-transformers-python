# Introduction to Embeddings

Embeddings are a way to represent data, especially text or words, as high-dimensional vectors. These vectors capture the semantic relationships between entities, allowing similar items to be close in the vector space. Embeddings are fundamental in Natural Language Processing (NLP) tasks, such as text classification, similarity search, and recommendation systems.

This code demonstrates how to generate and use embeddings for text data. By converting words or sentences into numerical vectors, you can apply various machine learning techniques.

### Key Concepts:
- **Embeddings**: Fixed-size vector representations of data (e.g., words, sentences).
- **Dimensionality**: Number of values in the vector (features), usually in the hundreds or thousands.
- **Semantic Similarity**: Words or phrases with similar meanings have similar vectors.

### Example:
Given the following 2D embeddings for three words:

- "King" → [0.8, 0.6]
- "Queen" → [0.7, 0.5]
- "Apple" → [-0.2, 0.9]

These vectors are normalized (L2 norm), meaning each vector has a magnitude of 1. After normalization, the vectors are:

- "King" → [0.8, 0.6]
- "Queen" → [0.815, 0.586]
- "Apple" → [-0.217, 0.974]

The vectors for "King" and "Queen" are closer to each other than either is to "Apple", reflecting their 
semantic relationship (both are royalty terms). "Apple", being unrelated, is farther away in the vector space.

### Creating a "semantic search engine":
1. Load text data.
2. Generate embeddings using "pre-trained models" via the transformers library from huggingface
3. Use these embeddings to figure out similarities

The main.py file contains one way for creating a semantic search engine, with a very simple dataset.

# Exercise 1

The file *prompts.csv* contains a list of prompts appropriate for feeding into LLMs such as ChatGPT. 
Turn your app into a "prompt generator" by output a prompt from prompts.csv that is most similar to the 
user's input sentence.

# Exercise 2

The current file uses *euclidean distance* to compare similarities, which is basically the geometric 
distance between points.  This approach is intuitive but is not the most performant. I have provided 
you with a function called *cosine_similarity()* which is much faster because of the use of dot product.
Simply put, the dot product of 2 normalized vector calculate the cosine of the angle between the 2 vectors.
Meaning that a value of 1 means they are pointing in the exact same direction because cosine of 0 is 1?

Change the code above to use dot product instead of euclidean distance.
