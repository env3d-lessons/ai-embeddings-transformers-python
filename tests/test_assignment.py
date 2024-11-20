import pytest
import pandas as pd
import numpy as np
from io import StringIO
import os, sys
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


mock_transformers = MagicMock()

# Mock the `pipeline` function
mock_pipeline = MagicMock()

def mock_embedder(text):
    return [[np.random.rand(3)]]  # Random 3D vector for testing

mock_pipeline.return_value.side_effect = mock_embedder
mock_transformers.pipeline = mock_pipeline

# Replace the `transformers` module in `sys.modules`
sys.modules["transformers"] = mock_transformers

from main import *

@pytest.fixture
def sample_vectors():
    """Fixture to provide sample vectors for distance testing."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    return vec1, vec2

def test_distance_cosine_similarity(sample_vectors):
    """Test if `distance` returns cosine similarity values."""
    vec1, vec2 = sample_vectors

    # Calculate expected cosine similarity
    vec1_normalized = vec1 / np.linalg.norm(vec1)
    vec2_normalized = vec2 / np.linalg.norm(vec2)
    expected_similarity = np.dot(vec1_normalized, vec2_normalized)

    # Call distance (replaced to use cosine similarity internally)
    from main import distance  # Adjust based on your module's structure
    calculated_similarity = distance(vec1, vec2)

    # Verify they match
    assert pytest.approx(calculated_similarity, rel=1e-5) == expected_similarity

def test_prepare_dataset():
    """Test if `prepare_dataset` correctly loads data from prompts.csv."""
    # Replace the embedding process with a mock function

    # Call prepare_dataset
    dataset = prepare_dataset()

    # Verify the DataFrame structure
    assert isinstance(dataset, pd.DataFrame)
    assert "prompt" in [c.lower() for c in dataset.columns]
    assert "embedding" in [c.lower() for c in dataset.columns]


