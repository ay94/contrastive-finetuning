import numpy as np
import pandas as pd
import pytest

from contrastive_experiment.eval_utils import GeneralRepresentation
from unittest.mock import mock_open, patch


import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset."""
    data = pd.DataFrame({
        'class_label': ['class1', 'class1', 'class2', 'class2', 'class3'],
        'text': ['text1', 'text2', 'text3', 'text4', 'text5'],
        'messageId': ['1', '2', '3', '4', '5']
    })
    embeddings = np.random.rand(5, 384)  # Random embeddings for testing
    return data, embeddings

@pytest.fixture
def model_instance(sample_data):
    """Fixture to create an instance of GeneralRepresentation with loaded embeddings."""
    data, embeddings = sample_data
    model = GeneralRepresentation('all-MiniLM-L6-v2', data, 'text')
    model.embeddings = embeddings
    return model

def test_silhouette_scores(model_instance):
    """Test that the silhouette scores are calculated and returned properly."""
    avg_silhouette, individual_silhouettes = model_instance.calculate_silhouette()
    assert isinstance(avg_silhouette, float), "Average silhouette score should be a float"
    assert len(individual_silhouettes) == model_instance.embeddings.shape[0], "Should return an individual score for each sample"

def test_average_similarity(model_instance):
    """Test the calculation of average similarities within classes."""
    average_similarities = model_instance.calculate_average_similarity()
    assert isinstance(average_similarities, dict), "Should return a dictionary"
    assert 'class1' in average_similarities, "Dictionary should contain classes as keys"
    assert isinstance(average_similarities['class1'], float), "Similarity scores should be floats"

def test_extraction_of_embeddings(model_instance):
    """Test that embeddings are extracted correctly."""
    original_embeddings = model_instance.embeddings.copy()
    model_instance.extract_embeddings(batch_size=2)
    assert model_instance.embeddings.shape == original_embeddings.shape, "Embeddings should be extracted and have the same shape."
    assert not np.array_equal(model_instance.embeddings, np.zeros(original_embeddings.shape)), "Embeddings should not be zero-filled."
    
    

def test_save_embeddings_line_by_line(model_instance, sample_data):
    """Test saving embeddings to a file."""
    data, _ = sample_data
    m = mock_open()
    with patch("builtins.open", m):
        model_instance.save_embeddings_line_by_line('dummy_path')
    m.assert_called_once()  # Checks if file open was called
    handle = m()
    handle.write.assert_called()  # Checks if write was called


def test_load_embeddings_line_by_line(model_instance):
    """Test loading embeddings from a file."""
    with patch("builtins.open", mock_open(read_data='{"id": "1", "embedding": [0.1, 0.2]}\n{"id": "2", "embedding": [0.3, 0.4]}')):
        ids, embeddings = model_instance.load_embeddings_line_by_line('dummy_path')
    assert len(ids) == 2, "Should load two ids"
    assert embeddings.shape == (2, 2), "Should load embeddings into a numpy array of shape (2, 2)"

    
    
def test_generate_coordinates(model_instance):
    """Test generation of 2D coordinates from embeddings."""
    coords = model_instance.generate_coordinates()
    assert coords.shape[1] == 2, "Should generate 2D coordinates"
    assert coords.shape[0] == model_instance.embeddings.shape[0], "Should generate coordinates for all embeddings"



