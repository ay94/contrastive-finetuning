from itertools import combinations

import pandas as pd
import pytest

from contrastive_experiment.train_utils import DataGenerator


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame to use in tests."""
    data = {
        "class_label": ["class1", "class1", "class2", "class2", "class2"],
        "text": ["text1", "text2", "text3", "text4", "text5"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def generator(sample_data):
    """Provides a DataGenerator instance configured for testing."""
    return DataGenerator(sample_data, sample_size=2, text_col="text")


def test_pairs_generated(generator):
    """Tests if pairs are generated."""
    pairs = generator.generate_positive_pairs()
    assert len(pairs) > 0, "Should generate at least one pair"


def test_pairs_have_positive_label(generator):
    """Tests that all generated pairs have a positive label."""
    pairs = generator.generate_positive_pairs()
    assert all(
        pair.label == 1 for pair in pairs
    ), "All pairs should have a positive label"


def test_correct_number_of_combinations(sample_data, generator):
    """Tests that the number of generated pairs matches the expected number of combinations."""
    pairs = generator.generate_positive_pairs()
    expected_pairs_count = 0
    for _, group in sample_data.groupby("class_label"):
        num_texts = min(len(group), generator.sample_size)
        if num_texts > 1:
            expected_pairs_count += len(list(combinations(range(num_texts), 2)))

    assert (
        len(pairs) == expected_pairs_count
    ), "Number of generated pairs should match expected number of combinations"


def test_no_class_samples_more_texts_than_sample_size(sample_data, generator):
    """Tests that no class samples more texts than the specified sample size."""
    pairs = generator.generate_positive_pairs()
    text_counts = {label: 0 for label in sample_data["class_label"].unique()}
    for pair in pairs:
        label = sample_data[sample_data["text"] == pair.texts[0]]["class_label"].iloc[0]
        text_counts[label] += 1

    assert all(
        count <= generator.sample_size for count in text_counts.values()
    ), "No class should sample more texts than the sample size"


def test_contrastive_dataset_columns(generator):
    """Test if the contrastive dataset has the correct structure and columns."""
    generator.train_samples = (
        generator.generate_positive_pairs()
    )  # Manually setting train_samples
    dataset = generator.generate_contrastive_dataset()
    expected_columns = {"sentence1", "sentence2", "label"}
    assert (
        set(dataset.column_names) == expected_columns
    ), "Dataset should contain exactly sentence1, sentence2, and label columns"


def test_contrastive_dataset_labels(generator):
    """Test if all labels in the dataset are correctly set to 1 (positive pairs)."""
    generator.train_samples = generator.generate_positive_pairs()
    dataset = generator.generate_contrastive_dataset()
    assert all(
        label == 1 for label in dataset["label"]
    ), "All labels should be 1 for positive pairs"


def test_contrastive_dataset_no_missing_data(generator):
    """Test if there are no missing values in the dataset."""
    generator.train_samples = generator.generate_positive_pairs()
    dataset = generator.generate_contrastive_dataset()
    for column in dataset.column_names:
        # Convert to pandas Series to use pandas functionality
        series = pd.Series(dataset[column])
        assert (
            not series.isnull().any()
        ), f"Column {column} should not contain any missing values"


# Run tests directly if this file is executed as a script
if __name__ == "__main__":
    pytest.main()
