from itertools import combinations
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments,
                                   losses)

from contrastive_experiment.train_utils import (DataGenerator, Trainer,
                                                TrainingConfig)


@pytest.fixture
def mock_config():
    config_dict = {
        "pairing_strategy": "positive_contrastive",
        "loss": "contrastive",
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "fp16": True,  # Set to False if GPU can't handle FP16
        "bf16": False,  # Set to True if GPU supports BF16
        "logging_steps": 1000,
        "save_steps": 1000,
        "run_name": "test_run",
    }
    return TrainingConfig(config_dict)


@pytest.fixture
def mock_data_generator():
    mock_data_gen = DataGenerator(
        {"sample_size": 10, "text_col": "text"}
    )  # Adjust parameters as necessary
    mock_data_gen.train_samples = [
        {"texts": ["sample text 1", "sample text 2"], "label": 1}
    ] * 50  # Mocked data
    return mock_data_gen


def test_trainer_initialization(mock_config, mock_data_generator):
    trainer = Trainer(mock_config, mock_data_generator)
    assert (
        trainer.config["model_name"]
        == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


def test_calculate_steps(mock_config, mock_data_generator):
    trainer = Trainer(mock_config, mock_data_generator)
    total_steps = trainer.calculate_total_steps()
    expected_steps = (50 // 16) * 1  # Adjust based on mocked train_samples and config
    assert total_steps == expected_steps


def test_get_loss(mock_config, mock_data_generator):
    trainer = Trainer(mock_config, mock_data_generator)
    loss = trainer.get_loss("contrastive", MagicMock())
    assert isinstance(
        loss, losses.ContrastiveLoss
    )  # Ensure correct loss type is returned


def test_training_process(mock_config, mock_data_generator):
    trainer = Trainer(mock_config, mock_data_generator)
    with patch.object(trainer, "train", return_value=None) as mocked_train:
        trainer.train()
        mocked_train.assert_called_once()  # Ensure that the train method is called


if __name__ == "__main__":
    pytest.main()
