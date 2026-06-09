import shutil
from pathlib import Path

import pytest

from contrastive_experiment.train_utils import TrainingConfig


@pytest.fixture
def config_dict():
    return {"learning_rate": 0.01, "epochs": 10}


def test_init_and_save_config(config_dict):
    config = TrainingConfig(config_dict)
    assert config.output_dir.exists(), "Output directory should be created"

    config.save_config()
    assert (config.output_dir / "config.json").exists(), "Config file should be saved"

    # Cleanup after test
    shutil.rmtree(config.base_dir)


def test_load_config(config_dict):
    config = TrainingConfig(config_dict)
    config.save_config()

    config.load_config()
    assert config.config["learning_rate"] == 0.01, "Config should be loaded correctly"

    # Cleanup after test
    shutil.rmtree(config.base_dir)
