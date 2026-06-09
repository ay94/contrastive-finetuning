#!/usr/bin/env python3
"""
End-to-end pipeline runner for contrastive-experiment using demo data.

Run after generating demo data with:
    python scripts/create_demo_data.py

Then:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --base-dir ~/my-experiments
    streamlit run app.py
"""

import argparse
import logging
from pathlib import Path

from contrastive_experiment.utils import get_base_folder, initialise_structure
from contrastive_experiment.workflows import (
    ClassificationWorkflow,
    EmbeddingExtractionWorkflow,
    ResultsWorkflow,
    TrainWorkflow,
    ValidationWorkflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR_NAME = "all_minilm_l6_v2"
EXPERIMENT_NAME = "contrastive_fixed"

# TrainingConfig params: num_samples is set dynamically in read_data
TRAIN_PARAMETERS = {
    "model_name": MODEL_NAME,
    "loss": "contrastive",
    "run_name": EXPERIMENT_NAME,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "logging_steps": 0.25,
    "save_steps": 0.5,
}

DATA_CONFIG = {
    "train_path": "contrastive-data/train.json",
    "test_path": "contrastive-data/validation.json",
    "strategy": "contrastive",
    "method": "stratified",
    "num_pairs": 500,
    "text_col": "text",
    "class_col": "class_label",
}

# Matches configs that used this training run
SEARCH_PARAMS = {"loss": "contrastive", "model_name": MODEL_NAME}


def run_pipeline(base_folder: Path):
    experiment_folder = base_folder / "contrastive-experiment" / MODEL_DIR_NAME
    baseline_dir = experiment_folder / "baseline"
    experiment_dir = experiment_folder / EXPERIMENT_NAME

    logger.info("Initialising folder structure at %s", experiment_folder)
    initialise_structure(experiment_folder, [EXPERIMENT_NAME])

    # Stage 1: Train
    logger.info("=== Stage 1: Training ===")
    train_workflow = TrainWorkflow(
        base_folder=base_folder,
        config=DATA_CONFIG.copy(),
        output_dir=experiment_dir,
        train_parameters=TRAIN_PARAMETERS.copy(),
    )
    train_workflow.run(evaluate=False)

    # Stage 2: Extract embeddings (baseline + fine-tuned checkpoints)
    logger.info("=== Stage 2: Extracting Embeddings ===")
    embed_workflow = EmbeddingExtractionWorkflow(
        base_folder=base_folder,
        experiment_dir=experiment_dir,
        search_params=SEARCH_PARAMS,
        text_col="text",
        model_name=MODEL_NAME,
        base_line_dir=baseline_dir,
    )
    embed_workflow.run()

    # Stage 3: Validation (UMAP maps, similarity metrics)
    logger.info("=== Stage 3: Validation ===")
    val_workflow = ValidationWorkflow(
        base_folder=base_folder,
        experiment_dir=experiment_dir,
        search_params=SEARCH_PARAMS,
        base_line_dir=baseline_dir,
    )
    val_workflow.run()

    # Stage 4: Classification (k-NN on exemplar embeddings)
    logger.info("=== Stage 4: Classification ===")
    class_workflow = ClassificationWorkflow(
        base_folder=base_folder,
        fine_tuned_model_dir=experiment_dir,
        search_params=SEARCH_PARAMS,
        base_model_dir=baseline_dir,
    )
    class_workflow.run()

    # Stage 5: Aggregate results for dashboard
    logger.info("=== Stage 5: Aggregating Results ===")
    results_workflow = ResultsWorkflow(
        model_dir=experiment_folder,
        experiment_name=EXPERIMENT_NAME,
    )
    results_workflow.run()

    logger.info("Pipeline complete.")
    logger.info("Launch dashboard: cd %s && streamlit run app.py", Path(__file__).parent.parent)


def main():
    parser = argparse.ArgumentParser(description="Run the full contrastive pipeline.")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for data and outputs (default: ~/contrastive-experiments)",
    )
    args = parser.parse_args()
    base_folder = get_base_folder(local_path=args.base_dir)
    logger.info("Base folder: %s", base_folder)
    run_pipeline(base_folder)


if __name__ == "__main__":
    main()
