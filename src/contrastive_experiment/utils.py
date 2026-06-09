import logging
from pathlib import Path
from typing import List

from IPython import get_ipython


def get_base_folder(
    local_path: str = None,
    colab_drive_path: str = None,
):
    """
    Returns the base folder for experiment outputs.

    - Locally: pass `local_path` as an absolute path string, or set the
      CONTRASTIVE_BASE_FOLDER environment variable. Defaults to ~/contrastive-experiments.
    - On Colab: pass `colab_drive_path` (e.g. '/content/drive/Shareddrives/MyProject').
      Google Drive is mounted automatically.
    """
    import os
    try:
        if "google.colab" in str(get_ipython()):
            from google.colab import drive
            drive.mount("/content/drive")
            if colab_drive_path:
                return Path(colab_drive_path)
            return Path("/content/drive/MyDrive/contrastive-experiments")
        else:
            if local_path:
                return Path(local_path).expanduser()
            env_path = os.environ.get("CONTRASTIVE_BASE_FOLDER")
            if env_path:
                return Path(env_path).expanduser()
            return Path("~/contrastive-experiments").expanduser()
    except Exception as e:
        logging.error("Error during initialisation: %s", e)


# utils.py


def initialise_structure(base_path: Path, experiment_names: List[str]):
    """
    Creates a standard folder structure for a new experiment.

    Args:
    - base_path (str): The base directory for contrastive experiments.
    - experiment_names (str): The names of the experiment.

    Structure created:
    - base_path/model_name/
      - baseline/
      - results/
      - experiment_name/
        ...
    """
    base_directory = base_path.expanduser()
    baseline_dir = base_directory / "baseline"

    results_dir = base_directory / "results"

    for subfolder in ["embeddings", "models", "results"]:
        (baseline_dir / subfolder).mkdir(parents=True, exist_ok=True)
    if isinstance(experiment_names, (list, tuple)):
        for experiment_name in experiment_names:
            experiment_dir = base_directory / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise TypeError("experiment_names must be a list.")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment structure created at: {base_directory}")


def add_experiment(base_path: Path, experiment_name: str):
    base_directory = base_path.expanduser()
    experiment_dir = base_directory / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment added: {experiment_dir}")


def setup_logging():
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to console handler
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)
    return logger


def main():
    base_folder = get_base_folder()
    print(base_folder)
    experiment_folder = base_folder / "contrastive-experiment"
    initialise_structure(experiment_folder / "paraphrase_multilingual_mpnet", ["contrastive_fixed"])


if __name__ == "__main__":
    main()
