import json
import logging
import re
from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report

from contrastive_experiment.utils import get_base_folder


class Results:
    def __init__(self, model_dir, experiment_name):
        self.model_dir = Path(model_dir).expanduser()
        self.base_line_dir = self.model_dir / "baseline"
        self.config_dir = self.model_dir / experiment_name
        self.reports = []
        self.semantic_maps = []
        self.average_similarities = []
        self.annotated_average_similarities = []
        self.similarity_matrices = []
        self.annotated_datasets = []
        self.unmasked_annotated_datasets = []
        self.combined_metrics = []

    def read_base_line_results(self):
        """General method to read all results from baseline directory."""
        results_dir = self.base_line_dir / "results"
        for file_path in results_dir.iterdir():
            if file_path.is_file():
                try:
                    self.process_file(file_path, "Baseline", 'baseline')
                    logging.info(f"Successfully processed baseline file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to process baseline file {file_path}: {e}")

    def read_configs_results(self):
        """General method to read all results from configuration directory."""
        for config_path in self.config_dir.iterdir():
            results_dir = config_path / "results"
            if results_dir.exists():
                for file_path in results_dir.iterdir():
                    model = self.extract_checkpoint(file_path.name)
                    if file_path.is_file():
                        try:
                            self.process_file(
                                file_path,
                                model,
                                self.load_config(config_path)["run_name"],
                            )
                            logging.info(
                                f"Successfully processed configuration file: {file_path}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Failed to process configuration file {file_path}: {e}"
                            )
            else:
                logging.info(f"Results not created yet for {config_path.name}")
                continue

    def process_file(self, file_path, model, config=None):
        """Process each file based on its type and content."""
        if "report" in file_path.name and file_path.suffix == ".csv":
            self.read_report(file_path, model, config)
        elif "semantic-map" in file_path.name and file_path.suffix == ".json":
            self.read_semantic_map(file_path, model, config)
        elif "average-similarity" in file_path.name and file_path.suffix == ".json":
            self.read_average_similarity(file_path, model, config)
        elif "similarity-matrix" in file_path.name and file_path.suffix == ".json":
            self.read_similarity_matrix(file_path, model, config)
        elif (
            "annotated" in file_path.name
            and "unmasked" not in file_path.name
            and file_path.suffix == ".json"
        ):
            self.read_annotated_data(file_path, model, config)
        elif (
            "annotated" in file_path.name
            and "unmasked" in file_path.name
            and file_path.suffix == ".json"
        ):
            self.read_unmasked_annotated_data(file_path, model, config)

    # Individual read methods for different file types
    def read_report(self, file_path, model, config):
        try:
            report = pd.read_csv(file_path)
            report["Experiment"] = self.model_dir.parts[-3]
            report["Model Name"] = self.model_dir.parts[-1]
            report["Model"] = model
            if config:
                report["Config"] = config
            self.reports.append(report)
            return report
        except Exception as e:
            logging.error(f"Error reading report {file_path}: {e}")
            return None

    def read_semantic_map(self, file_path, model, config):
        try:
            semantic_map = pd.read_json(file_path, lines=True)
            semantic_map["Split"] = "train" if "train" in file_path.name else "test"
            semantic_map["Experiment"] = self.model_dir.parts[-3]
            semantic_map["Model Name"] = self.model_dir.parts[-1]
            semantic_map["Model"] = model
            if config:
                semantic_map["Config"] = config
            self.semantic_maps.append(semantic_map)
        except Exception as e:
            logging.error(f"Error reading semantic map {file_path}: {e}")

    def read_average_similarity(self, file_path, model, config):
        try:
            average_similarity = pd.read_json(file_path, lines=True)
            average_similarity = average_similarity.melt(
                var_name="Class Label", value_name="Average Similarity"
            )
            average_similarity["Split"] = (
                "train" if "train" in file_path.name else "test"
            )
            average_similarity["Experiment"] = self.model_dir.parts[-3]
            average_similarity["Model Name"] = self.model_dir.parts[-1]
            average_similarity["Model"] = model
            if config:
                average_similarity["Config"] = config
            self.average_similarities.append(average_similarity)
        except Exception as e:
            logging.error(f"Error reading average similarity {file_path}: {e}")

    def read_similarity_matrix(self, file_path, model, config):
        try:
            similarity_matrix = pd.read_json(file_path, lines=True)
            similarity_matrix["Split"] = (
                "train" if "train" in file_path.name else "test"
            )
            similarity_matrix["Experiment"] = self.model_dir.parts[-3]
            similarity_matrix["Model Name"] = self.model_dir.parts[-1]
            similarity_matrix["Model"] = model
            if config:
                similarity_matrix["Config"] = config
            self.similarity_matrices.append(similarity_matrix)
        except Exception as e:
            logging.error(f"Error reading similarity matrix {file_path}: {e}")

    def read_annotated_data(self, file_path, model, config):
        try:
            annotated_data = pd.read_json(file_path, lines=True)
            annotated_data["Experiment"] = self.model_dir.parts[-3]
            annotated_data["Model Name"] = self.model_dir.parts[-1]
            annotated_data["Model"] = model
            if config:
                annotated_data["Config"] = config
            self.annotated_datasets.append(annotated_data)
        except Exception as e:
            logging.error(f"Error reading annotated data {file_path}: {e}")

    def read_unmasked_annotated_data(self, file_path, model, config):
        try:
            unmasked_annotated_data = pd.read_json(file_path, lines=True)
            unmasked_annotated_data["Experiment"] = self.model_dir.parts[-3]
            unmasked_annotated_data["Model Name"] = self.model_dir.parts[-1]
            unmasked_annotated_data["Model"] = model
            if config:
                unmasked_annotated_data["Config"] = config
            self.unmasked_annotated_datasets.append(unmasked_annotated_data)
        except Exception as e:
            logging.error(f"Error reading annotated data {file_path}: {e}")

    def load_config(self, config_path):
        filename = config_path / "config.json"
        if filename.exists():
            with open(filename, "r") as file:
                config = json.load(file)
            return config
        else:
            logging.info("No matching config file found.")
            return None

    def extract_metrics_from_report(self, report_df):
        if "class" in report_df.columns:
            macro_avg = (
                report_df[report_df["class"] == "macro avg"].iloc[0]
                if not report_df[report_df["class"] == "macro avg"].empty
                else None
            )
            weighted_avg = (
                report_df[report_df["class"] == "weighted avg"].iloc[0]
                if not report_df[report_df["class"] == "weighted avg"].empty
                else None
            )
        else:
            macro_avg, weighted_avg = None, None
        return macro_avg, weighted_avg

    def aggregate_silhouette_scores(self, data, label_col, score_col):
        overall_silhouette_score = data[score_col].mean()
        per_class_silhouette_scores = data.groupby(label_col)[score_col].mean()
        return overall_silhouette_score, per_class_silhouette_scores

    def extract_and_combine_metrics(
        self, report_df, semantic_map, annotated_data, average_similarity
    ):
        macro_avg, weighted_avg = self.extract_metrics_from_report(report_df)
        true_silhouette, _ = self.aggregate_silhouette_scores(
            semantic_map, "class_label", "true_score"
        )
        pred_silhouette, _ = self.aggregate_silhouette_scores(
            annotated_data, "class_label", "pred_score"
        )
        average_similarity = average_similarity["Average Similarity"].mean()

        combined_metrics = pd.DataFrame(
            {
                "macro_precision": [macro_avg["precision"]],
                "macro_recall": [macro_avg["recall"]],
                "macro_f1-score": [macro_avg["f1-score"]],
                "weighted_precision": [weighted_avg["precision"]],
                "weighted_recall": [weighted_avg["recall"]],
                "weighted_f1-score": [weighted_avg["f1-score"]],
                "true_silhouette": [true_silhouette],
                "pred_silhouette": [pred_silhouette],
                "average_similarity": [average_similarity],
                "support": [macro_avg["support"]],
                "Config": [macro_avg.get("Config", "-")],
                "Experiment": [macro_avg["Experiment"]],
                "Model Name": [macro_avg["Model Name"]],
                "Model": [macro_avg["Model"]],
            }
        )
        return combined_metrics

    def combine_data_metrics(self):
        for report, semantic_map, annotated_data, average_similarity in zip(
            self.reports,
            self.semantic_maps,
            self.annotated_datasets,
            self.average_similarities,
        ):
            self.combined_metrics.append(
                self.extract_and_combine_metrics(
                    report, semantic_map, annotated_data, average_similarity
                )
            )

    def aggregate_results(self):
        logging.info("Starting to aggregate results...")
        # First, read the initial data sets
        try:
            self.read_base_line_results()
            logging.info("Baseline results read successfully.")
        except Exception as e:
            logging.error(f"Failed to read baseline results: {e}")

        try:
            self.read_configs_results()
            logging.info("Configuration results read successfully.")
        except Exception as e:
            logging.error(f"Failed to read configuration results: {e}")
        try:
            self.combine_data_metrics()
            logging.info("Aggregate metrics combined.")
        except Exception as e:
            logging.error(f"Failed to combine metrics: {e}")

        # Aggregate data from successfully read files
        aggregated_data = {
            "reports": (
                pd.concat(self.reports, ignore_index=True)
                if self.reports
                else pd.DataFrame()
            ),
            "semantic_maps": (
                pd.concat(self.semantic_maps, ignore_index=True)
                if self.semantic_maps
                else pd.DataFrame()
            ),
            "average_similarities": (
                pd.concat(self.average_similarities, ignore_index=True)
                if self.average_similarities
                else pd.DataFrame()
            ),
            "similarity_matrices": (
                pd.concat(self.similarity_matrices, ignore_index=True)
                if self.similarity_matrices
                else pd.DataFrame()
            ),
            "annotated_datasets": (
                pd.concat(self.annotated_datasets, ignore_index=True)
                if self.annotated_datasets
                else pd.DataFrame()
            ),
            "unmasked_annotated_datasets": (
                pd.concat(self.unmasked_annotated_datasets, ignore_index=True)
                if self.annotated_datasets
                else pd.DataFrame()
            ),
            "combined_metrics": (
                pd.concat(self.combined_metrics, ignore_index=True)
                if self.annotated_datasets
                else pd.DataFrame()
            ),
        }
        logging.info("All data aggregated successfully.")
        return aggregated_data

    def extract_checkpoint(self, file_name):
        """
        Extracts the checkpoint identifier from a filename, using regex to handle both .json and .csv extensions.

        Args:
        - file_name (str): The filename from which to extract the checkpoint identifier.

        Returns:
        - str: The extracted checkpoint identifier.
        """
        # Remove file extensions using regex and extract the last two parts
        cleaned_name = re.sub(r"\.(json|csv)$", "", file_name)
        checkpoint = "-".join(cleaned_name.split("-")[-2:])
        return checkpoint


class DataManager:
    def __init__(self, directory):
        """
        Initialize the DataManager with a directory for data storage.

        Args:
            directory (str): Path to the directory where data will be saved and loaded.
        """
        self.directory = Path(directory).expanduser() / "results"
        self.directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        self.combined_metrics = pd.DataFrame()

    def save_data(self, data):
        """
        Save the specific data attributes to files in the specified format ('csv' or 'json').
        """

        for name, df in data.items():
            if not df.empty:
                file_path = self.directory / f"{name}.json"
                df.to_json(file_path, orient="records", lines=True)
                logging.info(f"Saved {name} to {file_path}")
            else:
                logging.info(f"No data to save for {name}, skipping.")

    def load_data(self, analysis):
        """
        Load the data from files into specific attributes based on the specified format.
        Only loads non-empty data and logs if the file is empty.
        """
        name = analysis
        file_path = self.directory / f"{name}.json"
        if file_path.exists():
            df = pd.read_json(file_path, lines=True)
            if df.empty:
                logging.warning(
                    f"Loaded {name} from {file_path}, but it is empty. Not assigning to attribute."
                )
            else:
                logging.info(f"Loaded {name} from {file_path} and updated attribute.")
            return df
        else:
            logging.warning(f"No file found for {name} in json")
        return None


