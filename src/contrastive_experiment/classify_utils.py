import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP


class KNNClassifier:
    """
    A class to handle the classification using embeddings and k-NN.
    """

    def __init__(self, n_neighbors: int = 10, metric: str = "cosine"):
        """
        Initializes the classifier with the specified Sentence Transformer model and k-NN parameters.

        Args:
        - n_neighbors (int): Number of neighbors to use for k-NN.
        - metric (str): The distance metric to use for k-NN.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = None

    def train_classifier(
        self,
        exemplar_embeddings: np.ndarray,
        exemplar_data: pd.DataFrame,
        label_col: str,
    ):
        """
        Trains the k-NN classifier using exemplar data.

        Args:
        - exemplar_data (pd.DataFrame): DataFrame containing the training data with texts and labels.
        - text_col (str): Column name containing the text data.
        - label_col (str): Column name containing the labels.
        """
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric=self.metric
        )
        self.knn.fit(exemplar_embeddings, exemplar_data[label_col])

    def predict_and_evaluate(
        self,
        validation_embeddings: np.ndarray,
        validation_data: pd.DataFrame,
        label_col: str,
        masking: pd.Series = None,
        report_path: str = None,
        output_data_path: str = None,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """
        Predicts the labels for validation data and evaluates the predictions, with optional masking.

        Args:
        - validation_embeddings (np.ndarray): Embeddings of the validation data.
        - validation_data (pd.DataFrame): DataFrame containing the validation data with actual labels.
        - label_col (str): Column name containing the labels.
        - masking (pd.Series, optional): A boolean series indicating which rows to include in the evaluation.
        - report_path (str, optional): Path to save the classification report as a CSV file.
        - output_data_path (str, optional): Path to save the modified validation data as a JSON file.
        - metric (str, optional): The metric used for silhouette score ('consine', 'euclidean', etc.).

        Returns:
        - pd.DataFrame: Validation data with additional column for predicted labels.
        - pd.DataFrame: Classification report as DataFrame.
        """
        if masking is not None:
            validation_embeddings = validation_embeddings[masking.values]
            validation_data = validation_data[masking].copy()

        try:

            predicted_labels = self.knn.predict(validation_embeddings)
            validation_data["predicted_label"] = predicted_labels
            individual_scores = self.calculate_silhouette(
                validation_embeddings, predicted_labels, metric
            )
            validation_data["pred_score"] = individual_scores

            report = classification_report(
                y_true=validation_data[label_col],
                y_pred=predicted_labels,
                output_dict=True,
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.reset_index(inplace=True)
            report_df = report_df.rename(columns={"index": "class"})

            if report_path:
                report_path = Path(report_path)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_df.to_csv(report_path, index=False)

            if output_data_path:
                output_data_path = Path(output_data_path)
                output_data_path.parent.mkdir(parents=True, exist_ok=True)
                validation_data.to_json(output_data_path, orient="records", lines=True)

            return validation_data, report_df
        except:
            logging.error("Model doesn't exist, please train or load knn model")
            raise

    def calculate_silhouette(
        self,
        embeddings: np.ndarray,
        predicted_labels: List[str],
        metric: str = "cosine",
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates the silhouette score for the given embeddings and prediction labels.

        Args:
        - label_col (str): The column name in the dataframe `data` that contains class labels.
        - metric (str): The metric to use when calculating distance between instances in a feature array.

        Returns:
        - Tuple[float, np.ndarray]: A tuple containing the average silhouette score and individual silhouette scores.

        The silhouette score is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
        """
        individual_scores = silhouette_samples(
            embeddings, predicted_labels, metric=metric
        )
        return individual_scores

    def save_model(self, file_path: str):
        """
        Saves the trained k-NN model to a file.

        Args:
        - file_path (str): The path where the model should be saved.
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            joblib.dump(self.knn, file_path)
            logging.info(f"Model saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save the model: {e}")
            raise

    def load_model(self, file_path: str):
        """
        Loads a k-NN model from a file.

        Args:
        - file_path (str): The path where the model is saved.
        """
        try:
            file_path = Path(file_path)
            self.knn = joblib.load(file_path)
            logging.info(f"Model loaded from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            raise
