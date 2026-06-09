import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm
from umap import UMAP


class SentenceRepresentation:
    """
    Base class for sentence representation extraction.
    """

    def __init__(self, model_name: str, data, text_col: str):
        self.data = data
        self.model = self.load_model(model_name)
        self.sentences = self.extract_sentences(data, text_col)
        self.embeddings: np.ndarray = np.array([])  # Initialize as empty array

    def load_model(self, model_name: str):
        """
        Load the model. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from data. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def extract_embeddings(self, batch_size):
        """
        Extract embeddings for the sentences. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def save_embeddings_line_by_line(
        self, file_path, id_col="messageId", verbose: bool = False
    ):
        """
        Saves embeddings along with IDs into a JSON Lines file.

        Parameters:
        - data: DataFrame containing IDs of the messages.
        - embeddings: List or array of embeddings.
        - file_path: Path to save the JSON Lines file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        iterable = (
            tqdm(
                zip(self.data[id_col].values, self.embeddings), desc="Saving embeddings"
            )
            if verbose
            else zip(self.data[id_col].values, self.embeddings)
        )

        with open(path, "w") as f:
            for id, embedding in iterable:
                record = {"id": id, "embedding": embedding.tolist()}
                f.write(json.dumps(record) + "\n")

    @staticmethod
    def load_embeddings_line_by_line(file_path, verbose: bool = False):
        """
        Loads embeddings from a JSONL (JSON Lines) file.

        Parameters:
        - path: Path to the JSONL file containing embeddings.

        Returns:
        - Tuple of IDs and their corresponding embeddings as a numpy array.
        """
        ids = []
        embeddings = []

        with open(file_path, "r") as f:
            iterable = tqdm(f, desc="Loading embeddings") if verbose else f
            for line in iterable:
                data = json.loads(line)
                ids.append(data["id"])
                embeddings.append(data["embedding"])
        return ids, np.array(embeddings)


class NomicRepresentation(SentenceRepresentation):
    """
    Representation class for extracting sentence embeddings using Sentence Transformers.
    """

    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a Sentence Transformer model.
        """
        return SentenceTransformer(model_name, trust_remote_code=True)

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from a pandas DataFrame column.
        """
        return data[text_col].apply(lambda x: f"clustering: {x}").tolist()

    def extract_embeddings(self, batch_size=100):
        """
        Extract embeddings using the loaded Sentence Transformer model.
        """
        extracted_embeddings = []
        num_batches = math.ceil(len(self.sentences) / batch_size)
        for i in tqdm(range(num_batches)):
            # Define the start and end of the current batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(self.sentences))

            # Extract the current batch
            batch_sentences = self.sentences[start_idx:end_idx]
            batch_embeddings = self.model.encode(
                batch_sentences, show_progress_bar=False
            )
            extracted_embeddings.append(batch_embeddings)
            torch.cuda.empty_cache()  # Clear unused memory

        self.embeddings = np.vstack(extracted_embeddings)


class GeneralRepresentation(SentenceRepresentation):
    """
    Representation class for extracting sentence embeddings using Sentence Transformers.
    """

    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a Sentence Transformer model.
        """
        return SentenceTransformer(model_name)

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from a pandas DataFrame column.
        """
        return data[text_col].tolist()

    def extract_embeddings(self, batch_size=64):
        """
        Extract embeddings using the loaded Sentence Transformer model.
        """
        self.embeddings = self.model.encode(
            self.sentences, show_progress_bar=True, batch_size=batch_size
        )


class Evaluation:
    """
    Class providing methods for producing semantic maps and clustering analysis.
    """

    def __init__(self, embeddings: np.ndarray, data: pd.DataFrame):
        self.embeddings = embeddings
        self.data = data

    def generate_coordinates(
        self, metric: str = "cosine", verbose: bool = True
    ) -> np.ndarray:
        """
        Generates a 2D representation of embeddings using UMAP for visualization.

        Parameters:
        - embeddings (np.ndarray): The embeddings array.
        - metric (str): The distance metric to use for UMAP.

        Returns:
        - np.ndarray: The 2D coordinates of the embeddings.
        """
        umap_model = UMAP(metric=metric, random_state=1, verbose=verbose)
        return umap_model.fit_transform(self.embeddings)

    def calculate_silhouette(
        self,
        label_col: str = "class_label",
        metric: str = "cosine",
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates the silhouette score for the given embeddings and labels.

        Args:
        - label_col (str): The column name in the dataframe `data` that contains class labels.
        - metric (str): The metric to use when calculating distance between instances in a feature array.

        Returns:
        - Tuple[float, np.ndarray]: A tuple containing the average silhouette score and individual silhouette scores.

        """
        avg_score = silhouette_score(
            self.embeddings, self.data[label_col], metric=metric
        )
        individual_scores = silhouette_samples(
            self.embeddings, self.data[label_col], metric=metric
        )

        return avg_score, individual_scores

    def visualise_and_save_embeddings(
        self,
        title: str,
        label_col: str = "class_label",
        id_col: str = "messageId",
        metric: str = "cosine",
        size: int = 2,
        opacity: float = 0.7,
        file_path: Path = None,
        verbose: bool = False,
    ) -> None:
        """
        Visualizes embeddings in a 2D plot with given labels and saves the data with silhouette scores.

        Parameters:
        - title (str): Title for the plot.
        - label_col (str): Column name that contains class labels.
        - id_col (str): Column name that contains message id.
        - metric (str): Distance metric for generating coordinates and calculating silhouette scores.
        - size (int): Size of each point in the plot.
        - opacity (float): Opacity of each point in the plot.
        - file_path (str, optional): File path where the data with silhouette scores will be saved. If None, no file will be saved.
        """
        try:
            coordinates = self.generate_coordinates(metric, verbose)
            df = pd.DataFrame(coordinates, columns=["x", "y"])
            df["messageId"] = self.data[id_col].tolist()
            df["class_label"] = self.data[label_col].tolist()

            avg_score, individual_scores = self.calculate_silhouette(label_col, metric)
            df["true_score"] = individual_scores

            print(f"The average silhouette score is {avg_score:.4f}")

            if file_path:
                # path = Path(file_path)
                file_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure the directory exists
                df.to_json(file_path, lines=True, orient="records")
                logging.info(f"Data with silhouette scores saved to {file_path}")

            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="class_label",
                title=title,
                size_max=size,
                opacity=opacity,
            )
            fig.update_traces(marker=dict(size=size))
            return fig
        except Exception as e:
            logging.error(
                "An error occurred while visualizing and saving embeddings: %s", e
            )
            raise

    def calculate_average_similarity(
        self,
        embed_col: str = "embeddings",
        label_col: str = "class_label",
        file_path: Path = None,
    ) -> Dict[str, float]:
        """
        Calculates and optionally saves the average cosine similarity within each class based on embeddings.

        Args:
        - embed_col (str): The column in the dataframe `data` that contains the embeddings as list or arrays.
        - label_col (str): The column name in the dataframe `data` that contains class labels.
        - file_path (str, optional): Path to save the results as a JSON file. If None, the results are not saved.

        Returns:
        - Dict[str, float]: A dictionary where keys are class labels and values are the average cosine similarity for each class.
        """
        average_similarities = {}
        self.data[embed_col] = self.embeddings.tolist()
        for class_label, group in self.data.groupby(label_col):
            group_embeddings = np.array(group[embed_col].tolist())
            if len(group_embeddings) > 1:
                similarities = cosine_similarity(group_embeddings)
                # Exclude the diagonal (self-similarity) from the average calculation by subtracting the sum of all embeddings which contains 1 for self similarity from the number of embeddings in that group.
                avg_similarity = (np.sum(similarities) - len(group_embeddings)) / (
                    len(group_embeddings) * (len(group_embeddings) - 1)
                )
            else:
                avg_similarity = 1.0  # Only one sample, assume perfect self-similarity
            average_similarities[class_label] = avg_similarity
            average_similarity_df = pd.DataFrame([average_similarities])
            logging.info(
                f"Average similarity for class {class_label}: {avg_similarity:.4f}"
            )
        print(f"Average Similarity: {np.mean(list(average_similarities.values()))}")

        if file_path:
            try:
                file_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  

                average_similarity_df.to_json(file_path, lines=True, orient="records")
                logging.info(f"Average similarities saved to {file_path}")
            except Exception as e:
                logging.error("Failed to save average similarities: %s", e)
                raise
        return average_similarity_df

    def compute_similarity_matrix(
        self, label_col: str = "class_label", file_path: Path = None
    ):
        """
        Computes a similarity matrix for intra-class and inter-class comparisons.

        Args:
        - label_col (str): The column in data that contains the class labels.

        Returns:
        - pd.DataFrame: A DataFrame containing the similarity scores between classes.
        """
        data = self.data.copy()
        embeddings = self.embeddings.copy()
        data["embeddings"] = embeddings.tolist()
        embeddings = np.vstack(data["embeddings"])
        unique_labels = data[label_col].unique()
        similarity_matrix = pd.DataFrame(index=unique_labels, columns=unique_labels)

        for label in tqdm(unique_labels):
            for other_label in unique_labels:
                # Extract embeddings for both labels
                label_embeddings = embeddings[data[label_col] == label]
                other_label_embeddings = embeddings[data[label_col] == other_label]

                # Calculate the mean cosine similarity between two sets of embeddings
                if len(label_embeddings) > 0 and len(other_label_embeddings) > 0:
                    similarity = cosine_similarity(
                        label_embeddings, other_label_embeddings
                    ).mean()
                else:
                    similarity = np.nan

                similarity_matrix.loc[label, other_label] = similarity
        if file_path:
            try:
                file_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure the directory exists
                similarity_matrix = similarity_matrix.reset_index()
                similarity_matrix.to_json(file_path, orient="records", lines=True)
                logging.info(f"Similarity Matrix saved to {file_path}")
            except Exception as e:
                logging.error("Failed to save similarity matrix: %s", e)
                raise

        return similarity_matrix


def find_checkpoint_folders(path):
    """
    Finds and returns a list of directories starting with 'checkpoint' within a given path.

    Args:
    - path (str or Path): The directory path where to look for checkpoint folders.

    Returns:
    - List[Path]: A list of paths to the directories that start with 'checkpoint'.
    """
    path = Path(path)  # Ensure the path is a Path object
    checkpoint_folders = [p for p in path.rglob("checkpoint*") if p.is_dir()]
    return checkpoint_folders


def main():
    from tqdm import tqdm

    from contrastive_experiment.utils import get_base_folder

    data = pd.DataFrame(
        {
            "class_label": ["class1", "class1", "class2", "class2", "class3"],
            "text": ["text1", "text2", "text3", "text4", "text5"],
        }
    )

    # Simulate embeddings
    embeddings = np.random.rand(5, 768)  # Assuming 768-dimensional embeddings


    model = GeneralRepresentation("all-MiniLM-L6-v2", data, "text")
    model.embeddings = embeddings  # Manually set embeddings for testing



if __name__ == "__main__":
    main()
