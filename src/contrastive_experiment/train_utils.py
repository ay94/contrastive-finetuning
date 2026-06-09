import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset  # Ensure this import if using Dataset.from_dict
from sentence_transformers import (InputExample, SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments,
                                   losses)
from torch.cuda import get_device_capability
from tqdm.autonotebook import tqdm


class TrainingConfig:
    def __init__(self, config_dict, base_dir="configs") -> None:
        """
        Initializes the TrainingConfig with a dictionary of configuration parameters.
        :param config_dict: A dictionary containing all the necessary training parameters.
        :param base_dir: The base directory to store configuration files.
        """

        # Calculate hash first based on the complete initial configuration
        self.config_dict = config_dict.copy()
        self.base_dir = base_dir
        self.config_hash_value = self.config_hash(config_dict)
        self.output_dir = self.base_dir / self.config_hash_value
        self.save_config(config_dict)
        
        self.generate_config()

        # Calculate dynamic configurations
        self.set_dynamic_config()

    def generate_config(self):
        """
        Extract and add configurations for the TrainerArguments by extracting specific settings
        from a copied configuration dictionary while preserving the original.

        This method removes certain key-value pairs related to training setup from a copy
        of the initial configuration dictionary and uses them to set class attributes.

        Returns:
            dict: A dictionary containing the remaining configuration settings
        """
        # Create a copy of the config dictionary to avoid modifying the original
        temp_config = self.config_dict.copy()

        temp_config = self.config_dict.copy()
        # This parameter is used to calculate the total number of steps dynamically if proportion is passed.
        self.num_samples = temp_config.pop("num_samples")
        self.loss = temp_config.pop("loss")
        self.model_name = temp_config.pop("model_name")

        # Remaining configuration
        self.config = temp_config
        self.config["output_dir"] = str(self.output_dir)
        return self.config

    def set_dynamic_config(self):
        """
        Configures dynamic settings based on the current system's GPU capabilities and num_samples.

        This method adjusts the configuration settings for floating-point precision based on GPU capabilities.
        It also dynamically calculates and sets the logging, saving, and evaluation steps based on the total
        number of training steps computed from the dataset size and batch configuration.
        """

        # Setup based on system's GPU capabilities
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "fp16" in self.config:  # Check if fp16 is not explicitly set
            self.config["fp16"] = device.type == "cuda" and (
                torch.cuda.get_device_capability(device.index)[0] > 6
            )
        if "bf16" in self.config:  # Check if bf16 is not explicitly set
            self.config["bf16"] = (
                device.type == "cuda" and torch.cuda.is_bf16_supported()
            )

        total_steps = self.calculate_total_steps()
        if "logging_steps" in self.config:  # Dynamically set if frac provided
            self.config["logging_steps"] = self.calculate_steps(
                self.config["logging_steps"], total_steps
            )
        if "save_steps" in self.config:  # Dynamically set if frac provided
            self.config["save_steps"] = self.calculate_steps(
                self.config["save_steps"], total_steps
            )
        if "eval_steps" in self.config:  # Dynamically set if frac provided
            self.config["eval_steps"] = self.calculate_steps(
                self.config["eval_steps"], total_steps
            )

    def calculate_steps(self, value, total_steps):
        """
        Calculates the number of steps for saving or logging.

        :param value: Fixed number or fraction of total steps.
        :param total_steps: Total number of training steps.
        :return: The number of steps to log or save.
        """
        if isinstance(value, float):
            return int(value * total_steps)
        return value

    def calculate_total_steps(self):
        """
        Calculates the total number of training steps based on batch size and number of epochs.

        :return: Total training steps.
        """
        num_batches = self.num_samples // self.config["per_device_train_batch_size"]
        return num_batches * self.config["num_train_epochs"]

    def config_hash(self, config_dict):
        hash_input = json.dumps(config_dict, sort_keys=True).encode()
        return hashlib.md5(hash_input).hexdigest()

    def save_config(self, config_dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.output_dir / "config.json"
        with open(filename, "w") as file:
            json.dump(config_dict, file, indent=4)

    def load_config(self):
        filename = self.output_dir / "config.json"
        if filename.exists():
            with open(filename, "r") as file:
                self.config = json.load(file)
        else:
            print("No matching config file found.")

    @staticmethod
    def find_configs(base_dir: str, search_params: dict = None):
        """
        Scans the given directory for configuration files stored within hashed subdirectories and optionally filters them by given parameters.

        :param base_dir: The base directory where hashed subdirectories with configuration files are stored.
        :param search_params: Optional dictionary of parameters to filter configurations.
        :return: A list of paths to configurations that match the search parameters.
        """
        base_path = Path(base_dir)
        matched_configs = []

        # Iterate through all directories within the base directory
        for config_dir in base_path.iterdir():
            if config_dir.is_dir():  # Ensure it's a directory
                config_path = config_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as file:
                        config = json.load(file)
                        if search_params:
                            # Check if all items in search_params are in the config's items
                            if all(
                                item in config.items() for item in search_params.items()
                            ):
                                matched_configs.append(
                                    {"config_dir": config_dir, "config": config}
                                )
                        else:
                            matched_configs.append(
                                {"config_dir": config_dir, "config": config}
                            )

        return matched_configs


class PairingStrategy(ABC):
    @abstractmethod
    def generate_pairs(self, data):
        pass


class PositivePairsStrategy(PairingStrategy):
    def __init__(
        self, data, text_col, class_col, sample_size=None, method="default"
    ) -> None:
        self.data = data
        self.text_col = text_col
        self.class_col = class_col
        self.sample_size = sample_size
        self.method = method

    def generate_pairs(self):
        logging.info("Generate Pairs")
        if self.method == "default":
            return self.default_positive_pairs()
        elif self.method == "stratified":
            return self.stratified_positive_pairs()
        else:
            raise ValueError("Unsupported method for generating pairs")

    def default_positive_pairs(self):
        """
        Generates all possible positive pairs within the same class up to a maximum of 'sample_size' texts per class.
        classA ........ classB.... 

        :return: List of InputExample with texts and a positive label (label=1).
        """
        positive_pairs = []
        stats = {"class_label": [], "num_pairs": [], "num_sentences": []}

        for clb, clb_df in self.data.groupby(self.class_col):
            class_texts = clb_df[self.text_col].tolist()
            if self.sample_size and len(class_texts) > self.sample_size:
                class_texts = random.sample(class_texts, self.sample_size)

            pairs = list(combinations(class_texts, 2))
            positive_pairs.extend(
                [
                    InputExample(texts=[text_a, text_b], label=1)
                    for text_a, text_b in pairs
                ]
            )

            # Record the stats
            stats["class_label"].append(clb)
            stats["num_pairs"].append(len(pairs))
            stats["num_sentences"].append(len(class_texts))
        self.stats_df = pd.DataFrame(stats)

        return positive_pairs

    def stratified_positive_pairs(self):
        """
        Generates a balanced number of positive pairs from each class, ensuring even representation across classes.
        classA, classB, classC....classA
        """

        class_texts = {}
        class_combinations = {}
        positive_pairs = []
        stats = {}

        # Shuffling and creating combinations for each class
        for clb, clb_df in self.data.groupby(self.class_col):
            texts = clb_df[self.text_col].tolist()
            if self.sample_size and len(texts) > self.sample_size:
                texts = random.sample(texts, self.sample_size)
            random.shuffle(texts)
            class_texts[clb] = texts
            class_combinations[clb] = list(combinations(texts, 2))

        total_combinations = sum(len(combs) for combs in class_combinations.values())

        # Process until all combinations are exhausted
        while any(class_combinations.values()):
            for clb, clb_combinations in list(class_combinations.items()):
                if clb_combinations:
                    chosen_combination = random.choice(clb_combinations)
                    positive_pairs.append(
                        InputExample(texts=chosen_combination, label=1)
                    )
                    class_combinations[clb].remove(
                        chosen_combination
                    )  # Remove used combination

        stats = {clb: len(combs) for clb, combs in class_combinations.items()}
        self.stats_df = pd.DataFrame(
            list(stats.items()), columns=["Class Label", "Number of Combinations"]
        )
        self.stats_df["Percentage of Total"] = round(
            (self.stats_df["Number of Combinations"] / total_combinations) * 100, 2
        )

        logging.info(f"Total Combinations Over All: {total_combinations}")
        return positive_pairs


class ContrastivePairsStrategy(PairingStrategy):
    def __init__(
        self,
        data,
        text_col,
        class_col,
        num_pairs=None,
        sample_size=None,
        method="default",
    ):
        self.data = data
        self.text_col = text_col
        self.class_col = class_col
        self.num_pairs = num_pairs
        self.sample_size = sample_size

        self.method = method

    def generate_pairs(self):
        logging.info("Generate Pairs")
        if self.method == "default":
            return self.default_contrastive_pairs()
        elif self.method == "stratified":
            return self.stratified_contrastive_pairs()
        else:
            raise ValueError(f"Unsupported method {self.method} for generating pairs")

    def default_contrastive_pairs(self):
        """
        Generate positive and negative contrastive pairs.
        (classA, classA), (classA, classB), ....., (classA, classZ), (classB, classB),....., etc.
        """
        class_texts = {}
        positive_pairs = {}
        negative_pairs = {}

        # Prepare class texts and shuffle
        for clb, clb_df in self.data.groupby(self.class_col):
            texts = clb_df[self.text_col].tolist()
            if self.sample_size and len(texts) > self.sample_size:
                texts = random.sample(texts, self.sample_size)
            random.shuffle(texts)
            class_texts[clb] = texts
            positive_pairs[clb] = list(combinations(texts, 2))

        # Generate negative pairs for each text in the positive pairs
        for clb_a, texts_a in class_texts.items():
            for text_a in texts_a:
                negative_pairs[text_a] = {}
                for clb_b, texts_b in class_texts.items():
                    if clb_a != clb_b:
                        negative_pairs[text_a][clb_b] = list(product([text_a], texts_b))

        # Store all pairs in a list
        contrastive_pairs = []

        # Round-robin looping until all pairs are exhausted
        class_order = list(class_texts.keys())
        while any(positive_pairs.values()):
            for clb in class_order:
                if positive_pairs[clb]:
                    pos_pair = positive_pairs[clb].pop(
                        0
                    )  # Change to pop(0) for a stable selection
                    contrastive_pairs.append(InputExample(texts=pos_pair, label=1))

                    # Generate negative pairs for the first element in the positive pair
                    text = pos_pair[0]
                    for other_clb in class_order:
                        if other_clb != clb and negative_pairs[text][other_clb]:
                            neg_pair = negative_pairs[text][other_clb].pop(0)
                            contrastive_pairs.append(
                                InputExample(texts=neg_pair, label=0)
                            )

        return contrastive_pairs

    def stratified_contrastive_pairs(self):
        """
        Generate a stratified contrastive pairs.
    
        Generates a set number of contrastive pairs, both positive (within the same class) and negative (across different classes).
        data size = num_pairs * 2
        """

        contrastive_pairs = []
        class_labels = self.data[self.class_col].unique()

        for _ in tqdm(range(self.num_pairs), desc="Number of Pairs"):
            text_a_label = random.choice(class_labels)
            text_b_label = random.choice(list(set(class_labels) - {text_a_label}))

            # Sample positive pair
            text_a1 = (
                self.data[self.data[self.class_col] == text_a_label][self.text_col]
                .sample(n=1)
                .values[0]
            )
            text_a2 = (
                self.data[self.data[self.class_col] == text_a_label][self.text_col]
                .sample(n=1)
                .values[0]
            )
            contrastive_pairs.append(InputExample(texts=[text_a1, text_a2], label=1))

            # Sample negative pair
            text_b = (
                self.data[self.data[self.class_col] == text_b_label][self.text_col]
                .sample(n=1)
                .values[0]
            )
            contrastive_pairs.append(InputExample(texts=[text_a1, text_b], label=0))

        return contrastive_pairs


class DataGenerator:
    def __init__(self, data, config):
        """
        Generate the datasets for training and evaluation and convert them to dataset object

        :param data: DataFrame containing the data, expected to include 'class_label' and the specified 'text_col'.
        :param config: configuration file specifying the data parameters.
        """
        self.data = data
        self.config = config
        # Initialize DataFrame to store stats
        self.stats_df = pd.DataFrame(
            columns=["class_label", "num_pairs", "num_sentences"]
        )

    def generate_pairs(self):
        """
        Generates pairs using the strategy specified.
        """
        return self.strategy.generate_pairs()

    def get_stats(self):
        """
        Returns the DataFrame containing statistics about the pairs generation.
        """
        return self.stats_df

    def create_dataset(self, pairs):
        """
        Converts the generated pairs into a dataset suitable for training using contrastive loss.

        :return: A Dataset object containing the pairs and their labels.
        """
        if pairs:
            data_dict = {"sentence1": [], "sentence2": [], "label": []}
            for sample in pairs:
                if len(sample.texts) == 2 and hasattr(sample, "label"):
                    data_dict["sentence1"].append(sample.texts[0])
                    data_dict["sentence2"].append(sample.texts[1])
                    data_dict["label"].append(sample.label)

            if (
                not data_dict["sentence1"]
                or not data_dict["sentence2"]
                or not data_dict["label"]
            ):
                raise ValueError("Data lists cannot be empty.")

            return Dataset.from_dict(data_dict)
        else:
            logging.error("Pairs are not generated")

    def setup_and_validate(self, config):
        """
        Adjust the configuration with some hard coded rules to avoid mistakes.
        This potentially dangerous and require careful consideration.
        TODO: provide more visibility over the hard coded changes.

        Args:
            config (Dict): configuration dictionary

        Returns:
            Dict: validated dictionary that has been modified according to the rules
        """
        exclusions = ["train_path", "test_path", "strategy"]
        validated_config = {}
        for k, v in config.items():
            if k not in exclusions:
                validated_config[k] = v

        # Validate and adjust num_pairs based on strategy
        if config["strategy"] == "contrastive":
            if config.get("num_pairs") is None and config["method"] != "default":
                validated_config["num_pairs"] = 1000  # Set a default if not specified
                logging.info(
                    "num_pairs was not set for contrastive strategy, setting to default 1000."
                )
        else:  
            if "num_pairs" in config:
                del validated_config["num_pairs"]  
                logging.warning(
                    "num_pairs is not applicable for positive strategy, ignoring it."
                )

        # Validate and adjust method based on num_pairs
        if config["method"] != "stratified" and config.get("num_pairs") is not None:
            validated_config["method"] = "stratified"
            logging.info("Setting method to 'stratified' due to num_pairs being set.")

        # Ensure sample_size is set for default method
        if config["method"] == "default" and config.get("sample_size") is None:
            validated_config["sample_size"] = 10
            logging.info("sample_size not set for default method, setting to 10.")

        # For positive strategy, ensure there's a default sample_size
        if config["strategy"] == "positive" and config.get("sample_size") is None:
            validated_config["sample_size"] = 10
            logging.info("sample_size not set for positive strategy, setting to 10.")

        return validated_config

    def generate_dataset(self):
        """
        Generates a dataset based on the specified pairing strategy.

        :param pairing_strategy: Strategy to use for pairing data points, e.g., 'positive_contrastive'.
        :return: A Dataset object configured according to the specified strategy.
        """
        pairing_strategy = self.config.get("strategy")
        self.config = self.setup_and_validate(self.config)
        match pairing_strategy:
            case "positive":
                self.strategy = PositivePairsStrategy(data=self.data, **self.config)
            case "contrastive":
                self.strategy = ContrastivePairsStrategy(data=self.data, **self.config)
            case _:
                raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")
        pairs = self.generate_pairs()
        return self.create_dataset(pairs)


class Trainer:
    def __init__(self, train_config):
        """
        Initializes the Trainer with a configuration.

        :param train_config: Configuration object containing training parameters.
        :param data_generator: Instance to generate the training/dev dataset.
        """
        self.CONFIG = train_config
        self.model = None
        self.loss = None
        self.device = None
        self.args = None
        self.set_train_arguments()

    def set_train_arguments(self):
        """
        Sets up the training arguments based on the configuration and checks for GPU capabilities for FP16 training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.CONFIG.model_name)
        self.model.to(self.device)
        self.loss = self.get_loss(self.CONFIG.loss, self.model)
        self.args = SentenceTransformerTrainingArguments(
            **self.CONFIG.generate_config()
        )

    def get_loss(self, loss_name, model):
        """
        Returns the loss function based on the specified type.

        :param loss_name: Name of the loss type.
        :param model: Sentence Transformer model to which the loss will be applied.
        :return: Configured loss object.
        """
        match loss_name:
            case "contrastive":
                return losses.ContrastiveLoss(model=model)
            case "triplets":
                return losses.TripletLoss(model=model)
            case _:
                raise ValueError(f"Unknown loss type: {loss_name}")

    def train(self, train_dataset, eval_dataset=None, evaluator=None):
        """
        Conducts the training process using the configured settings and dataset.

        :param evaluator: Evaluator object to be used during training for performance validation.
        """
        if eval_dataset and evaluator:
            trainer = SentenceTransformerTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=self.loss,
                evaluator=evaluator,
            )
        else:
            trainer = SentenceTransformerTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                loss=self.loss,
            )
        trainer.train()
