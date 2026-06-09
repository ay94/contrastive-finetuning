import logging
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from tqdm.autonotebook import tqdm

from contrastive_experiment.results_utils import DataManager, Results
from contrastive_experiment.classify_utils import KNNClassifier
from contrastive_experiment.eval_utils import (Evaluation,
                                               GeneralRepresentation,
                                               find_checkpoint_folders)
from contrastive_experiment.train_utils import (DataGenerator, Trainer,
                                                TrainingConfig)
from contrastive_experiment.utils import setup_logging

logger = setup_logging()


class TrainWorkflow:
    def __init__(self, base_folder, config, output_dir, train_parameters):
        self.base_folder = Path(base_folder)
        self.config = config
        self.output_dir = Path(output_dir)
        self.train_parameters = train_parameters
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None

    def read_data(self):
        logging.info("Reading data...")
        train_data = pd.read_json(
            self.base_folder / self.config.get("train_path"), lines=True
        )
        test_data = pd.read_json(
            self.base_folder / self.config.get("test_path"), lines=True
        )
        logging.info("Generating datasets...")
        train_generator = DataGenerator(train_data, config=self.config)
        test_generator = DataGenerator(test_data, config=self.config)
        self.train_dataset = train_generator.generate_dataset()
        self.test_dataset = test_generator.generate_dataset()
        self.train_parameters["num_samples"] = len(self.train_dataset)

    def create_trainer(self):
        logging.info("Creating trainer...")
        training_config = TrainingConfig(self.train_parameters, self.output_dir)
        self.trainer = Trainer(training_config)

    def evaluation(self):
        logging.info("Evaluating...")
        test_binary_evaluator = BinaryClassificationEvaluator(
            sentences1=self.test_dataset["sentence1"],
            sentences2=self.test_dataset["sentence2"],
            labels=self.test_dataset["label"],
            name="test",
        )
        test_results = test_binary_evaluator(self.trainer.model)
        return test_binary_evaluator, pd.DataFrame([test_results])

    def train(self, evaluator=None):
        logging.info("Training...")
        if evaluator:
            self.trainer.train(self.train_dataset, self.test_dataset, evaluator)
        else:
            self.trainer.train(self.train_dataset)

    def run(self, evaluate=False):
        self.read_data()
        self.create_trainer()
        if evaluate:
            evaluator, initial_results = self.evaluation()
            logging.info(f"Initial evaluation results:\n{initial_results}")
            self.train(evaluator)
            _, final_results = self.evaluation()
            logging.info(f"Final evaluation results:\n{final_results}")
            return initial_results, final_results
        else:
            self.train()


class EmbeddingExtractionWorkflow:
    def __init__(
        self,
        base_folder,
        experiment_dir,
        search_params,
        text_col,
        model_name,
        base_line_dir=None,
        data_dir="contrastive-data",
    ):
        self.base_folder = Path(base_folder)
        self.base_line_dir = Path(base_line_dir) if base_line_dir is not None else None
        self.experiment_dir = Path(experiment_dir)
        self.search_params = search_params
        self.text_col = text_col
        self.model_name = model_name
        self.data_dir = data_dir

    def read_data(self):
        logging.info("Reading data...")
        try:
            self.train_data = pd.read_json(
                self.base_folder / self.data_dir / "train.json", lines=True
            )
            self.test_data = pd.read_json(
                self.base_folder / self.data_dir / "validation.json",
                lines=True,
            )
            self.exemplar_data = pd.read_json(
                self.base_folder / self.data_dir / "exemplar.json",
                lines=True,
            )
        except Exception as e:
            logging.error(f"Failed to read data: {e}")

    def extract_base_line_embeddings(self):
        logging.info("Generating baseline embeddings...")
        base_line_embeddings_dir = self.base_line_dir / "embeddings"
        train_extractor = GeneralRepresentation(
            self.model_name, self.train_data, self.text_col
        )
        test_extractor = GeneralRepresentation(
            self.model_name, self.test_data, self.text_col
        )
        exemplar_extractor = GeneralRepresentation(
            self.model_name, self.exemplar_data, self.text_col
        )

        self.process_embeddings(
            train_extractor, base_line_embeddings_dir / "train-embeddings.json"
        )
        self.process_embeddings(
            test_extractor, base_line_embeddings_dir / "test-embeddings.json"
        )
        self.process_embeddings(
            exemplar_extractor,
            base_line_embeddings_dir / "exemplar-embeddings.json",
            batch_size=512,
        )

    def extract_fine_tuned_embeddings(self):
        logging.info("Searching for configurations...")
        configs = TrainingConfig.find_configs(self.experiment_dir, self.search_params)
        if not configs:
            logging.error("No configurations found matching the search criteria.")
            return

        for config in configs:
            training_config = TrainingConfig(config["config"], self.experiment_dir)
            config_embeddings_dir = training_config.output_dir / "embeddings"
            logging.info(f"Processing config: {config}")

            for checkpoint in tqdm(find_checkpoint_folders(training_config.output_dir)):
                train_file_path = config_embeddings_dir / f"train-embeddings-{checkpoint.name}.json"
                if not train_file_path.exists():
                    train_extractor = GeneralRepresentation(
                        str(checkpoint), self.train_data, self.text_col
                    )
                    self.process_embeddings(
                    train_extractor,
                    train_file_path,
                )
                else:
                    logging.info(f"train data already processed for this {checkpoint.name}")
                test_file_path = config_embeddings_dir / f"test-embeddings-{checkpoint.name}.json"
                if not test_file_path.exists():
                    test_extractor = GeneralRepresentation(
                        str(checkpoint), self.test_data, self.text_col
                    )
                    self.process_embeddings(
                    test_extractor,
                    test_file_path,
                )
                else:
                    logging.info(f"test data already processed for this {checkpoint.name}")
                exemplar_file_path = config_embeddings_dir / f"exemplar-embeddings-{checkpoint.name}.json"
                if not exemplar_file_path.exists():
                    exemplar_extractor = GeneralRepresentation(
                        str(checkpoint), self.exemplar_data, self.text_col
                    )
                    self.process_embeddings(
                    exemplar_extractor,
                    exemplar_file_path,
                    batch_size=256,
                )
                else:
                    logging.info(f"exemplar data already processed for this {checkpoint.name}")

                
                
                

    def process_embeddings(self, extractor, file_path, batch_size=None):
        if not file_path.exists():
            try:
                if batch_size:
                    extractor.extract_embeddings(batch_size=batch_size)
                else:
                    extractor.extract_embeddings()
                extractor.save_embeddings_line_by_line(file_path)
                logging.info(f"{file_path.name} embeddings generated and saved.")
            except Exception as e:
                logging.error(f"Failed to generate/save embeddings: {e}")
        else:
            logging.info(f"{file_path.name} embeddings already been saved.")

    def run(self):
        self.read_data()
        if (
            self.train_data is not None
            and self.test_data is not None
            and self.exemplar_data is not None
        ):
            if self.base_line_dir:
                logging.info(f"Baseline extraction ...")
                self.extract_base_line_embeddings()
            else:
                logging.info(f"Baseline extraction is skipped")
            
            logging.info(f"Experiment extraction ...")
            self.extract_fine_tuned_embeddings()
        else:
            logging.error("Data not loaded properly. Aborting embedding generation.")


class ValidationWorkflow:
    def __init__(self, base_folder, experiment_dir, search_params, base_line_dir=None, data_dir="contrastive-data"):
        self.base_folder = Path(base_folder)
        self.base_line_dir = Path(base_line_dir) if base_line_dir is not None else None
        self.experiment_dir = Path(experiment_dir)
        self.search_params = search_params
        self.data_dir = data_dir

    def read_data(self):
        logging.info("Reading data...")
        try:
            self.train_data = pd.read_json(
                self.base_folder / self.data_dir / "train.json", lines=True
            )
            self.test_data = pd.read_json(
                self.base_folder / self.data_dir / "validation.json",
                lines=True,
            )
            self.mask = pd.Series([True] * len(self.test_data))
            return True
        except Exception as e:
            logging.error(f"Failed to read data: {e}")
            return False

    def load_embeddings(self, file_path):
        try:
            _, embeddings = GeneralRepresentation.load_embeddings_line_by_line(
                file_path
            )
            return embeddings
        except Exception as e:
            logging.error(f"Failed to load embeddings from {file_path}: {e}")
            return None

    def process_evaluation(self, embeddings, data, results_dir, split, checkpoint):
        evaluator = Evaluation(embeddings, data)
        semantic_map_file_path = (
            results_dir
            / f'{split}-semantic-map{"" if checkpoint is None else "-"+checkpoint}.json'
        )
        if semantic_map_file_path.exists():
            logging.info(
                f"Semantic map file already exists. Skipping {split} evaluation."
            )
        else:
            evaluator.visualise_and_save_embeddings(
                "Semantic Map", file_path=semantic_map_file_path
            )
        average_similarity_file_path = (
            results_dir
            / f'{split}-average-similarity{"" if checkpoint is None else "-"+checkpoint}.json'
        )
        if average_similarity_file_path.exists():
            logging.info(
                f"Average similarity file already exists. Skipping {split} evaluation."
            )
        else:
            evaluator.calculate_average_similarity(
                file_path=average_similarity_file_path
            )
        similarity_matrix_file_path = (
            results_dir
            / f'{split}-similarity-matrix{"" if checkpoint is None else "-"+checkpoint}.json'
        )
        if similarity_matrix_file_path.exists():
            logging.info(
                f"Similarity matrix file already exists. Skipping {split} evaluation."
            )
        else:
            evaluator.compute_similarity_matrix(file_path=similarity_matrix_file_path)
        return

    def evaluate_embeddings(self, embeddings_dir, results_dir, checkpoint=None):
        if not self.read_data():
            return
        logging.info("Loading embeddings...")

        train_embeddings = self.load_embeddings(
            embeddings_dir
            / f'train-embeddings{"" if checkpoint is None else "-"+checkpoint}.json'
        )
        test_embeddings = self.load_embeddings(
            embeddings_dir
            / f'test-embeddings{"" if checkpoint is None else "-"+checkpoint}.json'
        )

        logging.info("Processing embeddings...")
        if train_embeddings is not None and test_embeddings is not None:
            self.process_evaluation(
                train_embeddings, self.train_data, results_dir, "train", checkpoint
            )
            self.process_evaluation(
                test_embeddings[self.mask],
                self.test_data[self.mask].copy(),
                results_dir,
                "test",
                checkpoint,
            )

    def evaluate_base_line_embeddings(self):
        logging.info("Evaluating baseline embeddings...")
        base_line_embeddings_dir = self.base_line_dir / "embeddings"
        base_line_results_dir = self.base_line_dir / "results"
        self.evaluate_embeddings(base_line_embeddings_dir, base_line_results_dir)

    def evaluate_fine_tuned_embeddings(self):
        logging.info("Searching for configurations...")
        configs = TrainingConfig.find_configs(self.experiment_dir, self.search_params)
        if not configs:
            logging.error("No configurations found matching the search criteria.")
            return

        for config in configs:
            training_config = TrainingConfig(config["config"], self.experiment_dir)
            config_embeddings_dir = training_config.output_dir / "embeddings"
            config_results_dir = training_config.output_dir / "results"
            for checkpoint in tqdm(
                find_checkpoint_folders(training_config.output_dir),
                desc="Processing Checkpoints",
            ):
                logging.info(f"Evaluating fine_tuned {checkpoint.name}...")
                self.evaluate_embeddings(
                    config_embeddings_dir, config_results_dir, checkpoint.name
                )

    def run(self):
        if not self.read_data():
            logging.error("Data not loaded properly. Aborting workflow.")
            return
        if self.base_line_dir:
            self.evaluate_base_line_embeddings()
        else:
            logging.info("skipping base line")
        self.evaluate_fine_tuned_embeddings()


class ClassificationWorkflow:
    def __init__(
        self,
        base_folder,
        fine_tuned_model_dir,
        search_params,
        base_model_dir=None,
        data_dir="contrastive-data",
    ):
        self.base_folder = Path(base_folder)
        self.base_model_dir = Path(base_model_dir) if base_model_dir is not None else None
        self.fine_tuned_model_dir = Path(fine_tuned_model_dir)
        self.search_params = search_params
        self.data_dir = data_dir

    def read_data(self):
        logging.info("Reading data...")
        try:
            self.train_data = pd.read_json(
                self.base_folder / self.data_dir / "train.json", lines=True
            )
            self.test_data = pd.read_json(
                self.base_folder / self.data_dir / "validation.json",
                lines=True,
            )
            self.exemplar_data = pd.read_json(
                self.base_folder / self.data_dir / "exemplar.json",
                lines=True,
            )
            self.mask = pd.Series([True] * len(self.test_data))
            return True
        except Exception as e:
            logging.error(f"Failed to read data: {e}")
            return False

    def load_embeddings(self, embeddings_path):
        try:
            _, embeddings = GeneralRepresentation.load_embeddings_line_by_line(
                embeddings_path, verbose=True
            )
            return embeddings
        except Exception as e:
            logging.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            return None

    def evaluate_classifier(
        self, classifier, embeddings, data, results_dir, models_dir, config_label=""
    ):
        if embeddings is None or data is None:
            logging.error("Invalid embeddings or data.")
            return

        output_data_path = results_dir / f"annotated_validation{config_label}.json"
        unmasked_output_data_path = (
            results_dir / f"unmasked_annotated_validation{config_label}.json"
        )
        report_path = results_dir / f"report{config_label}.csv"
        model_path = models_dir / f"knn_model{config_label}.pkl"

        if not (output_data_path.exists() and report_path.exists()):
            logging.info("Classifying masked data...")
            validation_data, report = classifier.predict_and_evaluate(
                embeddings,
                data,
                "class_label",
                masking=self.mask,
                report_path=report_path,
                output_data_path=output_data_path,
            )
        else:
            logging.info(
                "Masked output data and report already exist and will not be regenerated."
            )
        if not unmasked_output_data_path.exists():
            logging.info("Classifying unmasked data...")
            unmasked_validation_data, _ = classifier.predict_and_evaluate(
                embeddings,
                data,
                "class_label",
                output_data_path=unmasked_output_data_path,
            )
        else:
            logging.info(
                "Unmasked output data already exists and will not be regenerated."
            )
        
        if not model_path.exists():
            logging.info(f"saving model in {model_path}")
            classifier.save_model(model_path)

    def classify_base_model(self):
        if not self.read_data():
            logging.error("Failed to read data. Aborting baseline classification.")
            return
        base_model_embeddings_dir = self.base_model_dir / "embeddings"
        base_model_results_dir = self.base_model_dir / "results"
        base_model_models_dir = self.base_model_dir / "models"
        logging.info("Classifying baseline data...")
        # Check if classification results already exist
        output_data_path = base_model_results_dir / "annotated_validation.json"
        report_path = base_model_results_dir / "report.json"
        unmasked_output_data_path = (
            base_model_results_dir / f"unmasked_annotated_validation.json"
        )
        model_path = base_model_models_dir / f"knn_model.pkl"

        if output_data_path.exists() and report_path.exists() and unmasked_output_data_path.exists() and model_path.exists():
            logging.info("Baseline classification results already exist.")
            return
        
        test_embeddings = self.load_embeddings(
            base_model_embeddings_dir / "test-embeddings.json"
        )
        exemplar_embeddings = self.load_embeddings(
            base_model_embeddings_dir / "exemplar-embeddings.json"
        )

        classifier = KNNClassifier()
        classifier.train_classifier(
            exemplar_embeddings, self.exemplar_data, "class_label"
        )
        self.evaluate_classifier(
            classifier, test_embeddings, self.test_data, base_model_results_dir, base_model_models_dir
        )

    def classify_fine_tuned_model(self):
        logging.info("Searching for configurations...")
        configs = TrainingConfig.find_configs(
            self.fine_tuned_model_dir, self.search_params
        )
        if not configs:
            logging.error("No configurations found matching the search criteria.")
            return

        for config in configs:
            training_config = TrainingConfig(
                config["config"], self.fine_tuned_model_dir
            )
            config_embeddings_dir = training_config.output_dir / "embeddings"
            config_results_dir = training_config.output_dir / "results"
            config_models_dir = training_config.output_dir / "models"
            logging.info(f'Processing config: {config["config_dir"]}')

            for checkpoint in tqdm(
                find_checkpoint_folders(training_config.output_dir),
                desc="Processing checkpoints",
            ):
                logging.info(f"Loading {checkpoint.name}")
                # Check if classification results already exist for the current checkpoint
                output_data_path = config_results_dir / f"annotated_validation-{checkpoint.name}.json"
                report_path = config_results_dir / f"report-{checkpoint.name}.csv"
                unmasked_output_data_path = (
                    config_results_dir / f"unmasked_annotated_validation{checkpoint.name}.json"
                )
                model_path = config_results_dir / f"knn_model{checkpoint.name}.pkl"

                if output_data_path.exists() and report_path.exists() and unmasked_output_data_path.exists() and model_path.exists():
                    logging.info(f"Classification results for {checkpoint.name} already exist.")
                    continue
                
                test_embeddings = self.load_embeddings(
                    config_embeddings_dir / f"test-embeddings-{checkpoint.name}.json"
                )
                exemplar_embeddings = self.load_embeddings(
                    config_embeddings_dir
                    / f"exemplar-embeddings-{checkpoint.name}.json"
                )

                classifier = KNNClassifier()
                classifier.train_classifier(
                    exemplar_embeddings, self.exemplar_data, "class_label"
                )
                self.evaluate_classifier(
                    classifier,
                    test_embeddings,
                    self.test_data,
                    config_results_dir,
                    config_models_dir,
                    f"-{checkpoint.name}",
                )

    def run(self):
        if not self.read_data():
            logging.error("Data not loaded properly. Aborting workflow.")
            return
        if self.base_model_dir:
            self.classify_base_model()
        else:
            logging.info('skipping base model')
        self.classify_fine_tuned_model()


class ResultsWorkflow:
    def __init__(self, model_dir, experiment_name):
        self.model_dir = Path(model_dir).expanduser()
        self.experiment_name = experiment_name
        logging.basicConfig(level=logging.INFO)

    def run(self):
        try:
            logging.info(
                f"Starting results processing for {self.experiment_name} in {self.model_dir}"
            )
            results = Results(self.model_dir, self.experiment_name)
            outputs = results.aggregate_results()

            # Check if outputs are valid or not empty if needed
            if outputs:
                manager = DataManager(self.model_dir)
                manager.save_data(outputs)
                logging.info("Data processing and saving completed successfully.")
            else:
                logging.warning("No data to save, outputs were empty.")
        except Exception as e:
            logging.error(
                f"An error occurred during the results workflow: {e}", exc_info=True
            )
