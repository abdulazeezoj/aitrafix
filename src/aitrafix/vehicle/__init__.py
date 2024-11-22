import json
import logging
import os
from logging import Logger
from typing import Any

import joblib  # type: ignore
import numpy as np
from numpy._typing._array_like import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: Logger = logging.getLogger(__name__)


class VehicleCounter:
    def __init__(
        self,
        sequence_length: int = 3,
        model_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize the TrafficController.

        Args:
            sequence_length (int, optional): The length of the traffic state sequence. Defaults to 3.
            model_dir (str | None, optional): The directory to load the model from. Defaults to None.
            name (str | None, optional): The name of the model

        """

        self.name: str = name if name else "aitrafix-traffic"
        self.sequence_length: int = sequence_length

        self.directions: list[str] = ["north", "south", "east", "west"]

        # Define valid traffic light states
        self.valid_states: list[dict[str, str]] = [
            {"north": "green", "south": "red", "east": "red", "west": "red"},
            {"north": "yellow", "south": "red", "east": "red", "west": "red"},
            {"north": "red", "south": "green", "east": "red", "west": "red"},
            {"north": "red", "south": "yellow", "east": "red", "west": "red"},
            {"north": "red", "south": "red", "east": "green", "west": "red"},
            {"north": "red", "south": "red", "east": "yellow", "west": "red"},
            {"north": "red", "south": "red", "east": "red", "west": "green"},
            {"north": "red", "south": "red", "east": "red", "west": "yellow"},
            {"north": "red", "south": "red", "east": "red", "west": "red"},
        ]

        # Create state encoder
        self.state_encoder = LabelEncoder()
        self.state_encoder.fit(range(len(self.valid_states)))

        # Initialize model to predict state index
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names: list[str] = []

        if model_dir and os.path.exists(model_dir):
            self._load_model(model_dir)

    def _validate_state(self, light_state: dict[str, str]) -> bool:
        """
        Check if a light state combination is valid.

        Args:
            light_state (dict[str, str]): The light state to validate.

        Returns:
            bool: True if the state is valid, False otherwise.
        """

        return any(
            all(light_state[d] == valid_state[d] for d in self.directions)
            for valid_state in self.valid_states
        )

    def _get_state_index(self, light_state: dict[str, str]) -> int:
        """
        Get the index of a light state in valid_states.

        Args:
            light_state (dict[str, str]): The light state to get the index of.

        Returns:
            int: The index of the light state in valid_states.
        """

        for i, valid_state in enumerate(self.valid_states):
            if all(light_state[d] == valid_state[d] for d in self.directions):
                return i
        raise ValueError("Invalid light state combination")

    def _preprocess(
        self, data: list[dict[str, Any]]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Convert raw JSON data into features and labels.

        Args:
            data (list[dict[str, Any]]): The raw data to preprocess.

        Returns:
            tuple[NDArray[Any], NDArray[Any]]: The processed features and labels
        """

        processed_data: list[Any] = []
        feature_names: list[str] = []

        # Create feature names
        for i in range(self.sequence_length):
            for direction in self.directions:
                feature_names.append(f"t-{self.sequence_length-i}_vehicles_{direction}")
            for direction in self.directions:
                feature_names.append(
                    f"t-{self.sequence_length-i}_emergency_{direction}"
                )
            feature_names.append(f"t-{self.sequence_length-i}_state_index")

        self.feature_names = feature_names

        # Process sequences
        for i in range(len(data) - self.sequence_length):
            sequence: list[dict[str, Any]] = data[i : i + self.sequence_length]
            next_state: dict[str, Any] = data[i + self.sequence_length]

            if not all(
                self._validate_state(entry["light"])
                for entry in sequence + [next_state]
            ):
                logger.warning(f"Skipping invalid state at index {i}")
                continue

            features: list[Any] = []
            for entry in sequence:
                # Add vehicle counts
                for direction in self.directions:
                    features.append(entry["vehicles"][direction])
                # Add emergency vehicle presence
                for direction in self.directions:
                    features.append(entry["emergency"][direction])
                # Add state index
                features.append(self._get_state_index(entry["light"]))

            # Label is the index of the next state
            label: int = self._get_state_index(next_state["light"])

            processed_data.append((features, label))

        if not processed_data:
            raise ValueError("No valid sequences found in data")

        X: NDArray[Any] = np.array([x[0] for x in processed_data])
        y: NDArray[Any] = np.array([x[1] for x in processed_data])

        return X, y

    def predict(self, sequence: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Predict next traffic light states ensuring valid combinations.

        Args:
            sequence (list[dict[str, Any]]): The sequence of traffic light states.

        Returns:
            dict[str, Any]: The predicted traffic light states.
        """

        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Sequence must contain exactly {self.sequence_length} entries"
            )

        if not all(self._validate_state(entry["light"]) for entry in sequence):
            raise ValueError("Invalid state in input sequence")

        _features: list[Any] = []
        for entry in sequence:
            for direction in self.directions:
                _features.append(entry["vehicles"][direction])
            for direction in self.directions:
                _features.append(entry["emergency"][direction])
            _features.append(self._get_state_index(entry["light"]))

        features: NDArray[Any] = np.array(_features).reshape(1, -1)
        state_index: int = self.model.predict(features)[0]  # type: ignore

        result: dict[str, Any] = self.valid_states[state_index].copy()

        return result

    def train(
        self,
        data_dir: str,
        test_size: float = 0.2,
        random_state: int = 42,
        save_model: bool = False,
        save_report: bool = False,
        model_dir: str | None = None,
    ) -> None:
        """
        Train the model on the given data.

        Args:
            data_dir (str): The path to the data file.
            test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.
            save_model (bool, optional): Whether to save the trained model. Defaults to False.
            save_report (bool, optional): Whether to save the evaluation report. Defaults to False.
            model_dir (str | None, optional): The directory to save the model and report to. Defaults to None.
        """

        # Load data
        data: list[dict[str, Any]] = self._load_data(data_dir)

        # Prepare data
        _data: tuple[NDArray[Any], NDArray[Any]] = self._preprocess(data)
        X: NDArray[Any] = _data[0]
        y: NDArray[Any] = _data[1]

        _split: list[Any] = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train: NDArray[Any] = _split[0]
        X_test: NDArray[Any] = _split[1]
        y_train: NDArray[Any] = _split[2]
        y_test: NDArray[Any] = _split[3]

        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)  # type: ignore

        # Evaluate model
        logger.info("Evaluating model...")
        train_metrics: dict[Any, Any] = self._eval_model(X_train, y_train)
        logger.info("Train set metrics:")
        logger.info(json.dumps(train_metrics, indent=2))

        test_metrics: dict[Any, Any] = self._eval_model(X_test, y_test)
        logger.info("Test set metrics:")
        logger.info(json.dumps(test_metrics, indent=2))

        self.report: dict[str, dict[Any, Any]] = {
            "train": train_metrics,
            "test": test_metrics,
        }

        if save_model:
            # Save model
            logger.info("Saving model...")
            model_dir = model_dir if model_dir else "./"
            self._save_model(model_dir)

        if save_report:
            # Save report
            logger.info("Saving report...")
            report_dir: str = model_dir if model_dir else "./"
            self._save_report(report_dir)

    def _eval_model(self, X: NDArray[Any], y: NDArray[Any]) -> dict[Any, Any]:
        """
        Evaluate model performance.

        Args:
            X (NDArray[Any]): The input features.
            y (NDArray[Any]): The target labels.

        Returns:
            dict[Any, Any]: The evaluation metrics.
        """

        predictions: NDArray[Any] = self.model.predict(X)  # type: ignore

        # Calculate basic metrics
        report: str | dict[Any, Any] = classification_report(
            y, predictions, output_dict=True
        )

        # Create confusion matrix
        cm: NDArray[Any] = confusion_matrix(y, predictions)

        # Add state mapping to metrics
        state_mapping: dict[int, dict[str, str]] = {
            i: state for i, state in enumerate(self.valid_states)
        }

        if isinstance(report, str):
            report = {"classification_report": report}

        report["confusion_matrix"] = cm.tolist()
        report["state_mapping"] = state_mapping

        return report

    def _save_model(self, model_dir: str) -> None:
        """
        Save the trained model and associated data.

        Args:
            model_dir (str): The directory to save the model to.

        Raises:
            Exception: If an error occurs saving the model.
        """

        try:
            os.makedirs(model_dir, exist_ok=True)

            model_path: str = os.path.join(model_dir, f"{self.name}.joblib")
            joblib.dump(self.model, model_path)  # type: ignore

            config: dict[str, Any] = {
                "sequence_length": self.sequence_length,
                "directions": self.directions,
                "valid_states": self.valid_states,
                "feature_names": self.feature_names,
            }
            config_path: str = os.path.join(model_dir, f"{self.name}.config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            logger.info(f"Model and configuration saved to {model_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def _load_model(self, model_dir: str) -> None:
        """
        Load a trained model and associated data.

        Args:
            model_dir (str): The directory to load the model from.

        Raises:
            Exception: If an error occurs loading the model.
        """

        try:
            model_path: str = os.path.join(model_dir, f"{self.name}.joblib")
            self.model: RandomForestClassifier = joblib.load(model_path)  # type: ignore

            config_path: str = os.path.join(model_dir, f"{self.name}.config.json")
            with open(config_path, "r") as f:
                config: dict[str, Any] = json.load(f)

            self.sequence_length = config["sequence_length"]
            self.directions = config["directions"]
            self.valid_states = config["valid_states"]
            self.feature_names = config["feature_names"]

            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_data(self, data_dir: str) -> list[dict[str, Any]]:
        """
        Load data from a JSON file.

        Args:
            data_dir (str): The path to the data file.

        Returns:
            list[dict[str, Any]]: The loaded data.

        Raises:
            Exception: If an error occurs loading the data.
        """

        try:
            with open(data_dir, "r") as f:
                data: list[dict[str, Any]] = json.load(f)

            logger.info(f"Data loaded from {data_dir}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _save_report(self, report_dir: str) -> None:
        """
        Save a classification report to a file.

        Args:
            report_dir (str): The directory to save the report to.

        Raises:
            Exception: If an error occurs saving the report.
        """

        try:
            os.makedirs(report_dir, exist_ok=True)

            report_path: str = os.path.join(report_dir, f"{self.name}.report.json")
            with open(report_path, "w") as f:
                json.dump(self.report, f)

            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise
