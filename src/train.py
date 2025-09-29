"""
This script trains a machine learning model on the preprocessed data of iris flowers.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import json
import dvc.api
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    # Load data using DVC API to ensure version control
    logger.info(f"Loading data from {file_path}")
    with dvc.api.open(file_path, mode='r') as fd:
        df = pd.read_json(fd, lines=True)
    return df

def train_model(data: pd.DataFrame, model_path: str):
    # Split the data into features and target
    X = data.drop(columns=['species'])
    y = data['species']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Model Evaluation Report:\n{report}")

    # Save the trained model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_path = os.path.join("data", "iris_modified.json")
    model_path = os.path.join("models", "random_forest_model.joblib")

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load the preprocessed data
    data = load_data(data_path)

    # Train and save the model
    train_model(data, model_path)
    logger.info("Training complete.")