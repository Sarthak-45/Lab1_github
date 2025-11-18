
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def load_and_prepare_data():
    """
    Load the Iris dataset and return features and labels
    Returns: X (features), y (labels)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest Classifier on the training data
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the random forest
    Returns: Trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_and_evaluate(model, X_test, y_test):
    """
    Make predictions and calculate accuracy
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    Returns: accuracy score (float)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Main execution example
if __name__ == "__main__":
    # Load data
    X, y = load_and_prepare_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_classifier(X_train, y_train)
    print("Model training completed")
    
    # Evaluate model
    accuracy = predict_and_evaluate(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")