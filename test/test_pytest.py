import pytest
import numpy as np
from src.ml_model import (
    load_and_prepare_data,
    split_dataset,
    train_classifier,
    predict_and_evaluate
)


def test_load_and_prepare_data():
    """Test that data is loaded correctly with proper shape and labels"""
    X, y = load_and_prepare_data()
    
    assert X.shape == (150, 4), "Features should have shape (150, 4)"
    assert y.shape == (150,), "Labels should have shape (150,)"
    assert len(np.unique(y)) == 3, "Should have 3 unique classes"


def test_split_dataset():
    """Test that dataset splitting produces correct train/test sizes"""
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    
    assert X_train.shape[0] == 120, "Training set should have 120 samples"
    assert X_test.shape[0] == 30, "Test set should have 30 samples"
    assert y_train.shape[0] == 120, "Training labels should have 120 samples"
    assert y_test.shape[0] == 30, "Test labels should have 30 samples"


def test_train_classifier():
    """Test that classifier training returns a valid model"""
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_classifier(X_train, y_train)
    
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'predict'), "Model should have predict method"


def test_predict_and_evaluate():
    """Test that model evaluation returns valid accuracy score"""
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_classifier(X_train, y_train)
    accuracy = predict_and_evaluate(model, X_test, y_test)
    
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1"
    assert accuracy > 0.85, "Model accuracy should be greater than 85%"


# Parametrized test example (commented out by default)
# Uncomment to test different test_size values
# @pytest.mark.parametrize("test_size,expected_train,expected_test", [
#     (0.2, 120, 30),
#     (0.3, 105, 45),
#     (0.25, 112, 38),
# ])
# def test_split_dataset_parametrized(test_size, expected_train, expected_test):
#     """Test dataset splitting with various test sizes"""
#     X, y = load_and_prepare_data()
#     X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size)
#     
#     assert X_train.shape[0] == expected_train
#     assert X_test.shape[0] == expected_test