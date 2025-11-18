

import unittest
import numpy as np
from src.ml_model import (
    load_and_prepare_data,
    split_dataset,
    train_classifier,
    predict_and_evaluate
)


class TestMLModel(unittest.TestCase):
    """Test case class for ML Model functions"""
    
    def test_load_and_prepare_data(self):
        """Test that data is loaded correctly with proper shape and labels"""
        X, y = load_and_prepare_data()
        
        self.assertEqual(X.shape, (150, 4), "Features should have shape (150, 4)")
        self.assertEqual(y.shape, (150,), "Labels should have shape (150,)")
        self.assertEqual(len(np.unique(y)), 3, "Should have 3 unique classes")
    
    def test_split_dataset(self):
        """Test that dataset splitting produces correct train/test sizes"""
        X, y = load_and_prepare_data()
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
        
        self.assertEqual(X_train.shape[0], 120, "Training set should have 120 samples")
        self.assertEqual(X_test.shape[0], 30, "Test set should have 30 samples")
        self.assertEqual(y_train.shape[0], 120, "Training labels should have 120 samples")
        self.assertEqual(y_test.shape[0], 30, "Test labels should have 30 samples")
    
    def test_train_classifier(self):
        """Test that classifier training returns a valid model"""
        X, y = load_and_prepare_data()
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        model = train_classifier(X_train, y_train)
        
        self.assertIsNotNone(model, "Model should not be None")
        self.assertTrue(hasattr(model, 'predict'), "Model should have predict method")
    
    def test_predict_and_evaluate(self):
        """Test that model evaluation returns valid accuracy score"""
        X, y = load_and_prepare_data()
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        model = train_classifier(X_train, y_train)
        accuracy = predict_and_evaluate(model, X_test, y_test)
        
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be <= 1")
        self.assertGreater(accuracy, 0.85, "Model accuracy should be > 85%")


if __name__ == '__main__':
    unittest.main()