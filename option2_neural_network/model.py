"""
Neural Network Classifier for Industrial Signal Quality Classification

Model: Neural Network with Softmax Output Layer

This implementation uses a feedforward neural network with softmax activation
for multi-class classification, as covered in Lecture 06 (Optimization in 
Neural Networks) and Lecture 12 (Deep Learning).

Features:
- Multi-class classification (4 classes) via softmax
- L2 regularization (weight decay) for preventing overfitting
- Early stopping based on validation loss
- Feature scaling (standardization)
- Feature selection using correlation
- Class-weighted loss for handling imbalance
- Gradient descent with backpropagation
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import base_class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_class import BaseMLModel


class Model(BaseMLModel):
    # Model: Neural Network with Softmax Output and L2 Regularization
    
    def __init__(self, hidden_units=128, learning_rate=0.01, lambda_reg=0.01,
                 n_iterations=1000, batch_size=256, early_stopping_patience=20,
                 n_features_to_select=150, use_feature_selection=True,
                 validation_split=0.15, random_state=42):
        """
        Initialize Neural Network Classifier
        
        Args:
            hidden_units: Number of hidden layer neurons
            learning_rate: Learning rate for gradient descent
            lambda_reg: L2 regularization strength
            n_iterations: Maximum number of training iterations
            batch_size: Batch size for mini-batch gradient descent
            early_stopping_patience: Epochs to wait before early stopping
            n_features_to_select: Number of top features to select
            use_feature_selection: Whether to apply feature selection
            validation_split: Fraction of training data for validation
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.n_features_to_select = n_features_to_select
        self.use_feature_selection = use_feature_selection
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.n_classes = 4
        
        # Model parameters
        self.W1 = None  # Input to hidden weights
        self.b1 = None  # Hidden layer bias
        self.W2 = None  # Hidden to output weights
        self.b2 = None  # Output layer bias
        
        # Feature preprocessing
        self.selected_features = None
        self.feature_mean = None
        self.feature_std = None
        
        # Class weights
        self.class_weights = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def _softmax(self, z):
        """
        Softmax activation function
        
        Args:
            z: Linear combinations (n_samples, n_classes)
            
        Returns:
            Softmax probabilities (n_samples, n_classes)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def _one_hot_encode(self, y):
        """Convert class labels to one-hot encoding"""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def _cross_entropy_loss(self, y_true_onehot, y_pred_probs, sample_weights=None):
        """
        Cross-entropy loss with L2 regularization
        
        Args:
            y_true_onehot: True labels (one-hot encoded)
            y_pred_probs: Predicted probabilities
            sample_weights: Sample weights for class imbalance
            
        Returns:
            Loss value
        """
        n_samples = len(y_true_onehot)
        
        # Clip probabilities to avoid log(0)
        y_pred_probs = np.clip(y_pred_probs, 1e-10, 1 - 1e-10)
        
        # Cross-entropy
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        
        ce_loss = -np.sum(sample_weights.reshape(-1, 1) * y_true_onehot * np.log(y_pred_probs)) / np.sum(sample_weights)
        
        # L2 regularization (weight decay)
        l2_reg = (self.lambda_reg / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        
        return ce_loss + l2_reg
    
    def _forward_pass(self, X):
        """
        Forward propagation
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            a1 (hidden activations), a2 (output probabilities)
        """
        # Input to hidden
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        
        # Hidden to output
        z2 = a1 @ self.W2 + self.b2
        a2 = self._softmax(z2)
        
        return a1, a2
    
    def _backward_pass(self, X, y_onehot, a1, a2, sample_weights=None):
        """
        Backward propagation (compute gradients)
        
        Args:
            X: Input features
            y_onehot: True labels (one-hot)
            a1: Hidden layer activations
            a2: Output probabilities
            sample_weights: Sample weights
            
        Returns:
            Gradients for W1, b1, W2, b2
        """
        n_samples = len(X)
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        
        # Output layer gradients
        dz2 = a2 - y_onehot  # Cross-entropy + softmax derivative
        dz2 = dz2 * sample_weights.reshape(-1, 1) / np.sum(sample_weights)
        
        dW2 = a1.T @ dz2 + self.lambda_reg * self.W2  # L2 regularization
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_derivative(X @ self.W1 + self.b1)
        
        dW1 = X.T @ dz1 + self.lambda_reg * self.W1  # L2 regularization
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def _select_features_by_correlation(self, X, y):
        """
        Select features based on correlation with target
        
        Args:
            X: Features
            y: Target labels
            
        Returns:
            Selected feature indices
        """
        n_features = X.shape[1]
        correlations = np.zeros(n_features)
        
        for i in range(n_features):
            # Pearson correlation with target
            correlations[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        
        # Handle NaN correlations (constant features)
        correlations = np.nan_to_num(correlations, nan=0)
        
        # Select top features
        n_to_select = min(self.n_features_to_select, n_features)
        selected_indices = np.argsort(correlations)[-n_to_select:]
        
        return selected_indices
    
    def _standardize_features(self, X, fit=True):
        """
        Standardize features to zero mean and unit variance
        
        Args:
            X: Features
            fit: If True, compute mean and std; if False, use stored values
            
        Returns:
            Standardized features
        """
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_std[self.feature_std == 0] = 1
        
        X_scaled = (X - self.feature_mean) / self.feature_std
        return X_scaled
    
    def fit(self, X, y):
        """
        Train the Neural Network model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            self
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if len(y.shape) != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        np.random.seed(self.random_state)
        
        # Feature selection
        if self.use_feature_selection and X.shape[1] > self.n_features_to_select:
            print(f"Selecting {self.n_features_to_select} features out of {X.shape[1]} using correlation...")
            self.selected_features = self._select_features_by_correlation(X, y)
            X = X[:, self.selected_features]
            print(f"Feature selection complete. Using {X.shape[1]} features.")
        else:
            self.selected_features = np.arange(X.shape[1])
        
        # Feature standardization
        print("Standardizing features...")
        X = self._standardize_features(X, fit=True)
        
        # Train/validation split
        n_samples = X.shape[0]
        n_val = int(self.validation_split * n_samples)
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Calculate class weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        self.class_weights = len(y_train) / (self.n_classes * class_counts)
        
        # Create sample weights
        sample_weights = np.zeros(len(y_train))
        for i, class_val in enumerate(unique_classes):
            sample_weights[y_train == class_val] = self.class_weights[i]
        
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"Class weights: {dict(zip(unique_classes, np.round(self.class_weights, 3)))}")
        
        # One-hot encode labels
        y_train_onehot = self._one_hot_encode(y_train)
        y_val_onehot = self._one_hot_encode(y_val)
        
        # Initialize weights (He initialization for ReLU)
        n_input = X_train.shape[1]
        self.W1 = np.random.randn(n_input, self.hidden_units) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, self.hidden_units))
        self.W2 = np.random.randn(self.hidden_units, self.n_classes) * np.sqrt(2.0 / self.hidden_units)
        self.b2 = np.zeros((1, self.n_classes))
        
        # Training loop
        print(f"\nTraining Neural Network:")
        print(f"  Hidden units: {self.hidden_units}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  L2 lambda: {self.lambda_reg}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max iterations: {self.n_iterations}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_iterations):
            # Mini-batch gradient descent
            n_batches = max(1, len(X_train) // self.batch_size)
            batch_indices = np.array_split(np.random.permutation(len(X_train)), n_batches)
            
            for batch_idx in batch_indices:
                X_batch = X_train[batch_idx]
                y_batch_onehot = y_train_onehot[batch_idx]
                batch_weights = sample_weights[batch_idx]
                
                # Forward pass
                a1, a2 = self._forward_pass(X_batch)
                
                # Backward pass
                dW1, db1, dW2, db2 = self._backward_pass(X_batch, y_batch_onehot, a1, a2, batch_weights)
                
                # Update weights
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
            
            # Compute losses
            _, train_probs = self._forward_pass(X_train)
            train_loss = self._cross_entropy_loss(y_train_onehot, train_probs, sample_weights)
            self.train_losses.append(train_loss)
            
            _, val_probs = self._forward_pass(X_val)
            val_loss = self._cross_entropy_loss(y_val_onehot, val_probs)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_W1, best_b1 = self.W1.copy(), self.b1.copy()
                best_W2, best_b2 = self.W2.copy(), self.b2.copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.n_iterations} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                # Restore best weights
                self.W1, self.b1 = best_W1, best_b1
                self.W2, self.b2 = best_W2, best_b2
                break
        
        print("Training complete!")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if self.W1 is None:
            raise ValueError("Model has not been fitted yet")
        
        # Apply feature selection
        if self.selected_features is not None:
            X = X[:, self.selected_features]
        
        # Apply standardization
        X = self._standardize_features(X, fit=False)
        
        # Forward pass
        _, probs = self._forward_pass(X)
        
        # Return class with highest probability
        predictions = np.argmax(probs, axis=1)
        
        return predictions
