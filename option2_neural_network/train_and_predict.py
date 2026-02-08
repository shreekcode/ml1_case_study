"""
Training and Prediction Script for Neural Network Classifier

This script:
1. Loads training and test data
2. Trains the Neural Network model with early stopping
3. Evaluates on validation set using F1 macro score
4. Generates predictions for test set
5. Creates submission.csv
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model


def calculate_f1_macro(y_true, y_pred, num_classes=4):
    """
    Calculate macro-averaged F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Macro F1 score
    """
    f1_scores = []
    
    for class_i in range(num_classes):
        # Binary classification for this class vs. rest
        tp = np.sum((y_true == class_i) & (y_pred == class_i))
        fp = np.sum((y_true != class_i) & (y_pred == class_i))
        fn = np.sum((y_true == class_i) & (y_pred != class_i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        print(f"  Class {class_i}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    macro_f1 = np.mean(f1_scores)
    return macro_f1


def main():
    print("="*70)
    print("Improved Neural Network Classifier - 2 Hidden Layers")
    print("="*70)
    
    # Load training data
    print("\n[1/5] Loading training data...")
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_df = pd.read_csv(os.path.join(dataset_path, 'train_df.csv'))
    print(f"Training data shape: {train_df.shape}")
    
    # Separate features and target
    X_full = train_df.drop(['ID', 'num_errors'], axis=1).values
    y_full = train_df['num_errors'].values
    
    print(f"Features shape: {X_full.shape}")
    print(f"Target shape: {y_full.shape}")
    
    # Train model (with internal validation split for early    # Train model
    print("\n[2/5] Training Improved Neural Network model (2 Hidden Layers)...")
    model = Model(
        hidden_units_1=256,  # First hidden layer
        hidden_units_2=128,  # Second hidden layer
        learning_rate=0.01,
        lambda_reg=0.01,
        n_iterations=500,
        batch_size=256,
        early_stopping_patience=20,
        n_features_to_select=150,
        use_feature_selection=True,
        validation_split=0.15,
        random_state=42
    )
    
    model.fit(X_full, y_full)
    
    # Create separate validation set for evaluation
    print("\n[3/5] Evaluating on final validation set...")
    np.random.seed(42)
    n_samples = X_full.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(0.85 * n_samples)
    
    X_train_eval = X_full[indices[:split_idx]]
    y_train_eval = y_full[indices[:split_idx]]
    X_val_eval = X_full[indices[split_idx:]]
    y_val_eval = y_full[indices[split_idx:]]
    
    y_val_pred = model.predict(X_val_eval)
    
    print("\nValidation F1 Scores by Class:")
    val_f1 = calculate_f1_macro(y_val_eval, y_val_pred)
    print(f"\n>>> Validation Macro F1 Score: {val_f1:.4f}")
    
    if val_f1 > 0.5:
        print("✅ F1 Score > 0.5 - Eligible for 5 bonus points!")
    else:
        print("⚠️  F1 Score <= 0.5 - Below bonus point threshold")
    
    # Calculate accuracy for reference
    val_accuracy = np.mean(y_val_eval == y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f} (for reference only)")
    
    # Load test data and make predictions
    print("\n[4/5] Generating predictions for test set...")
    test_df = pd.read_csv(os.path.join(dataset_path, 'test_df.csv'))
    test_indices = test_df['index'].values
    X_test = test_df.drop(['index'], axis=1).values
    
    print(f"Test data shape: {X_test.shape}")
    
    y_test_pred = model.predict(X_test)
    
    # Create submission file
    print("\n[5/5] Creating submission file...")
    submission_df = pd.DataFrame({
        'index': test_indices,
        'Predicted': y_test_pred
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print(f"\n✅ Submission file created: submission.csv")
    print(f"   Shape: {submission_df.shape}")
    print(f"   Sample predictions:\n{submission_df.head(10)}")
    
    # Verify predictions
    unique_preds = np.unique(y_test_pred)
    print(f"\nPrediction value counts:")
    for pred_class in range(4):
        count = np.sum(y_test_pred == pred_class)
        pct = 100 * count / len(y_test_pred)
        print(f"  Class {pred_class}: {count} ({pct:.2f}%)")
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(model.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(model.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("\n📊 Training history plot saved: training_history.png")
    except Exception as e:
        print(f"\nNote: Could not create training plot: {e}")
    
    print("\n" + "="*70)
    print("Neural Network Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
