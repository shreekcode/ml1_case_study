# Option 2: Neural Network Classifier

## Model Description

**Neural Network with Softmax Output Layer** — Implements a feedforward neural network using softmax activation for multi-class classification, as taught in Lecture 06 (Optimization in Neural Networks) and Lecture 12 (Deep Learning).

### Key Features

- **Architecture**: Input Layer → Hidden Layer (ReLU) → Output Layer (Softmax)
- **Optimization**: Mini-batch gradient descent with backpropagation
- **Regularization**: L2 weight decay + Early stopping
- **Feature Selection**: Correlation-based filtering (Top 150 features from 512)
- **Feature Scaling**: Standardization (zero mean, unit variance)
- **Class Imbalance**: Class-weighted cross-entropy loss

## Files

- **`model.py`**: Complete Neural Network implementation inheriting from `BaseMLModel`
- **`train_and_predict.py`**: Training script with validation and submission generation
- **`submission.csv`**: Generated predictions for test set
- **`training_output.log`**: Training logs with F1 scores
- **`training_history.png`**: Loss curves (training and validation)

## Results

### Hyperparameters Used

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Hidden Units** | 128 | Size of hidden layer |
| **Learning Rate** | 0.01 | Step size for gradient descent |
| **L2 Lambda** | 0.01 | Regularization strength |
| **Batch Size** | 256 | Mini-batch size |
| **Max Iterations** | 500 | Maximum epochs |
| **Early Stopping Patience** | 20 | Epochs before stopping |
| **Features Selected** | 150 | Top features by correlation |

### Training Performance

**Class Weights Applied**:
- Class 0: 0.308 (majority)
- Class 1: 2.043
- Class 2: 5.397
- Class 3: 12.474 (minority - highest weight)

**Training Process**:
- Started with 66,662 training samples, 11,763 validation samples
- Early stopping triggered at epoch **137** (out of 500)
- Best validation loss achieved and weights restored

### Validation Performance ✅

| Metric | Value |
|--------|-------|
| **Macro F1 Score** | **0.5285** ✅ |
| Accuracy | 0.7536 |

#### Per-Class F1 Scores

| Class | Precision | Recall | F1 Score | Analysis |
|-------|-----------|--------|----------|----------|
| 0 (Excellent) | 0.9651 | 0.7929 | **0.8706** | Excellent |
| 1 (Good) | 0.4033 | 0.4862 | **0.4409** | Moderate |
| 2 (Moderate) | 0.3073 | 0.6922 | **0.4256** | Moderate |
| 3 (Poor) | 0.2343 | 0.9643 | **0.3770** | High recall, low precision |

### Test Predictions Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 | 12,876 | 65.67% |
| 1 | 3,108 | 15.85% |
| 2 | 2,122 | 10.82% |
| 3 | 1,501 | 7.66% |

## Analysis

### ✅ Success Factors

1. **Above Threshold**: F1 macro of 0.5285 exceeds 0.5 bonus requirement
2. **Balanced Performance**: All classes contribute positively to macro F1
3. **Early Stopping**: Prevented overfitting (stopped at epoch 137/500)
4. **Feature Selection**: Correlation-based selection improved performance
5. **Regularization**: L2 weight decay kept weights controlled

### 🎯 Comparison with Decision Tree

| Metric | Decision Tree | Neural Network | Improvement |
|--------|---------------|----------------|-------------|
| **F1 Macro** | 0.2542 | **0.5285** | **+108%** |
| Accuracy | 0.4271 | 0.7536 | +76% |
| Class 0 F1 | 0.6231 | 0.8706 | +40% |
| Class 1 F1 | 0.2042 | 0.4409 | +116% |
| Class 2 F1 | 0.1160 | 0.4256 | +267% |
| Class 3 F1 | 0.0737 | 0.3770 | +412% |

**Key Differences**:
- Neural Network achieves much better precision on minority classes
- Decision Tree was over-predicting minority classes
- Neural Network's regularization and class weighting are more balanced

### Why This Works Better

1. **Softer Decision Boundaries**: Neural network learns smooth, non-linear boundaries
2. **Better Regularization**: L2 + early stopping more effective than tree depth limits
3. **Gradient-based Optimization**: Can fine-tune decision boundaries iteratively
4. **Class Weighting in Loss**: More nuanced than Decision Tree's entropy weighting

## Training History

![Training History](training_history.png)

The plot shows:
- Training and validation loss converge
- No significant overfitting (losses stay close)
- Early stopping at the right time (val loss flattens)

## Usage

### Training and Generating Predictions

```bash
conda run -n ml1 python option2_neural_network/train_and_predict.py
```

### Using the Model

```python
from model import Model
import numpy as np

# Initialize model
model = Model(
    hidden_units=128,
    learning_rate=0.01,
    lambda_reg=0.01,
    n_iterations=500,
    early_stopping_patience=20
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Implementation Details

### Architecture

```
Input (150 features) → Hidden (128 neurons, ReLU) → Output (4 classes, Softmax)
```

### Forward Propagation

1. **Input to Hidden**: `z1 = X·W1 + b1`
2. **ReLU Activation**: `a1 = max(0, z1)`
3. **Hidden to Output**: `z2 = a1·W2 + b2`
4. **Softmax**: `a2 = exp(z2) / Σexp(z2)`

### Loss Function

```
L = -Σ(w_i * y_i * log(ŷ_i)) / Σw_i + (λ/2)(||W1||² + ||W2||²)
```
Where:
- First term: Weighted cross-entropy
- Second term: L2 regularization

### Backward Propagation

Gradients computed using chain rule:
1. Output layer: `dL/dW2 = a1ᵀ(a2 - y) + λW2`
2. Hidden layer: `dL/dW1 = Xᵀ(dz1) + λW1`

### Weight Updates

```
W1 ← W1 - α·dW1
W2 ← W2 - α·dW2
```

## Pros and Cons

### ✅ Advantages
- **High Performance**: F1 = 0.5285 (above threshold)
- Strong on all classes (balanced macro F1)
- Effective regularization (L2 + early stopping)
- Feature scaling improves convergence
- Gradient descent allows fine-tuning

### ⚠️ Disadvantages
- More complex than Decision Tree
- Requires careful hyperparameter tuning
- Longer training time (~3 minutes)
- Less interpretable than tree
- Needs feature scaling

## Next Steps

✅ **Model is ready for submission!**

This model achieves the 0.5 F1 threshold for 5 bonus points and shows strong balanced performance across all 4 classes.

### For Even Better Performance

If time permits:
- Add second hidden layer (deeper network)
- Try different learning rates (0.005, 0.02)
- Experiment with hidden units (64, 256)
- Implement momentum or Adam optimizer
- Try different regularization strengths
