# ============================================================
# DAY 43 — MULTI-LAYER PERCEPTRON (MLP) FROM SCRATCH
# ============================================================
#
# Goal:
# Build and understand a 2-layer neural network using only NumPy.
#
# Architecture:
#
#       Input
#         │
#         ▼
#   Linear Layer 1
#      Z1 = XW1 + b1
#         │
#         ▼
#       ReLU
#         │
#         ▼
#   Linear Layer 2
#      Z2 = A1W2 + b2
#         │
#         ▼
#      Softmax
#         │
#         ▼
#   Class Probabilities
#
# IMPORTANT:
# Day 43 = Forward propagation + Loss
# Day 44 = Backpropagation + Gradients
#
# ============================================================


# ============================================================
# CELL 1 — IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("NumPy version:", np.__version__)


# ============================================================
# CELL 2 — CREATE A SYNTHETIC CLASSIFICATION DATASET
# ============================================================
#
# We create a simple 3-class dataset.
#
# Each sample has 4 input features.
#
# X shape:
#     (number_of_samples, number_of_features)
#
# y:
#     Integer class labels: 0, 1, 2
#
# ============================================================

n_samples = 300
n_features = 4
n_classes = 3

X = np.random.randn(n_samples, n_features)

# Create labels using a simple relationship between features.
scores = np.column_stack([
    X[:, 0] + 0.5 * X[:, 1],
    X[:, 1] - X[:, 2],
    X[:, 2] + X[:, 3]
])

y = np.argmax(scores, axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))
print("Class distribution:", np.bincount(y))


# ============================================================
# CELL 3 — TRAIN / TEST SPLIT
# ============================================================
#
# We are not training yet, but keeping a train/test split
# establishes proper ML engineering practice.
#
# ============================================================

indices = np.random.permutation(len(X))

split_index = int(0.8 * len(X))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CELL 4 — ONE-HOT ENCODING
# ============================================================
#
# Example:
#
# class 0 → [1, 0, 0]
# class 1 → [0, 1, 0]
# class 2 → [0, 0, 1]
#
# Useful for understanding categorical cross entropy.
#
# ============================================================

def one_hot_encode(labels, num_classes):
    """
    Convert integer class labels into one-hot vectors.
    """
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1

    return encoded


y_train_one_hot = one_hot_encode(y_train, n_classes)

print("One-hot shape:", y_train_one_hot.shape)
print("Example label:", y_train[0])
print("One-hot label:", y_train_one_hot[0])


# ============================================================
# CELL 5 — RELU ACTIVATION
# ============================================================
#
# ReLU:
#
#     ReLU(x) = max(0, x)
#
# Negative values → 0
# Positive values → unchanged
#
# ============================================================

def relu(x):
    """
    Rectified Linear Unit activation.
    """
    return np.maximum(0, x)


# Test ReLU

test_values = np.array([-3, -1, 0, 1, 3])

print("Input :", test_values)
print("ReLU  :", relu(test_values))


# ============================================================
# CELL 6 — SOFTMAX ACTIVATION
# ============================================================
#
# Softmax converts logits into probabilities.
#
# For each sample:
#
#             exp(z_i)
# P_i = ---------------------
#        sum(exp(z_j))
#
# Important:
# We subtract the maximum logit before exponentiation.
#
# This does NOT change the final probability distribution,
# but greatly improves numerical stability.
#
# ============================================================

def softmax(logits):
    """
    Numerically stable softmax.
    """

    # Subtract maximum value from each row.
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(shifted_logits)

    probabilities = (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )

    return probabilities


# Test softmax

test_logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0]
])

test_probabilities = softmax(test_logits)

print("Softmax probabilities:")
print(test_probabilities)

print("\nProbability sums:")
print(test_probabilities.sum(axis=1))


# ============================================================
# CELL 7 — CROSS ENTROPY LOSS
# ============================================================
#
# For one-hot classification:
#
#     Loss = -Σ y_i log(p_i)
#
# For the correct class:
#
#     Loss = -log(probability_of_correct_class)
#
# We add a tiny epsilon to prevent:
#
#     log(0)
#
# ============================================================

def cross_entropy_loss(y_true, y_pred):
    """
    Calculate average categorical cross-entropy loss.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded labels.

    y_pred : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Mean cross-entropy loss.
    """

    epsilon = 1e-12

    y_pred_clipped = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred_clipped),
        axis=1
    )

    return np.mean(loss)


# Example

example_true = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

example_pred = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1]
])

loss = cross_entropy_loss(
    example_true,
    example_pred
)

print("Example cross-entropy loss:", loss)


# ============================================================
# CELL 8 — LINEAR LAYER
# ============================================================
#
# Every fully connected neural-network layer performs:
#
#     Z = XW + b
#
# X = input
# W = weights
# b = bias
# Z = output / logits before activation
#
# ============================================================

class LinearLayer:
    """
    Fully connected / dense neural-network layer.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize weights and biases.
        """

        self.weights = (
            np.random.randn(input_size, output_size)
            * 0.01
        )

        self.bias = np.zeros(
            (1, output_size)
        )

    def forward(self, x):
        """
        Forward propagation through linear layer.
        """

        return x @ self.weights + self.bias


# ============================================================
# CELL 9 — TEST LINEAR LAYER
# ============================================================

linear = LinearLayer(
    input_size=4,
    output_size=8
)

sample_input = np.random.randn(5, 4)

sample_output = linear.forward(sample_input)

print("Input shape :", sample_input.shape)
print("Weights     :", linear.weights.shape)
print("Bias        :", linear.bias.shape)
print("Output shape:", sample_output.shape)


# ============================================================
# CELL 10 — BUILD THE COMPLETE MLP
# ============================================================
#
# Architecture:
#
# Input:  4 features
#
#       ↓
#
# Linear:
# 4 → 8
#
#       ↓
#
# ReLU
#
#       ↓
#
# Linear:
# 8 → 3
#
#       ↓
#
# Softmax
#
#       ↓
#
# 3 class probabilities
#
# ============================================================

class MLP:
    """
    Two-layer Multi-Layer Perceptron.

    Architecture:
        Input → Linear → ReLU → Linear → Softmax
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        """
        Initialize network layers.
        """

        self.layer1 = LinearLayer(
            input_size,
            hidden_size
        )

        self.layer2 = LinearLayer(
            hidden_size,
            output_size
        )

    def forward(self, x):
        """
        Complete forward propagation.
        """

        # First linear transformation.
        z1 = self.layer1.forward(x)

        # Non-linear activation.
        a1 = relu(z1)

        # Second linear transformation.
        z2 = self.layer2.forward(a1)

        # Convert logits to probabilities.
        probabilities = softmax(z2)

        return probabilities

    def predict(self, x):
        """
        Return predicted class indices.
        """

        probabilities = self.forward(x)

        return np.argmax(
            probabilities,
            axis=1
        )


# ============================================================
# CELL 11 — CREATE THE MODEL
# ============================================================

input_size = n_features
hidden_size = 8
output_size = n_classes

model = MLP(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

print("MLP created successfully.")


# ============================================================
# CELL 12 — INSPECT MODEL PARAMETERS
# ============================================================

print("Layer 1 weights:")
print(model.layer1.weights.shape)

print("\nLayer 1 bias:")
print(model.layer1.bias.shape)

print("\nLayer 2 weights:")
print(model.layer2.weights.shape)

print("\nLayer 2 bias:")
print(model.layer2.bias.shape)


# ============================================================
# CELL 13 — FORWARD PROPAGATION
# ============================================================

predicted_probabilities = model.forward(X_train)

print("Prediction shape:")
print(predicted_probabilities.shape)

print("\nFirst 5 predictions:")
print(predicted_probabilities[:5])

print("\nProbability sums:")
print(predicted_probabilities[:5].sum(axis=1))


# ============================================================
# CELL 14 — CALCULATE LOSS
# ============================================================

initial_loss = cross_entropy_loss(
    y_train_one_hot,
    predicted_probabilities
)

print("Initial loss:", initial_loss)


# ============================================================
# CELL 15 — MAKE PREDICTIONS
# ============================================================

predicted_classes = model.predict(X_train)

print("Predicted classes:")
print(predicted_classes[:20])

print("\nActual classes:")
print(y_train[:20])


# ============================================================
# CELL 16 — CALCULATE ACCURACY
# ============================================================

def accuracy_score(y_true, y_pred):
    """
    Calculate classification accuracy.
    """

    return np.mean(y_true == y_pred)


train_accuracy = accuracy_score(
    y_train,
    predicted_classes
)

print("Initial training accuracy:", train_accuracy)


# ============================================================
# CELL 17 — PARAMETER COUNT
# ============================================================
#
# Layer 1:
#
#     weights = 4 × 8 = 32
#     biases  = 8
#
#     total = 40
#
# Layer 2:
#
#     weights = 8 × 3 = 24
#     biases  = 3
#
#     total = 27
#
# Total:
#
#     40 + 27 = 67 parameters
#
# ============================================================

def count_parameters(model):
    """
    Count trainable parameters in the MLP.
    """

    layer1_params = (
        model.layer1.weights.size
        + model.layer1.bias.size
    )

    layer2_params = (
        model.layer2.weights.size
        + model.layer2.bias.size
    )

    return layer1_params + layer2_params


total_parameters = count_parameters(model)

print("Total trainable parameters:", total_parameters)


# ============================================================
# CELL 18 — MANUALLY VERIFY PARAMETER COUNT
# ============================================================

layer1_weights = (
    input_size * hidden_size
)

layer1_bias = hidden_size

layer2_weights = (
    hidden_size * output_size
)

layer2_bias = output_size

manual_parameter_count = (
    layer1_weights
    + layer1_bias
    + layer2_weights
    + layer2_bias
)

print("Layer 1 weights:", layer1_weights)
print("Layer 1 bias:", layer1_bias)

print("Layer 2 weights:", layer2_weights)
print("Layer 2 bias:", layer2_bias)

print("\nTotal:", manual_parameter_count)


# ============================================================
# CELL 19 — VERIFY FORWARD PASS STEP BY STEP
# ============================================================
#
# Instead of hiding everything inside model.forward(),
# let's explicitly inspect each stage.
#
# ============================================================

X_batch = X_train[:5]

# Step 1 — Linear transformation
Z1 = X_batch @ model.layer1.weights + model.layer1.bias

# Step 2 — ReLU
A1 = relu(Z1)

# Step 3 — Second linear transformation
Z2 = A1 @ model.layer2.weights + model.layer2.bias

# Step 4 — Softmax
A2 = softmax(Z2)

print("X shape :", X_batch.shape)
print("Z1 shape:", Z1.shape)
print("A1 shape:", A1.shape)
print("Z2 shape:", Z2.shape)
print("A2 shape:", A2.shape)


# ============================================================
# CELL 20 — VISUALIZE THE NETWORK PIPELINE
# ============================================================

print(
    """
    INPUT
      │
      │ X
      ▼
┌──────────────────┐
│   Linear Layer   │
│    Z1 = XW1+b1   │
└──────────────────┘
      │
      │ Z1
      ▼
┌──────────────────┐
│      ReLU        │
│   A1 = ReLU(Z1)  │
└──────────────────┘
      │
      │ A1
      ▼
┌──────────────────┐
│   Linear Layer   │
│    Z2 = A1W2+b2  │
└──────────────────┘
      │
      │ Z2 = logits
      ▼
┌──────────────────┐
│     Softmax      │
│   probabilities  │
└──────────────────┘
      │
      ▼
   Prediction
"""
)


# ============================================================
# CELL 21 — UNDERSTAND LOGITS VS PROBABILITIES
# ============================================================

print("Logits:")
print(Z2)

print("\nProbabilities:")
print(A2)

print("\nPredicted classes:")
print(np.argmax(A2, axis=1))


# ============================================================
# CELL 22 — HIDDEN LAYER ACTIVATIONS
# ============================================================
#
# ReLU creates sparse activations because negative values
# become zero.
#
# This is useful for understanding what happens internally.
#
# ============================================================

zero_fraction = np.mean(A1 == 0)

print(
    "Fraction of hidden activations equal to zero:",
    zero_fraction
)


# ============================================================
# CELL 23 — HIDDEN SIZE EXPERIMENT
# ============================================================
#
# Change hidden layer size and observe parameter count.
#
# This builds intuition about model capacity.
#
# ============================================================

hidden_sizes = [4, 8, 16, 32, 64, 128]

for hidden in hidden_sizes:

    experiment_model = MLP(
        input_size=input_size,
        hidden_size=hidden,
        output_size=output_size
    )

    params = count_parameters(
        experiment_model
    )

    print(
        f"Hidden units: {hidden:3d} "
        f"| Parameters: {params:5d}"
    )


# ============================================================
# CELL 24 — GENERALIZE THE NETWORK TO MULTIPLE HIDDEN LAYERS
# ============================================================
#
# This is an engineering extension.
#
# Instead of hard-coding:
#
#     Linear → ReLU → Linear
#
# we create arbitrary hidden-layer architectures.
#
# Example:
#
#     4 → 16 → 32 → 16 → 3
#
# ============================================================

class DeepMLP:
    """
    Flexible MLP supporting multiple hidden layers.
    """

    def __init__(
        self,
        layer_sizes
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            Example:
                [4, 16, 32, 3]
        """

        self.layers = []

        for i in range(
            len(layer_sizes) - 1
        ):
            self.layers.append(
                LinearLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1]
                )
            )

    def forward(self, x):
        """
        Forward propagation through
        all layers.
        """

        for i, layer in enumerate(
            self.layers
        ):

            x = layer.forward(x)

            # Apply ReLU to every hidden layer.
            if i < len(self.layers) - 1:
                x = relu(x)

        return softmax(x)

    def predict(self, x):
        """
        Return predicted class.
        """

        probabilities = self.forward(x)

        return np.argmax(
            probabilities,
            axis=1
        )

    def count_parameters(self):
        """
        Count all network parameters.
        """

        total = 0

        for layer in self.layers:
            total += layer.weights.size
            total += layer.bias.size

        return total


# ============================================================
# CELL 25 — TEST DEEP MLP
# ============================================================

deep_model = DeepMLP(
    layer_sizes=[
        4,
        16,
        32,
        16,
        3
    ]
)

deep_predictions = deep_model.predict(
    X_test
)

print("Deep MLP prediction shape:")
print(deep_predictions.shape)

print(
    "\nDeep MLP parameter count:",
    deep_model.count_parameters()
)


# ============================================================
# CELL 26 — VERIFY SOFTMAX PROBABILITIES
# ============================================================

probabilities = model.forward(
    X_test[:10]
)

for i, probs in enumerate(
    probabilities
):

    print(
        f"Sample {i}: "
        f"sum={probs.sum():.6f}, "
        f"prediction={np.argmax(probs)}"
    )


# ============================================================
# CELL 27 — INITIAL MODEL PERFORMANCE
# ============================================================
#
# IMPORTANT:
#
# The model has NOT learned yet.
#
# We have not implemented:
#
#     gradients
#     backpropagation
#     optimizer
#     weight updates
#
# Therefore, this accuracy is essentially the performance
# of a randomly initialized neural network.
#
# ============================================================

test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("Test accuracy before training:", test_accuracy)


# ============================================================
# CELL 28 — WHY THE MODEL DOES NOT LEARN YET
# ============================================================

print(
    """
    CURRENT PIPELINE

    X
    │
    ▼
    Forward Pass
    │
    ▼
    Prediction
    │
    ▼
    Loss
    │
    X

    We STOP here.

    We do NOT yet have:

    Loss
      │
      ▼
    Backpropagation
      │
      ▼
    Gradients
      │
      ▼
    Optimizer
      │
      ▼
    Updated Weights
      │
      └───────────────► Forward Pass again

    DAY 44 will implement this learning loop.
    """
)


# ============================================================
# CELL 29 — FINAL DAY 43 SUMMARY
# ============================================================

print(
    """
    ========================================================
                     DAY 43 COMPLETED
    ========================================================

    Implemented from scratch:

    ✓ NumPy-based neural network
    ✓ Linear layer
    ✓ Weights
    ✓ Biases
    ✓ ReLU
    ✓ Softmax
    ✓ Cross-entropy loss
    ✓ Forward propagation
    ✓ Prediction
    ✓ Accuracy
    ✓ Parameter counting
    ✓ Hidden-layer inspection
    ✓ Variable hidden-layer architectures
    ✓ Deep MLP extension

    NOT IMPLEMENTED YET:

    ✗ Backpropagation
    ✗ Gradients
    ✗ Gradient descent
    ✗ SGD
    ✗ Adam
    ✗ Weight updates
    ✗ Training loop

    Those belong to Day 44.
    ========================================================
    """
)


# ============================================================
# DAY 43 — THEORY RECAP
# ============================================================
#
# Keep this section at the END of the notebook.
# These comments are designed for quick revision later.
#
# ------------------------------------------------------------
# 1. WHAT IS AN MLP?
# ------------------------------------------------------------
#
# MLP = Multi-Layer Perceptron.
#
# It is a feed-forward neural network consisting primarily
# of fully connected / dense layers.
#
# Example:
#
# Input → Linear → ReLU → Linear → Softmax
#
#
# ------------------------------------------------------------
# 2. WHY DO WE NEED NEURAL NETWORKS?
# ------------------------------------------------------------
#
# Classical linear models learn linear relationships:
#
#     y = XW + b
#
# A neural network adds nonlinear activation functions:
#
#     Z = XW + b
#     A = f(Z)
#
# This allows the network to represent nonlinear functions.
#
#
# ------------------------------------------------------------
# 3. LINEAR LAYER
# ------------------------------------------------------------
#
# Core equation:
#
#     Z = XW + b
#
# X = input matrix
# W = weight matrix
# b = bias vector
# Z = output before activation
#
# Example:
#
#     X:  (N, 4)
#     W:  (4, 8)
#     b:  (1, 8)
#     Z:  (N, 8)
#
#
# ------------------------------------------------------------
# 4. WHY BIAS?
# ------------------------------------------------------------
#
# Without bias:
#
#     Z = XW
#
# With bias:
#
#     Z = XW + b
#
# Bias allows the activation/function to shift instead of
# being forced through the origin.
#
#
# ------------------------------------------------------------
# 5. WHY ACTIVATION FUNCTIONS?
# ------------------------------------------------------------
#
# Without activation:
#
#     XW1W2W3
#
# is still just a linear transformation.
#
# Even multiple linear layers collapse mathematically into
# one equivalent linear transformation.
#
# Nonlinear activations prevent this collapse.
#
#
# ------------------------------------------------------------
# 6. RELU
# ------------------------------------------------------------
#
# Formula:
#
#     ReLU(x) = max(0, x)
#
# Positive values remain.
# Negative values become zero.
#
# Advantages:
#
# ✓ Simple
# ✓ Computationally cheap
# ✓ Helps reduce vanishing gradients compared with sigmoid
#
# Limitation:
#
# ✗ Dead ReLU problem can occur when neurons consistently
#   produce negative inputs.
#
#
# ------------------------------------------------------------
# 7. SOFTMAX
# ------------------------------------------------------------
#
# Softmax converts logits into a probability distribution.
#
# Formula:
#
#             exp(z_i)
# P_i = ---------------------
#        Σ exp(z_j)
#
# Properties:
#
#     0 <= P_i <= 1
#
#     Σ P_i = 1
#
# Usually used for multi-class classification output layers.
#
#
# ------------------------------------------------------------
# 8. NUMERICALLY STABLE SOFTMAX
# ------------------------------------------------------------
#
# Instead of:
#
#     exp(z)
#
# directly, use:
#
#     exp(z - max(z))
#
# This prevents extremely large exponential values.
#
# Important engineering principle:
#
# NUMERICAL STABILITY MATTERS.
#
#
# ------------------------------------------------------------
# 9. LOGITS
# ------------------------------------------------------------
#
# The output of the final linear layer before softmax:
#
#     Z2 = A1W2 + b2
#
# is called logits.
#
# Logits are NOT probabilities.
#
# Softmax(logits) → probabilities.
#
#
# ------------------------------------------------------------
# 10. CROSS-ENTROPY LOSS
# ------------------------------------------------------------
#
# For a correct class probability p:
#
#     Loss = -log(p)
#
# High confidence in the correct class:
#
#     p → 1
#     loss → 0
#
# Low confidence:
#
#     p → 0
#     loss → very large
#
#
# ------------------------------------------------------------
# 11. COMPLETE FORWARD PASS
# ------------------------------------------------------------
#
# For our 2-layer network:
#
#     Z1 = XW1 + b1
#
#     A1 = ReLU(Z1)
#
#     Z2 = A1W2 + b2
#
#     A2 = Softmax(Z2)
#
#     Loss = CrossEntropy(y, A2)
#
#
# ------------------------------------------------------------
# 12. TENSOR SHAPES
# ------------------------------------------------------------
#
# Example:
#
#     X  = (N, 4)
#     W1 = (4, 8)
#     b1 = (1, 8)
#
#     Z1 = (N, 8)
#
#     W2 = (8, 3)
#     b2 = (1, 3)
#
#     Z2 = (N, 3)
#
#     A2 = (N, 3)
#
# Shape reasoning is one of the most important neural
# network debugging skills.
#
#
# ------------------------------------------------------------
# 13. PARAMETERS
# ------------------------------------------------------------
#
# A parameter is learned during training.
#
# Examples:
#
#     weights
#     biases
#
# Hyperparameters are selected by us:
#
#     learning rate
#     batch size
#     number of layers
#     hidden size
#     number of epochs
#
#
# ------------------------------------------------------------
# 14. PARAMETER COUNT
# ------------------------------------------------------------
#
# For a fully connected layer:
#
#     parameters =
#         input_size × output_size
#         + output_size
#
# because:
#
#     weights = input_size × output_size
#     biases  = output_size
#
#
# ------------------------------------------------------------
# 15. WHY NUMPY?
# ------------------------------------------------------------
#
# NumPy forces us to understand the underlying mathematics.
#
# We manually implement:
#
#     matrix multiplication
#     activations
#     probability conversion
#     loss
#     parameter storage
#
# This prevents neural networks from becoming a black box.
#
#
# ------------------------------------------------------------
# 16. WHY NOT TRAIN TODAY?
# ------------------------------------------------------------
#
# A neural network learns by changing its parameters.
#
# To change parameters we need:
#
#     Loss
#       ↓
#     Gradient
#       ↓
#     Parameter update
#
# Finding those gradients efficiently is the job of
# BACKPROPAGATION.
#
# Therefore:
#
# Day 43 → Forward propagation
#
# Day 44 → Backpropagation
#
#
# ------------------------------------------------------------
# 17. COMPLETE LEARNING PIPELINE
# ------------------------------------------------------------
#
# Eventually the complete training process becomes:
#
#       Input
#         ↓
#   Forward Propagation
#         ↓
#    Predictions
#         ↓
#        Loss
#         ↓
#   Backpropagation
#         ↓
#      Gradients
#         ↓
#     Optimizer
#         ↓
#   Update Weights
#         ↓
#       Repeat
#
#
# ------------------------------------------------------------
# 18. CONNECTION TO CLASSICAL ML
# ------------------------------------------------------------
#
# Linear Regression:
#
#     y_hat = XW + b
#
# Neural Network:
#
#     Z = XW + b
#     A = activation(Z)
#
# So neural networks do NOT replace the mathematics we learned.
#
# They build more expressive architectures on top of it.
#
#
# ------------------------------------------------------------
# 19. CONNECTION TO PYTORCH
# ------------------------------------------------------------
#
# Our NumPy implementation:
#
#     LinearLayer
#     ReLU
#     Softmax
#
# PyTorch provides optimized abstractions:
#
#     nn.Linear
#     nn.ReLU
#     nn.Softmax
#
# PyTorch additionally provides:
#
#     Autograd
#     GPU acceleration
#     Optimizers
#     DataLoaders
#     Automatic differentiation
#     Mixed precision
#     Distributed training
#
# The mathematics remains fundamentally the same.
#
#
# ------------------------------------------------------------
# 20. CONNECTION TO CNNs
# ------------------------------------------------------------
#
# CNN:
#
#     Convolution → Activation → Pooling → ...
#
# MLP:
#
#     Linear → Activation → Linear → ...
#
# The architecture changes, but the fundamental pattern
# remains:
#
#     Transformation → Nonlinearity → Transformation
#
#
# ------------------------------------------------------------
# 21. CONNECTION TO TRANSFORMERS
# ------------------------------------------------------------
#
# Transformers also rely heavily on:
#
#     Matrix multiplication
#     Linear layers
#     Softmax
#     Nonlinear activations
#     Residual connections
#     Normalization
#
# Understanding today's MLP makes Transformer internals
# significantly easier later.
#
#
# ------------------------------------------------------------
# 22. IMPORTANT DAY 43 LIMITATIONS
# ------------------------------------------------------------
#
# This implementation is educational, not production-ready.
#
# Missing:
#
#     Backpropagation
#     Automatic differentiation
#     Optimized GPU kernels
#     Mini-batch DataLoader
#     Checkpointing
#     Learning-rate scheduling
#     Mixed precision
#     Distributed training
#     Experiment tracking
#
# We deliberately add these concepts step-by-step.
#
#
# ============================================================
# FINAL MENTAL MODEL
# ============================================================
#
# Remember:
#
#       NEURAL NETWORK
#
#       X
#       │
#       ▼
#    XW1 + b1
#       │
#       ▼
#      ReLU
#       │
#       ▼
#    XW2 + b2
#       │
#       ▼
#     Softmax
#       │
#       ▼
#  Probabilities
#       │
#       ▼
#      Loss
#
# Day 43 = "How does a neural network COMPUTE?"
#
# Day 44 = "How does a neural network LEARN?"
#
# ============================================================