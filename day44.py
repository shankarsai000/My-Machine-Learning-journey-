# ============================================================
# DAY 44 — BACKPROPAGATION FROM SCRATCH
# ============================================================
#
# Goal:
#   Implement backpropagation manually using NumPy.
#
# Architecture:
#
#       X
#       │
#       ▼
#   Linear 1
#       │
#       ▼
#      ReLU
#       │
#       ▼
#   Linear 2
#       │
#       ▼
#    Softmax
#       │
#       ▼
#      Loss
#
# Then:
#
#      Loss
#        │
#        ▼
#   Backpropagation
#        │
#        ▼
#     Gradients
#        │
#        ▼
#       SGD
#        │
#        ▼
#   Updated Parameters
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("NumPy version:", np.__version__)


# ============================================================
# CELL 1 — CREATE DATA
# ============================================================

n_samples = 300
n_features = 4
n_classes = 3

X = np.random.randn(
    n_samples,
    n_features
)

scores = np.column_stack([
    X[:, 0] + 0.5 * X[:, 1],
    X[:, 1] - X[:, 2],
    X[:, 2] + X[:, 3]
])

y = np.argmax(
    scores,
    axis=1
)


# ============================================================
# CELL 2 — TRAIN / TEST SPLIT
# ============================================================

indices = np.random.permutation(
    n_samples
)

split_index = int(
    0.8 * n_samples
)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# ============================================================
# CELL 3 — ONE-HOT ENCODING
# ============================================================

def one_hot_encode(labels, num_classes):
    """
    Convert integer labels into one-hot vectors.
    """

    encoded = np.zeros(
        (len(labels), num_classes)
    )

    encoded[
        np.arange(len(labels)),
        labels
    ] = 1

    return encoded


Y_train = one_hot_encode(
    y_train,
    n_classes
)


# ============================================================
# CELL 4 — ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    """
    ReLU activation.

    ReLU(x) = max(0, x)
    """

    return np.maximum(
        0,
        x
    )


def relu_derivative(x):
    """
    Derivative of ReLU.

    1 when x > 0
    0 otherwise
    """

    return (
        x > 0
    ).astype(float)


def softmax(logits):
    """
    Numerically stable softmax.
    """

    shifted_logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        shifted_logits
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# CELL 5 — LOSS FUNCTION
# ============================================================

def cross_entropy_loss(
    y_true,
    y_pred
):
    """
    Mean categorical cross-entropy.
    """

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred),
        axis=1
    )

    return np.mean(loss)


# ============================================================
# CELL 6 — MLP WITH MANUAL BACKPROPAGATION
# ============================================================

class MLP:
    """
    Two-layer MLP implemented entirely with NumPy.

    Architecture:

        Input
          ↓
        Linear
          ↓
        ReLU
          ↓
        Linear
          ↓
        Softmax
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        # He-style initialization for ReLU.
        self.W1 = (
            np.random.randn(
                input_size,
                hidden_size
            )
            * np.sqrt(
                2.0 / input_size
            )
        )

        self.b1 = np.zeros(
            (1, hidden_size)
        )

        self.W2 = (
            np.random.randn(
                hidden_size,
                output_size
            )
            * np.sqrt(
                2.0 / hidden_size
            )
        )

        self.b2 = np.zeros(
            (1, output_size)
        )

        # Cache stores intermediate values
        # required during backpropagation.
        self.cache = {}

        # Gradients.
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    def forward(self, X):
        """
        Perform forward propagation.
        """

        # First linear transformation.
        Z1 = (
            X @ self.W1
            + self.b1
        )

        # ReLU activation.
        A1 = relu(Z1)

        # Second linear transformation.
        Z2 = (
            A1 @ self.W2
            + self.b2
        )

        # Convert logits to probabilities.
        A2 = softmax(Z2)

        # Save values needed for backward pass.
        self.cache["X"] = X
        self.cache["Z1"] = Z1
        self.cache["A1"] = A1
        self.cache["Z2"] = Z2
        self.cache["A2"] = A2

        return A2


    # --------------------------------------------------------
    # BACKWARD PASS
    # --------------------------------------------------------

    def backward(self, Y):
        """
        Manually calculate gradients.

        Y = one-hot encoded true labels.
        """

        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        batch_size = X.shape[0]

        # ----------------------------------------------------
        # Gradient of Softmax + Cross Entropy
        #
        # dZ2 = A2 - Y
        # ----------------------------------------------------

        dZ2 = (
            A2 - Y
        ) / batch_size

        # ----------------------------------------------------
        # Gradient of W2
        #
        # Z2 = A1W2 + b2
        #
        # dW2 = A1.T @ dZ2
        # ----------------------------------------------------

        self.dW2 = (
            A1.T @ dZ2
        )

        # ----------------------------------------------------
        # Gradient of b2
        # ----------------------------------------------------

        self.db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # Propagate gradient backward
        #
        # dA1 = dZ2 @ W2.T
        # ----------------------------------------------------

        dA1 = (
            dZ2 @ self.W2.T
        )

        # ----------------------------------------------------
        # Backprop through ReLU
        #
        # dZ1 = dA1 * ReLU'(Z1)
        # ----------------------------------------------------

        dZ1 = (
            dA1
            * relu_derivative(Z1)
        )

        # ----------------------------------------------------
        # Gradient of W1
        #
        # Z1 = XW1 + b1
        #
        # dW1 = X.T @ dZ1
        # ----------------------------------------------------

        self.dW1 = (
            X.T @ dZ1
        )

        # ----------------------------------------------------
        # Gradient of b1
        # ----------------------------------------------------

        self.db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )


    # --------------------------------------------------------
    # PARAMETER UPDATE
    # --------------------------------------------------------

    def update_parameters(
        self,
        learning_rate
    ):
        """
        Perform one SGD parameter update.
        """

        self.W1 -= (
            learning_rate
            * self.dW1
        )

        self.b1 -= (
            learning_rate
            * self.db1
        )

        self.W2 -= (
            learning_rate
            * self.dW2
        )

        self.b2 -= (
            learning_rate
            * self.db2
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, X):
        """
        Return predicted class indices.
        """

        probabilities = self.forward(
            X
        )

        return np.argmax(
            probabilities,
            axis=1
        )


    # --------------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------------

    def count_parameters(self):
        """
        Count all trainable parameters.
        """

        return (
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
        )


# ============================================================
# CELL 7 — CREATE MODEL
# ============================================================

model = MLP(
    input_size=n_features,
    hidden_size=16,
    output_size=n_classes
)

print(
    "Trainable parameters:",
    model.count_parameters()
)


# ============================================================
# CELL 8 — FORWARD PASS BEFORE TRAINING
# ============================================================

predictions = model.forward(
    X_train
)

initial_loss = cross_entropy_loss(
    Y_train,
    predictions
)

initial_accuracy = np.mean(
    np.argmax(
        predictions,
        axis=1
    )
    == y_train
)

print(
    f"Initial loss: "
    f"{initial_loss:.4f}"
)

print(
    f"Initial accuracy: "
    f"{initial_accuracy:.4f}"
)


# ============================================================
# CELL 9 — BACKPROPAGATION
# ============================================================

model.backward(
    Y_train
)

print("Gradients calculated successfully.")

print(
    "dW1 shape:",
    model.dW1.shape
)

print(
    "db1 shape:",
    model.db1.shape
)

print(
    "dW2 shape:",
    model.dW2.shape
)

print(
    "db2 shape:",
    model.db2.shape
)


# ============================================================
# CELL 10 — INSPECT GRADIENTS
# ============================================================

print(
    "dW1 mean:",
    np.mean(model.dW1)
)

print(
    "dW1 std:",
    np.std(model.dW1)
)

print(
    "dW2 mean:",
    np.mean(model.dW2)
)

print(
    "dW2 std:",
    np.std(model.dW2)
)


# ============================================================
# CELL 11 — TRAINING LOOP
# ============================================================

learning_rate = 0.1
epochs = 1000

loss_history = []
accuracy_history = []

for epoch in range(epochs):

    # ----------------------------
    # 1. Forward pass
    # ----------------------------

    probabilities = model.forward(
        X_train
    )

    # ----------------------------
    # 2. Calculate loss
    # ----------------------------

    loss = cross_entropy_loss(
        Y_train,
        probabilities
    )

    # ----------------------------
    # 3. Backpropagation
    # ----------------------------

    model.backward(
        Y_train
    )

    # ----------------------------
    # 4. Update parameters
    # ----------------------------

    model.update_parameters(
        learning_rate
    )

    # ----------------------------
    # 5. Calculate accuracy
    # ----------------------------

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = np.mean(
        predictions == y_train
    )

    loss_history.append(
        loss
    )

    accuracy_history.append(
        accuracy
    )

    # ----------------------------
    # Logging
    # ----------------------------

    if (
        epoch == 0
        or (epoch + 1) % 100 == 0
    ):

        print(
            f"Epoch {epoch + 1:4d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# CELL 12 — EVALUATE TEST SET
# ============================================================

test_probabilities = model.forward(
    X_test
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

test_accuracy = np.mean(
    test_predictions == y_test
)

test_Y = one_hot_encode(
    y_test,
    n_classes
)

test_loss = cross_entropy_loss(
    test_Y,
    test_probabilities
)

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)


# ============================================================
# CELL 13 — VISUALIZE LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title(
    "Training Loss During Backpropagation"
)

plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 14 — VISUALIZE ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    accuracy_history
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Training Accuracy"
)

plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 15 — GRADIENT CHECKING
# ============================================================
#
# This is extremely important.
#
# We compare our analytical gradient:
#
#     Backpropagation
#
# against a numerical approximation:
#
#     Finite Difference
#
# Numerical derivative:
#
#     f(x + epsilon) - f(x - epsilon)
#     --------------------------------
#               2 * epsilon
#
# If they are close, our backward implementation is probably
# correct.
#
# ============================================================

def compute_loss(model, X, Y):
    """
    Compute model loss.
    """

    predictions = model.forward(
        X
    )

    return cross_entropy_loss(
        Y,
        predictions
    )


def numerical_gradient(
    model,
    parameter,
    X,
    Y,
    epsilon=1e-5
):
    """
    Calculate numerical gradient
    using finite differences.
    """

    gradient = np.zeros_like(
        parameter
    )

    iterator = np.nditer(
        parameter,
        flags=["multi_index"],
        op_flags=["readwrite"]
    )

    while not iterator.finished:

        index = iterator.multi_index

        original_value = parameter[index]

        # f(x + epsilon)
        parameter[index] = (
            original_value
            + epsilon
        )

        loss_plus = compute_loss(
            model,
            X,
            Y
        )

        # f(x - epsilon)
        parameter[index] = (
            original_value
            - epsilon
        )

        loss_minus = compute_loss(
            model,
            X,
            Y
        )

        # Central difference.
        gradient[index] = (
            loss_plus - loss_minus
        ) / (
            2 * epsilon
        )

        # Restore original value.
        parameter[index] = (
            original_value
        )

        iterator.iternext()

    return gradient


# ============================================================
# CELL 16 — GRADIENT CHECK ON SMALL BATCH
# ============================================================

X_small = X_train[:5]
Y_small = Y_train[:5]

# Forward pass.
model.forward(
    X_small
)

# Analytical gradients.
model.backward(
    Y_small
)

analytical_dW1 = model.dW1.copy()

# Numerical gradients.
numerical_dW1 = numerical_gradient(
    model,
    model.W1,
    X_small,
    Y_small
)


# ============================================================
# CELL 17 — COMPARE GRADIENTS
# ============================================================

difference = np.linalg.norm(
    analytical_dW1
    - numerical_dW1
)

relative_difference = (
    difference
    / (
        np.linalg.norm(
            analytical_dW1
        )
        + np.linalg.norm(
            numerical_dW1
        )
        + 1e-12
    )
)

print(
    "Absolute gradient difference:",
    difference
)

print(
    "Relative gradient difference:",
    relative_difference
)


# ============================================================
# CELL 18 — SIMPLE GRADIENT CHECK RESULT
# ============================================================

if relative_difference < 1e-7:

    print(
        "✓ Gradient check passed."
    )

elif relative_difference < 1e-5:

    print(
        "✓ Gradient check is reasonably close."
    )

else:

    print(
        "✗ Gradient check failed."
    )


# ============================================================
# CELL 19 — GRADIENT MAGNITUDE INSPECTION
# ============================================================

gradient_norms = {
    "dW1": np.linalg.norm(
        model.dW1
    ),
    "db1": np.linalg.norm(
        model.db1
    ),
    "dW2": np.linalg.norm(
        model.dW2
    ),
    "db2": np.linalg.norm(
        model.db2
    )
}

for name, value in gradient_norms.items():

    print(
        f"{name}: {value:.6f}"
    )


# ============================================================
# CELL 20 — PYTORCH AUTOGRAD VERIFICATION
# ============================================================
#
# Now we compare our understanding with PyTorch.
#
# NumPy:
#
#     We calculate gradients manually.
#
# PyTorch:
#
#     Autograd calculates them automatically.
#
# ============================================================

try:

    import torch

    print(
        "PyTorch version:",
        torch.__version__
    )

except ImportError:

    print(
        "PyTorch is not installed."
    )


# ============================================================
# CELL 21 — AUTOGRAD CONCEPT
# ============================================================
#
# PyTorch tracks operations:
#
#     X
#     ↓
#     W
#     ↓
#     Forward computation
#     ↓
#     Loss
#
# Then:
#
#     loss.backward()
#
# automatically calculates gradients.
#
# ============================================================

if "torch" in globals():

    torch.manual_seed(42)

    x = torch.tensor(
        2.0,
        requires_grad=True
    )

    y = x ** 2

    y.backward()

    print(
        "x:",
        x.item()
    )

    print(
        "y:",
        y.item()
    )

    print(
        "dy/dx:",
        x.grad.item()
    )


# ============================================================
# CELL 22 — AUTOGRAD WITH A SIMPLE COMPUTATIONAL GRAPH
# ============================================================
#
# Example:
#
#     y = x²
#
# derivative:
#
#     dy/dx = 2x
#
# At x = 2:
#
#     dy/dx = 4
#
# ============================================================

if "torch" in globals():

    x = torch.tensor(
        3.0,
        requires_grad=True
    )

    y = (
        x ** 2
        + 2 * x
        + 1
    )

    y.backward()

    print(
        "x:",
        x.item()
    )

    print(
        "y:",
        y.item()
    )

    print(
        "dy/dx:",
        x.grad.item()
    )


# ============================================================
# CELL 23 — NUMPY VS PYTORCH
# ============================================================

print(
    """
    ========================================================
                  NUMPY vs PYTORCH
    ========================================================

    NUMPY

    Forward:
        manually implemented

    Backward:
        manually implemented

    Gradients:
        manually derived

    Optimizer:
        manually implemented

    GPU:
        not the focus


    PYTORCH

    Forward:
        nn.Module / tensor operations

    Backward:
        automatic differentiation

    Gradients:
        tensor.grad

    Optimizer:
        torch.optim

    GPU:
        CUDA support


    IMPORTANT:

    PyTorch does NOT replace the mathematics.

    It AUTOMATES the mathematics.
    ========================================================
    """
)


# ============================================================
# CELL 24 — COMPLETE DAY 44 PIPELINE
# ============================================================

print(
    """
    ========================================================
                   DAY 44 COMPLETE PIPELINE
    ========================================================

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
                    ▼
             Backpropagation
                    │
                    ▼
                Gradients
                    │
                    ▼
                  SGD
                    │
                    ▼
            Update Parameters
                    │
                    └─────────────┐
                                  │
                                  ▼
                           Forward Again


    MANUAL NUMPY IMPLEMENTATION
              +
    NUMERICAL GRADIENT CHECK
              +
    PYTORCH AUTOGRAD


    ========================================================
    """
)


# ============================================================
# DAY 44 THEORY RECAP
# ============================================================
#
# BACKPROPAGATION
#
# Backpropagation calculates how much the loss changes with
# respect to each trainable parameter.
#
#
# ------------------------------------------------------------
# CHAIN RULE
# ------------------------------------------------------------
#
# If:
#
#     y = f(g(x))
#
# then:
#
#     dy/dx = dy/dg × dg/dx
#
# Neural networks contain many nested functions, so the chain
# rule allows gradients to flow backward through the network.
#
#
# ------------------------------------------------------------
# GRADIENT
# ------------------------------------------------------------
#
# A gradient tells us the direction and magnitude of change
# of the loss with respect to a parameter.
#
#     ∂L/∂W
#
#
# ------------------------------------------------------------
# GRADIENT DESCENT
# ------------------------------------------------------------
#
# Parameter update:
#
#     W_new = W - η * ∂L/∂W
#
# η = learning rate
#
#
# ------------------------------------------------------------
# OUTPUT LAYER
# ------------------------------------------------------------
#
# For Softmax + Cross Entropy:
#
#     dZ2 = A2 - Y
#
# This simplification is extremely important.
#
#
# ------------------------------------------------------------
# SECOND LAYER
# ------------------------------------------------------------
#
#     Z2 = A1W2 + b2
#
# Therefore:
#
#     dW2 = A1.T @ dZ2
#
#     db2 = sum(dZ2)
#
#     dA1 = dZ2 @ W2.T
#
#
# ------------------------------------------------------------
# RELU BACKPROPAGATION
# ------------------------------------------------------------
#
#     ReLU'(x) =
#
#          1 if x > 0
#          0 otherwise
#
# Therefore:
#
#     dZ1 = dA1 * ReLU'(Z1)
#
#
# ------------------------------------------------------------
# FIRST LAYER
# ------------------------------------------------------------
#
#     Z1 = XW1 + b1
#
# Therefore:
#
#     dW1 = X.T @ dZ1
#
#     db1 = sum(dZ1)
#
#
# ------------------------------------------------------------
# WHY CACHE?
# ------------------------------------------------------------
#
# Backpropagation needs intermediate values from the forward
# pass.
#
# Example:
#
#     Z1
#     A1
#     A2
#
# These are stored so backward() can reuse them.
#
#
# ------------------------------------------------------------
# ANALYTICAL VS NUMERICAL GRADIENTS
# ------------------------------------------------------------
#
# Analytical gradient:
#
#     Derived using calculus.
#
# Numerical gradient:
#
#     Approximation using:
#
#     f(x+ε) - f(x-ε)
#     ----------------
#           2ε
#
# Numerical gradient checking is useful for validating custom
# backpropagation implementations.
#
#
# ------------------------------------------------------------
# AUTOGRAD
# ------------------------------------------------------------
#
# PyTorch builds a computational graph while operations happen.
#
# Calling:
#
#     loss.backward()
#
# traverses the graph backward and calculates gradients.
#
#
# ------------------------------------------------------------
# IMPORTANT DISTINCTION
# ------------------------------------------------------------
#
# Backpropagation ≠ Gradient Descent
#
# Backpropagation:
#
#     calculates gradients.
#
# Gradient descent:
#
#     uses gradients to update parameters.
#
#
# ------------------------------------------------------------
# COMPLETE TRAINING LOOP
# ------------------------------------------------------------
#
#     Forward
#       ↓
#     Loss
#       ↓
#     Backward
#       ↓
#     Gradients
#       ↓
#     Optimizer
#       ↓
#     Parameter Update
#       ↓
#     Repeat
#
#
# ------------------------------------------------------------
# CONNECTION TO FUTURE TOPICS
# ------------------------------------------------------------
#
# Day 45:
#     SGD → Momentum → RMSProp → Adam → AdamW
#
# CNN:
#     Backpropagation also computes gradients through
#     convolution operations.
#
# RNN:
#     Backpropagation through time (BPTT).
#
# Transformers:
#     Gradients flow through attention, linear layers,
#     normalization and residual connections.
#
# PyTorch:
#     Autograd performs this differentiation automatically.
#
# ============================================================