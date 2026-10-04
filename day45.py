# ============================================================
# DAY 45 — OPTIMIZATION LAB
# ============================================================
#
# Topics:
#
#   1. SGD
#   2. Momentum
#   3. RMSProp
#   4. Adam
#   5. AdamW
#   6. Learning-rate decay
#   7. Gradient clipping
#   8. Optimizer comparison
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# CELL 1 — DATASET
# ============================================================

n_samples = 600
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
# CELL 2 — ONE-HOT ENCODING
# ============================================================

def one_hot_encode(labels, num_classes):
    encoded = np.zeros(
        (len(labels), num_classes)
    )

    encoded[
        np.arange(len(labels)),
        labels
    ] = 1

    return encoded


Y = one_hot_encode(
    y,
    n_classes
)


# ============================================================
# CELL 3 — ACTIVATIONS
# ============================================================

def relu(x):
    return np.maximum(
        0,
        x
    )


def relu_derivative(x):
    return (
        x > 0
    ).astype(float)


def softmax(logits):
    shifted = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        shifted
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
# CELL 4 — LOSS
# ============================================================

def cross_entropy_loss(
    y_true,
    y_pred
):
    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    return np.mean(
        -np.sum(
            y_true * np.log(y_pred),
            axis=1
        )
    )


# ============================================================
# CELL 5 — MLP
# ============================================================

class MLP:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        self.W1 = (
            np.random.randn(
                input_size,
                hidden_size
            )
            * np.sqrt(
                2 / input_size
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
                2 / hidden_size
            )
        )

        self.b2 = np.zeros(
            (1, output_size)
        )

        self.cache = {}

        self.grads = {}


    def forward(self, X):

        Z1 = (
            X @ self.W1
            + self.b1
        )

        A1 = relu(
            Z1
        )

        Z2 = (
            A1 @ self.W2
            + self.b2
        )

        A2 = softmax(
            Z2
        )

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }

        return A2


    def backward(self, Y):

        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        batch_size = X.shape[0]

        dZ2 = (
            A2 - Y
        ) / batch_size

        dW2 = (
            A1.T @ dZ2
        )

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        dA1 = (
            dZ2 @ self.W2.T
        )

        dZ1 = (
            dA1
            * relu_derivative(Z1)
        )

        dW1 = (
            X.T @ dZ1
        )

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        self.grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }


    def parameters(self):

        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }


# ============================================================
# CELL 6 — GRADIENT CLIPPING
# ============================================================

def clip_gradients(
    gradients,
    max_norm
):
    """
    Clip all gradients using global norm.
    """

    total_norm = 0.0

    for gradient in gradients.values():

        total_norm += np.sum(
            gradient ** 2
        )

    total_norm = np.sqrt(
        total_norm
    )

    scale = min(
        1.0,
        max_norm / (
            total_norm + 1e-8
        )
    )

    clipped = {}

    for name, gradient in gradients.items():

        clipped[name] = (
            gradient * scale
        )

    return clipped


# ============================================================
# CELL 7 — SGD OPTIMIZER
# ============================================================

class SGD:

    def __init__(
        self,
        learning_rate=0.01
    ):

        self.learning_rate = (
            learning_rate
        )


    def step(
        self,
        parameters,
        gradients
    ):

        for name in parameters:

            parameters[name] -= (
                self.learning_rate
                * gradients[name]
            )


# ============================================================
# CELL 8 — MOMENTUM OPTIMIZER
# ============================================================

class Momentum:

    def __init__(
        self,
        learning_rate=0.01,
        beta=0.9
    ):

        self.learning_rate = (
            learning_rate
        )

        self.beta = beta

        self.velocity = {}


    def step(
        self,
        parameters,
        gradients
    ):

        for name in parameters:

            if name not in self.velocity:

                self.velocity[name] = (
                    np.zeros_like(
                        parameters[name]
                    )
                )

            self.velocity[name] = (
                self.beta
                * self.velocity[name]
                + (1 - self.beta)
                * gradients[name]
            )

            parameters[name] -= (
                self.learning_rate
                * self.velocity[name]
            )


# ============================================================
# CELL 9 — RMSPROP
# ============================================================

class RMSProp:

    def __init__(
        self,
        learning_rate=0.001,
        beta=0.9,
        epsilon=1e-8
    ):

        self.learning_rate = (
            learning_rate
        )

        self.beta = beta
        self.epsilon = epsilon

        self.squared_average = {}


    def step(
        self,
        parameters,
        gradients
    ):

        for name in parameters:

            if name not in self.squared_average:

                self.squared_average[name] = (
                    np.zeros_like(
                        parameters[name]
                    )
                )

            self.squared_average[name] = (
                self.beta
                * self.squared_average[name]
                + (1 - self.beta)
                * gradients[name] ** 2
            )

            parameters[name] -= (
                self.learning_rate
                * gradients[name]
                / (
                    np.sqrt(
                        self.squared_average[name]
                    )
                    + self.epsilon
                )
            )


# ============================================================
# CELL 10 — ADAM
# ============================================================

class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = (
            learning_rate
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        self.step_count = 0


    def step(
        self,
        parameters,
        gradients
    ):

        self.step_count += 1

        for name in parameters:

            if name not in self.m:

                self.m[name] = (
                    np.zeros_like(
                        parameters[name]
                    )
                )

                self.v[name] = (
                    np.zeros_like(
                        parameters[name]
                    )
                )

            self.m[name] = (
                self.beta1
                * self.m[name]
                + (1 - self.beta1)
                * gradients[name]
            )

            self.v[name] = (
                self.beta2
                * self.v[name]
                + (1 - self.beta2)
                * gradients[name] ** 2
            )

            # Bias correction.
            m_hat = (
                self.m[name]
                / (
                    1
                    - self.beta1
                    ** self.step_count
                )
            )

            v_hat = (
                self.v[name]
                / (
                    1
                    - self.beta2
                    ** self.step_count
                )
            )

            parameters[name] -= (
                self.learning_rate
                * m_hat
                / (
                    np.sqrt(v_hat)
                    + self.epsilon
                )
            )


# ============================================================
# CELL 11 — ADAMW
# ============================================================

class AdamW(Adam):

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        super().__init__(
            learning_rate,
            beta1,
            beta2,
            epsilon
        )

        self.weight_decay = (
            weight_decay
        )


    def step(
        self,
        parameters,
        gradients
    ):

        super().step(
            parameters,
            gradients
        )

        # Decoupled weight decay.
        for name in parameters:

            parameters[name] *= (
                1
                - self.learning_rate
                * self.weight_decay
            )


# ============================================================
# CELL 12 — LEARNING-RATE SCHEDULE
# ============================================================

def cosine_decay(
    initial_lr,
    epoch,
    total_epochs,
    minimum_lr=1e-5
):
    """
    Cosine learning-rate decay.
    """

    cosine_value = (
        0.5
        * (
            1
            + np.cos(
                np.pi
                * epoch
                / total_epochs
            )
        )
    )

    return (
        minimum_lr
        + (
            initial_lr
            - minimum_lr
        )
        * cosine_value
    )


# ============================================================
# CELL 13 — TRAINING FUNCTION
# ============================================================

def train_model(
    optimizer_class,
    learning_rate,
    epochs=300,
    weight_decay=0.0
):

    np.random.seed(42)

    model = MLP(
        input_size=n_features,
        hidden_size=16,
        output_size=n_classes
    )

    if optimizer_class == AdamW:

        optimizer = optimizer_class(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    else:

        optimizer = optimizer_class(
            learning_rate=learning_rate
        )

    losses = []
    accuracies = []

    for epoch in range(epochs):

        # Forward.
        probabilities = model.forward(
            X
        )

        # Loss.
        loss = cross_entropy_loss(
            Y,
            probabilities
        )

        # Backward.
        model.backward(
            Y
        )

        # Gradient clipping.
        model.grads = clip_gradients(
            model.grads,
            max_norm=5.0
        )

        # Update.
        optimizer.step(
            model.parameters(),
            model.grads
        )

        # Metrics.
        predictions = np.argmax(
            probabilities,
            axis=1
        )

        accuracy = np.mean(
            predictions == y
        )

        losses.append(
            loss
        )

        accuracies.append(
            accuracy
        )

    return (
        model,
        losses,
        accuracies
    )


# ============================================================
# CELL 14 — TRAIN WITH SGD
# ============================================================

sgd_model, sgd_losses, sgd_accuracies = (
    train_model(
        SGD,
        learning_rate=0.1
    )
)

print(
    "SGD final loss:",
    sgd_losses[-1]
)

print(
    "SGD final accuracy:",
    sgd_accuracies[-1]
)


# ============================================================
# CELL 15 — TRAIN WITH MOMENTUM
# ============================================================

momentum_model, momentum_losses, momentum_accuracies = (
    train_model(
        Momentum,
        learning_rate=0.1
    )
)

print(
    "Momentum final loss:",
    momentum_losses[-1]
)

print(
    "Momentum final accuracy:",
    momentum_accuracies[-1]
)


# ============================================================
# CELL 16 — TRAIN WITH RMSPROP
# ============================================================

rms_model, rms_losses, rms_accuracies = (
    train_model(
        RMSProp,
        learning_rate=0.01
    )
)

print(
    "RMSProp final loss:",
    rms_losses[-1]
)

print(
    "RMSProp final accuracy:",
    rms_accuracies[-1]
)


# ============================================================
# CELL 17 — TRAIN WITH ADAM
# ============================================================

adam_model, adam_losses, adam_accuracies = (
    train_model(
        Adam,
        learning_rate=0.01
    )
)

print(
    "Adam final loss:",
    adam_losses[-1]
)

print(
    "Adam final accuracy:",
    adam_accuracies[-1]
)


# ============================================================
# CELL 18 — TRAIN WITH ADAMW
# ============================================================

adamw_model, adamw_losses, adamw_accuracies = (
    train_model(
        AdamW,
        learning_rate=0.01,
        weight_decay=0.01
    )
)

print(
    "AdamW final loss:",
    adamw_losses[-1]
)

print(
    "AdamW final accuracy:",
    adamw_accuracies[-1]
)


# ============================================================
# CELL 19 — COMPARE LOSS CURVES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    sgd_losses,
    label="SGD"
)

plt.plot(
    momentum_losses,
    label="Momentum"
)

plt.plot(
    rms_losses,
    label="RMSProp"
)

plt.plot(
    adam_losses,
    label="Adam"
)

plt.plot(
    adamw_losses,
    label="AdamW"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Optimizer Comparison — Loss"
)

plt.legend()
plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 20 — COMPARE ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    sgd_accuracies,
    label="SGD"
)

plt.plot(
    momentum_accuracies,
    label="Momentum"
)

plt.plot(
    rms_accuracies,
    label="RMSProp"
)

plt.plot(
    adam_accuracies,
    label="Adam"
)

plt.plot(
    adamw_accuracies,
    label="AdamW"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Optimizer Comparison — Accuracy"
)

plt.legend()
plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 21 — FINAL COMPARISON
# ============================================================

results = {
    "SGD": (
        sgd_losses[-1],
        sgd_accuracies[-1]
    ),

    "Momentum": (
        momentum_losses[-1],
        momentum_accuracies[-1]
    ),

    "RMSProp": (
        rms_losses[-1],
        rms_accuracies[-1]
    ),

    "Adam": (
        adam_losses[-1],
        adam_accuracies[-1]
    ),

    "AdamW": (
        adamw_losses[-1],
        adamw_accuracies[-1]
    )
}

print(
    "\nOptimizer Results\n"
)

print(
    f"{'Optimizer':<12}"
    f"{'Loss':<15}"
    f"{'Accuracy':<15}"
)

print("-" * 42)

for name, (
    loss,
    accuracy
) in results.items():

    print(
        f"{name:<12}"
        f"{loss:<15.6f}"
        f"{accuracy:<15.4f}"
    )


# ============================================================
# CELL 22 — LEARNING RATE EXPERIMENT
# ============================================================

learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0
]

lr_results = {}

for lr in learning_rates:

    model_lr, losses_lr, _ = (
        train_model(
            Adam,
            learning_rate=lr,
            epochs=200
        )
    )

    lr_results[lr] = losses_lr


# ============================================================
# CELL 23 — LEARNING RATE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for lr, losses in lr_results.items():

    plt.plot(
        losses,
        label=f"LR={lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Effect of Learning Rate"
)

plt.legend()
plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 24 — COSINE DECAY EXAMPLE
# ============================================================

epochs = 100

learning_rates_schedule = [
    cosine_decay(
        initial_lr=0.01,
        epoch=epoch,
        total_epochs=epochs
    )
    for epoch in range(epochs)
]


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    learning_rates_schedule
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(
    "Cosine Learning Rate Decay"
)

plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 25 — OPTIMIZATION SUMMARY
# ============================================================

print(
    """
    ========================================================
                     DAY 45 COMPLETE
    ========================================================

    Implemented:

    ✓ SGD
    ✓ Momentum
    ✓ RMSProp
    ✓ Adam
    ✓ AdamW
    ✓ Gradient clipping
    ✓ Cosine learning-rate decay
    ✓ Learning-rate experiments
    ✓ Optimizer comparison

    CORE IDEA:

        Gradient
           ↓
        Optimizer
           ↓
       Parameter
        Update

    ========================================================
    """
)