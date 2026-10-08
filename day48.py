# ============================================================
# DAY 48 — CNN ARCHITECTURE EVOLUTION + INTERACTIVE NN LAB
# Reference experiment:
# TensorFlow Playground — XOR + ReLU + 6,6,6,2
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DAY 48 — CNN ARCHITECTURE + NEURAL NETWORK EXPERIMENT LAB")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# 1. CREATE XOR DATASET
# ============================================================

# Same basic classification problem used in your
# TensorFlow Playground experiment.

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=np.float32)

y = np.array([
    0,
    1,
    1,
    0
], dtype=np.int64)


# Add multiple samples around each XOR point
# so that we have enough data for training.

samples_per_point = 100

X_train = []
y_train = []

for point, label in zip(X, y):

    noise = np.random.normal(
        loc=0.0,
        scale=0.08,
        size=(samples_per_point, 2)
    )

    points = point + noise

    X_train.append(points)
    y_train.extend(
        [label] * samples_per_point
    )

X_train = np.vstack(X_train).astype(np.float32)
y_train = np.array(y_train, dtype=np.int64)


# Convert to PyTorch tensors

X_tensor = torch.tensor(X_train)
y_tensor = torch.tensor(y_train)


dataset = TensorDataset(
    X_tensor,
    y_tensor
)


# ============================================================
# 2. VISUALIZE XOR DATA
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    label="Class 0"
)

plt.scatter(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    label="Class 1"
)

plt.title("XOR Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 3. MODEL FACTORY
# ============================================================

def build_mlp(hidden_layers):
    """
    Create an MLP with a configurable number of hidden layers.

    Example:
        [2]       -> Input → 2 → Output
        [6, 6]    -> Input → 6 → 6 → Output
        [6, 6, 6] -> Input → 6 → 6 → 6 → Output
    """

    layers = []

    input_features = 2

    for hidden_size in hidden_layers:

        layers.append(
            nn.Linear(
                input_features,
                hidden_size
            )
        )

        layers.append(
            nn.ReLU()
        )

        input_features = hidden_size

    layers.append(
        nn.Linear(
            input_features,
            2
        )
    )

    return nn.Sequential(*layers)


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    learning_rate=0.03,
    epochs=300,
    batch_size=28
):
    """
    Train a classification model using SGD.
    """

    model = model.to(DEVICE)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    losses = []
    accuracies = []

    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate
        model.eval()

        with torch.no_grad():

            logits = model(
                X_tensor.to(DEVICE)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            accuracy = (
                predictions
                == y_tensor.to(DEVICE)
            ).float().mean().item()

        losses.append(
            epoch_loss / len(loader)
        )

        accuracies.append(
            accuracy
        )

    return model, losses, accuracies


# ============================================================
# 5. EXPERIMENT A — SHALLOW NETWORK
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT A — SHALLOW NETWORK")
print("=" * 70)

model_shallow = build_mlp(
    [2]
)

model_shallow, loss_shallow, acc_shallow = train_model(
    model_shallow,
    learning_rate=0.03,
    epochs=300,
    batch_size=28
)

print(
    f"Final accuracy: "
    f"{acc_shallow[-1] * 100:.2f}%"
)


# ============================================================
# 6. EXPERIMENT B — DEEPER NETWORK
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT B — DEEPER NETWORK")
print("=" * 70)

model_deep = build_mlp(
    [6, 6, 6]
)

model_deep, loss_deep, acc_deep = train_model(
    model_deep,
    learning_rate=0.03,
    epochs=300,
    batch_size=28
)

print(
    f"Final accuracy: "
    f"{acc_deep[-1] * 100:.2f}%"
)


# ============================================================
# 7. EXPERIMENT C — PLAYGROUND-STYLE 6,6,6,2 NETWORK
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT C — PLAYGROUND-STYLE NETWORK")
print("=" * 70)

print(
    """
Reference Playground configuration:

Activation        : ReLU
Dataset            : XOR
Batch size         : 28
Learning rate      : 0.03
Regularization     : 0
Hidden structure   : 6 → 6 → 6
Output             : 2 classes
"""
)

playground_model = build_mlp(
    [6, 6, 6]
)

playground_model, loss_playground, acc_playground = train_model(
    playground_model,
    learning_rate=0.03,
    epochs=300,
    batch_size=28
)

print(
    f"Final accuracy: "
    f"{acc_playground[-1] * 100:.2f}%"
)


# ============================================================
# 8. COMPARE TRAINING CURVES
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    loss_shallow,
    label="Shallow [2]"
)

plt.plot(
    loss_deep,
    label="Deep [6,6,6]"
)

plt.plot(
    loss_playground,
    label="Playground-style [6,6,6]"
)

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 9. COMPARE ACCURACY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.array(acc_shallow) * 100,
    label="Shallow [2]"
)

plt.plot(
    np.array(acc_deep) * 100,
    label="Deep [6,6,6]"
)

plt.plot(
    np.array(acc_playground) * 100,
    label="Playground-style [6,6,6]"
)

plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 10. DECISION BOUNDARY VISUALIZATION
# ============================================================

def plot_decision_boundary(
    model,
    title,
    ax
):
    """
    Visualize how a neural network separates XOR classes.
    """

    model.eval()

    x_min = X_train[:, 0].min() - 0.3
    x_max = X_train[:, 0].max() + 0.3

    y_min = X_train[:, 1].min() - 0.3
    y_max = X_train[:, 1].max() + 0.3

    grid_x, grid_y = np.meshgrid(
        np.linspace(
            x_min,
            x_max,
            300
        ),
        np.linspace(
            y_min,
            y_max,
            300
        )
    )

    grid = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel()
    ]).astype(np.float32)

    grid_tensor = torch.tensor(
        grid
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(
            grid_tensor
        )

        predictions = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

    predictions = predictions.reshape(
        grid_x.shape
    )

    ax.contourf(
        grid_x,
        grid_y,
        predictions,
        alpha=0.25
    )

    ax.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        label="Class 0"
    )

    ax.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        label="Class 1"
    )

    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

plot_decision_boundary(
    model_shallow,
    "Shallow Network",
    axes[0]
)

plot_decision_boundary(
    model_deep,
    "Deep Network",
    axes[1]
)

plt.tight_layout()
plt.show()


# ============================================================
# 11. RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    """
    Basic ResNet-style residual block.

    H(x) = F(x) + x

    The skip connection directly adds the input
    to the transformed representation.
    """

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # THE RESIDUAL CONNECTION
        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# 12. TEST RESIDUAL BLOCK
# ============================================================

print("\n" + "=" * 70)
print("RESIDUAL CONNECTION EXPERIMENT")
print("=" * 70)

residual_block = ResidualBlock(
    channels=16
).to(DEVICE)

image_batch = torch.randn(
    4,
    16,
    32,
    32
).to(DEVICE)

residual_output = residual_block(
    image_batch
)

print(
    f"Input shape  : {tuple(image_batch.shape)}"
)

print(
    f"Output shape : {tuple(residual_output.shape)}"
)

print(
    "\nResidual connection:"
)

print(
    "H(x) = F(x) + x"
)


# ============================================================
# 13. SIMPLE RESNET-STYLE CNN
# ============================================================

class SmallResNet(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ReLU()
        )

        self.residual_layers = nn.Sequential(

            ResidualBlock(16),

            ResidualBlock(16),

            ResidualBlock(16)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            16,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.residual_layers(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


resnet = SmallResNet(
    num_classes=10
).to(DEVICE)


# ============================================================
# 14. TEST COMPLETE RESNET
# ============================================================

dummy_images = torch.randn(
    8,
    3,
    32,
    32
).to(DEVICE)

with torch.no_grad():

    resnet_logits = resnet(
        dummy_images
    )

print("\n" + "=" * 70)
print("SMALL RESNET FORWARD PASS")
print("=" * 70)

print(
    f"Input  : {tuple(dummy_images.shape)}"
)

print(
    f"Output : {tuple(resnet_logits.shape)}"
)


# ============================================================
# 15. PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


print(
    f"\nTrainable parameters: "
    f"{count_parameters(resnet):,}"
)


# ============================================================
# 16. LEARNING-RATE EXPERIMENT
# ============================================================

learning_rates = [
    0.001,
    0.03,
    0.3
]

lr_results = {}

print("\n" + "=" * 70)
print("LEARNING-RATE EXPERIMENT")
print("=" * 70)

for lr in learning_rates:

    model = build_mlp(
        [6, 6, 6]
    )

    model, losses, accuracies = train_model(
        model,
        learning_rate=lr,
        epochs=200,
        batch_size=28
    )

    lr_results[lr] = {
        "loss": losses,
        "accuracy": accuracies
    }

    print(
        f"LR = {lr:<6} | "
        f"Final accuracy = "
        f"{accuracies[-1] * 100:.2f}%"
    )


# ============================================================
# 17. LEARNING-RATE CURVES
# ============================================================

plt.figure(figsize=(10, 5))

for lr, result in lr_results.items():

    plt.plot(
        result["loss"],
        label=f"LR = {lr}"
    )

plt.title("Learning Rate Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 48 — EXPERIMENT SUMMARY")
print("=" * 70)

print(
    """
1. SHALLOW NETWORK
   └── Limited representation capacity

2. DEEP NETWORK
   └── Learns more complex representations

3. PLAYGROUND-STYLE NETWORK
   └── ReLU + XOR + batch=28 + LR=0.03
   └── 6 → 6 → 6 hidden architecture

4. RESIDUAL NETWORK
   └── H(x) = F(x) + x
   └── Skip connections improve gradient flow

5. LEARNING RATE
   └── Too small → slow learning
   └── Reasonable → stable convergence
   └── Too large → unstable optimization

6. ARCHITECTURE EVOLUTION

   LeNet
      ↓
   AlexNet
      ↓
   VGG
      ↓
   Inception
      ↓
   ResNet
      ↓
   DenseNet
      ↓
   EfficientNet
"""
)

print("\n🔥 Day 48 experiment lab complete!")