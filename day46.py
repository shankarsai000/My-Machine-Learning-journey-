# ============================================================
# DAY 46 — REGULARIZATION, NORMALIZATION & INITIALIZATION
# ============================================================
#
# Project:
# MNIST Digit Classification with PyTorch
#
# Concepts:
#   - L2 / Weight Decay
#   - Dropout
#   - BatchNorm
#   - He Initialization
#   - AdamW
#   - Learning Rate Scheduler
#   - Validation
#   - Early Stopping
#   - Checkpointing
#
# ============================================================


# ============================================================
# CELL 1 — IMPORTS
# ============================================================

import copy
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CELL 2 — REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CELL 3 — DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CELL 4 — HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 20

HIDDEN_SIZE = 256
DROPOUT_RATE = 0.2

PATIENCE = 3


# ============================================================
# CELL 5 — DATA TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ============================================================
# CELL 6 — LOAD MNIST
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# CELL 7 — TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    0.9 * len(train_dataset)
)

validation_size = (
    len(train_dataset)
    - train_size
)

train_dataset, validation_dataset = (
    torch.utils.data.random_split(
        train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )
)


# ============================================================
# CELL 8 — DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# CELL 9 — INSPECT DATA
# ============================================================

images, labels = next(
    iter(train_loader)
)

print("Images:", images.shape)
print("Labels:", labels.shape)


# ============================================================
# CELL 10 — VISUALIZE SAMPLE
# ============================================================

plt.figure(
    figsize=(10, 4)
)

for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )

    # Undo normalization for visualization.
    image = (
        images[i].squeeze()
        * 0.3081
        + 0.1307
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"Label: {labels[i].item()}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# CELL 11 — MODEL
# ============================================================

class MNISTMLP(nn.Module):

    def __init__(
        self,
        hidden_size=256,
        dropout_rate=0.2
    ):

        super().__init__()

        self.network = nn.Sequential(

            # Flatten:
            # 1 × 28 × 28 → 784
            nn.Flatten(),

            # First fully connected layer.
            nn.Linear(
                28 * 28,
                hidden_size
            ),

            # Batch Normalization.
            nn.BatchNorm1d(
                hidden_size
            ),

            # Non-linearity.
            nn.ReLU(),

            # Regularization.
            nn.Dropout(
                dropout_rate
            ),

            # Second layer.
            nn.Linear(
                hidden_size,
                hidden_size
            ),

            # Another normalization layer.
            nn.BatchNorm1d(
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            # Output layer.
            nn.Linear(
                hidden_size,
                10
            )
        )

        self._initialize_weights()


    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear
            ):

                # He initialization.
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )


    def forward(self, x):

        return self.network(x)


# ============================================================
# CELL 12 — CREATE MODEL
# ============================================================

model = MNISTMLP(
    hidden_size=HIDDEN_SIZE,
    dropout_rate=DROPOUT_RATE
).to(device)

print(model)


# ============================================================
# CELL 13 — COUNT PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    "Trainable parameters:",
    f"{total_parameters:,}"
)


# ============================================================
# CELL 14 — LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# CELL 15 — ADAMW OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# CELL 16 — LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# CELL 17 — TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear old gradients.
        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass.
        outputs = model(
            images
        )

        # Loss.
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation.
        loss.backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update parameters.
        optimizer.step()

        # Metrics.
        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# CELL 18 — VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# CELL 19 — TRAINING LOOP + EARLY STOPPING
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_state = None

epochs_without_improvement = 0

for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
    )

    val_loss, val_accuracy = (
        evaluate(
            model,
            validation_loader,
            criterion,
            device
        )
    )

    scheduler.step()

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Early stopping logic.
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        torch.save(
            best_state,
            "best_mnist_mlp.pt"
        )

    else:

        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# CELL 20 — RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# CELL 21 — FINAL TEST EVALUATION
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CELL 22 — TRAINING CURVES
# ============================================================

epochs_completed = range(
    1,
    len(history["train_loss"]) + 1
)

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    epochs_completed,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_completed,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# CELL 23 — ACCURACY CURVES
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    epochs_completed,
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs_completed,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# CELL 24 — CHECK TRAIN / EVAL DROPOUT BEHAVIOR
# ============================================================

model.train()

train_output_1 = model(
    images[:10].to(device)
)

train_output_2 = model(
    images[:10].to(device)
)

train_difference = torch.mean(
    torch.abs(
        train_output_1
        - train_output_2
    )
).item()


model.eval()

eval_output_1 = model(
    images[:10].to(device)
)

eval_output_2 = model(
    images[:10].to(device)
)

eval_difference = torch.mean(
    torch.abs(
        eval_output_1
        - eval_output_2
    )
).item()

print(
    "Difference in train mode:",
    train_difference
)

print(
    "Difference in eval mode:",
    eval_difference
)


# ============================================================
# CELL 25 — SAMPLE PREDICTIONS
# ============================================================

model.eval()

sample_images, sample_labels = next(
    iter(test_loader)
)

sample_images = sample_images.to(
    device
)

with torch.no_grad():

    sample_outputs = model(
        sample_images
    )

sample_predictions = (
    sample_outputs.argmax(
        dim=1
    )
)


plt.figure(
    figsize=(12, 6)
)

for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )

    image = (
        sample_images[i]
        .cpu()
        .squeeze()
        * 0.3081
        + 0.1307
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"Pred: {sample_predictions[i].item()}\n"
        f"True: {sample_labels[i].item()}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# DAY 46 COMPLETE
# ============================================================

print(
    """
    ========================================================
                     DAY 46 COMPLETE
    ========================================================

    ✓ L2 / Weight Decay
    ✓ Dropout
    ✓ BatchNorm
    ✓ He Initialization
    ✓ AdamW
    ✓ Learning-rate scheduling
    ✓ Gradient clipping
    ✓ Train / validation split
    ✓ Early stopping
    ✓ Checkpointing
    ✓ PyTorch train/eval modes
    ✓ MNIST MLP

    CORE PIPELINE:

        Data
         ↓
      DataLoader
         ↓
        MLP
         ↓
    BatchNorm
         ↓
       ReLU
         ↓
      Dropout
         ↓
      Linear
         ↓
       Loss
         ↓
    Backpropagation
         ↓
       AdamW
         ↓
    LR Scheduler
         ↓
    Validation
         ↓
   Early Stopping

    ========================================================
    """
)