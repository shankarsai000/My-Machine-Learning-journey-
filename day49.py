import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2. Configuration
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path("best_cifar10_cnn.pt")
DATA_DIR = Path("./data")

print("Device:", DEVICE)

# CIFAR-10 channel statistics commonly used for normalization.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Augmentation is applied ONLY to training images.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validation and test data must remain unaugmented.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load the training data twice with different transforms.
train_augmented = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

train_for_validation = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=eval_transform,
)

# Fixed 90/10 split of the original training dataset.
train_size = int(0.9 * len(train_augmented))
val_size = len(train_augmented) - train_size

split_generator = torch.Generator().manual_seed(SEED)
train_indices, val_indices = random_split(
    range(len(train_augmented)),
    [train_size, val_size],
    generator=split_generator,
)

# Use the same indices but different transformations.
train_dataset = torch.utils.data.Subset(
    train_augmented, train_indices.indices
)
val_dataset = torch.utils.data.Subset(
    train_for_validation, val_indices.indices
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

CLASSES = train_augmented.classes

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))
print("Classes:", CLASSES)

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CIFAR10CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(model)
print(f"Trainable parameters: {parameter_count:,}")

# Verify tensor dimensions before training.
sample_images, sample_labels = next(iter(train_loader))
sample_images = sample_images.to(DEVICE)

with torch.no_grad():
    sample_logits = model(sample_images)

print("Input shape:", sample_images.shape)
print("Output logits shape:", sample_logits.shape)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        # Protect against unusually large gradients.
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total



    history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "learning_rate": [],
}

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, criterion, optimizer
    )

    val_loss, val_accuracy = evaluate(
        model, val_loader, criterion
    )

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)
    history["learning_rate"].append(current_lr)

    # Save a snapshot whenever validation loss improves.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "classes": CLASSES,
            },
            CHECKPOINT_PATH,
        )
    else:
        epochs_without_improvement += 1

    scheduler.step()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.2%} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.2%} | "
        f"LR: {current_lr:.6f}"
    )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

# Restore the best validation checkpoint.
if best_state is not None:
    model.load_state_dict(best_state)

print("Best validation loss:", round(best_val_loss, 4))
print("Checkpoint:", CHECKPOINT_PATH)


epochs_run = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))

axes[0].plot(epochs_run, history["train_loss"], label="Train")
axes[0].plot(epochs_run, history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(
    epochs_run, history["train_accuracy"], label="Train"
)
axes[1].plot(
    epochs_run, history["val_accuracy"], label="Validation"
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(epochs_run, history["learning_rate"])
axes[2].set_title("Learning Rate")
axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.show()

# Run test evaluation only after model selection is complete.
test_loss, test_accuracy = evaluate(
    model, test_loader, criterion
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

# Inspect a few predictions.
model.eval()
images, labels = next(iter(test_loader))
images = images.to(DEVICE)

with torch.no_grad():
    predictions = model(images).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    # Undo normalization for visualization.
    image = images[i].cpu().permute(1, 2, 0)
    mean = torch.tensor(MEAN).view(1, 1, 3)
    std = torch.tensor(STD).view(1, 1, 3)
    image = (image * std + mean).clamp(0, 1)

    ax.imshow(image.numpy())
    true_label = CLASSES[labels[i].item()]
    predicted_label = CLASSES[predictions[i].item()]

    ax.set_title(
        f"True: {true_label}\nPred: {predicted_label}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()