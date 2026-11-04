import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CONFIG = {
    "seed": 42,
    "test_size": 0.20,
    "validation_size": 0.20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 8,
    "hidden_dim": 32,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ARTIFACT_DIR = Path("artifacts/day56")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Artifact directory:", ARTIFACT_DIR)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])

dataset = load_breast_cancer()

X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["seed"],
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=CONFIG["validation_size"],
    random_state=CONFIG["seed"],
    stratify=y_train_val,
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

def create_loader(X, y, batch_size, shuffle):
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


train_loader = create_loader(
    X_train, y_train, CONFIG["batch_size"], True
)

val_loader = create_loader(
    X_val, y_val, CONFIG["batch_size"], False
)

test_loader = create_loader(
    X_test, y_test, CONFIG["batch_size"], False
)

class ClassificationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


model = ClassificationMLP(
    input_dim=X_train.shape[1],
    hidden_dim=CONFIG["hidden_dim"],
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

print(model)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss detected.")

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        if not torch.isfinite(gradient_norm):
            raise ValueError("Non-finite gradient norm detected.")

        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite evaluation loss detected.")

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy().ravel()
            )
            all_labels.extend(labels.cpu().numpy().ravel())

    average_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


def save_checkpoint(path, epoch, model, optimizer, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": CONFIG,
        },
        path,
    )


best_val_loss = float("inf")
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
}

for epoch in range(CONFIG["epochs"]):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer
    )

    val_loss, val_accuracy = evaluate(
        model, val_loader, criterion
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    save_checkpoint(
        ARTIFACT_DIR / "latest_checkpoint.pt",
        epoch,
        model,
        optimizer,
        best_val_loss,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        save_checkpoint(
            ARTIFACT_DIR / "best_model.pt",
            epoch,
            model,
            optimizer,
            best_val_loss,
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1:02d}/{CONFIG['epochs']} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )

    if epochs_without_improvement >= CONFIG["patience"]:
        print("Early stopping triggered.")
        break

plt.figure(figsize=(9, 5))

plt.plot(history["train_loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

checkpoint = torch.load(
    ARTIFACT_DIR / "best_model.pt",
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss, _ = evaluate(model, test_loader, criterion)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        logits = model(features.to(DEVICE))
        predictions = (torch.sigmoid(logits) >= 0.5).int()

        all_predictions.extend(
            predictions.cpu().numpy().ravel()
        )
        all_labels.extend(labels.numpy().ravel())

print("Test loss:", round(test_loss, 4))
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.target_names,
    )
)

cm = confusion_matrix(all_labels, all_predictions)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names,
).plot()

plt.title("Test Confusion Matrix")
plt.show()

joblib.dump(
    scaler,
    ARTIFACT_DIR / "scaler.joblib",
)

with open(
    ARTIFACT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(CONFIG, file, indent=4)

with open(
    ARTIFACT_DIR / "history.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(history, file, indent=4)

print("Experiment artifacts saved.")

