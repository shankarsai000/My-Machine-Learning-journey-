import numpy as np
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES = 2_000
SEQ_LENGTH = 10
INPUT_SIZE = 1

X = rng.normal(
    size=(N_SAMPLES, SEQ_LENGTH, INPUT_SIZE)
).astype(np.float32)

y = (X.sum(axis=(1, 2)) > 0).astype(np.int64)

# Shuffle and split into train, validation, and test sets.
indices = rng.permutation(N_SAMPLES)
train_end = int(0.70 * N_SAMPLES)
val_end = int(0.85 * N_SAMPLES)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
print("Class balance:", np.bincount(y))
class SimpleRNN:
    """A vanilla tanh RNN for sequence classification."""

    def __init__(self, input_size, hidden_size, num_classes, seed=42):
        rng = np.random.default_rng(seed)

        self.hidden_size = hidden_size

        # Xavier-style initialization for input and output weights.
        self.Wx = rng.normal(
            0, np.sqrt(1 / input_size),
            size=(input_size, hidden_size)
        )
        self.Wh = rng.normal(
            0, np.sqrt(1 / hidden_size),
            size=(hidden_size, hidden_size)
        )
        self.Wy = rng.normal(
            0, np.sqrt(1 / hidden_size),
            size=(hidden_size, num_classes)
        )

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, num_classes))

    def forward(self, X):
        """X shape: (batch, sequence_length, input_size)."""
        batch_size, seq_length, _ = X.shape

        h = np.zeros((batch_size, self.hidden_size))
        hidden_states = [h]

        for t in range(seq_length):
            x_t = X[:, t, :]

            h = np.tanh(
                x_t @ self.Wx
                + h @ self.Wh
                + self.bh
            )
            hidden_states.append(h)

        logits = h @ self.Wy + self.by

        # Numerically stable softmax.
        shifted_logits = logits - logits.max(
            axis=1, keepdims=True
        )
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / exp_logits.sum(
            axis=1, keepdims=True
        )

        cache = (X, hidden_states, probs)
        return probs, cache

    def loss_and_backward(self, X, y):
        """Compute cross-entropy loss and BPTT gradients."""
        probs, cache = self.forward(X)
        X, hidden_states, probs = cache

        batch_size, seq_length, _ = X.shape
        rows = np.arange(batch_size)

        loss = -np.log(
            np.clip(probs[rows, y], 1e-12, 1.0)
        ).mean()

        # Softmax + cross-entropy gradient.
        dlogits = probs.copy()
        dlogits[rows, y] -= 1
        dlogits /= batch_size

        grads = {
            "Wx": np.zeros_like(self.Wx),
            "Wh": np.zeros_like(self.Wh),
            "Wy": hidden_states[-1].T @ dlogits,
            "bh": np.zeros_like(self.bh),
            "by": dlogits.sum(axis=0, keepdims=True),
        }

        dh = dlogits @ self.Wy.T

        # BPTT: walk backward through time.
        for t in reversed(range(seq_length)):
            h_t = hidden_states[t + 1]
            h_prev = hidden_states[t]
            x_t = X[:, t, :]

            # Derivative of tanh.
            da = dh * (1 - h_t**2)

            grads["Wx"] += x_t.T @ da
            grads["Wh"] += h_prev.T @ da
            grads["bh"] += da.sum(axis=0, keepdims=True)

            # Send gradient to the previous hidden state.
            dh = da @ self.Wh.T

        return loss, grads

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def parameters(self):
        return {
            "Wx": self.Wx,
            "Wh": self.Wh,
            "Wy": self.Wy,
            "bh": self.bh,
            "by": self.by,
        }


model = SimpleRNN(
    input_size=INPUT_SIZE,
    hidden_size=16,
    num_classes=2,
    seed=SEED,
)

EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.03

history = {"loss": [], "train_acc": [], "val_acc": []}
rng_train = np.random.default_rng(SEED)

for epoch in range(EPOCHS):
    epoch_losses = []
    order = rng_train.permutation(len(X_train))

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        xb = X_train[batch_idx]
        yb = y_train[batch_idx]

        loss, grads = model.loss_and_backward(xb, yb)

        # Global gradient clipping: control exploding gradients.
        grad_norm = np.sqrt(sum(
            np.sum(g**2) for g in grads.values()
        ))
        clip_threshold = 1.0

        if grad_norm > clip_threshold:
            scale = clip_threshold / (grad_norm + 1e-12)
            grads = {k: v * scale for k, v in grads.items()}

        # SGD parameter update.
        for name, param in model.parameters().items():
            param -= LEARNING_RATE * grads[name]

        epoch_losses.append(loss)

    train_acc = np.mean(
        model.predict(X_train) == y_train
    )
    val_acc = np.mean(
        model.predict(X_val) == y_val
    )

    history["loss"].append(np.mean(epoch_losses))
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {history['loss'][-1]:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f}"
        )

test_acc = np.mean(model.predict(X_test) == y_test)
print(f"\nFinal test accuracy: {test_acc:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["loss"])
axes[0].set_title("RNN Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"], label="Validation")
axes[1].set_title("Classification Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def make_loader(X_np, y_np, batch_size=64, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)


class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=2):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        _, h_n = self.rnn(x)

        # Final hidden state: (num_layers, batch, hidden_size)
        final_hidden = h_n[-1]

        # Return logits; CrossEntropyLoss applies log-softmax internally.
        return self.classifier(final_hidden)


model_torch = RNNClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model_torch.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

    return total_loss / total, correct / total


EPOCHS_TORCH = 20

for epoch in range(EPOCHS_TORCH):
    model_torch.train()
    total_loss = 0.0
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model_torch(xb)
        loss = criterion(logits, yb)

        loss.backward()

        # Clip gradients to limit excessively large updates.
        nn.utils.clip_grad_norm_(
            model_torch.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(yb)
        total += len(yb)

    val_loss, val_acc = evaluate(model_torch, val_loader)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train loss: {total_loss / total:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.3f}"
        )

test_loss, test_acc = evaluate(model_torch, test_loader)
print(f"\nPyTorch test accuracy: {test_acc:.3f}")