import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
# --- MONKEYPATCH FOR PYTORCH BUG ---
class _DummyTorchUtils:
    @staticmethod
    def _get_device_index(*args, **kwargs):
        return -1
torch._utils = _DummyTorchUtils()
# -----------------------------------
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. SETUP & DATA GENERATION
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

N_SAMPLES = 3000
SEQ_LENGTH = 30
INPUT_SIZE = 1
HIDDEN_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

rng = np.random.default_rng(SEED)
X = rng.normal(size=(N_SAMPLES, SEQ_LENGTH, INPUT_SIZE)).astype(np.float32)
y = (X.sum(axis=(1, 2)) > 0).astype(np.int64)

indices = rng.permutation(N_SAMPLES)
train_end = int(0.70 * N_SAMPLES)
val_end = int(0.85 * N_SAMPLES)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

def make_loader(indices, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X[indices]),
        torch.from_numpy(y[indices]),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx)
test_loader = make_loader(test_idx)

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class SequenceClassifier(nn.Module):
    def __init__(self, cell_type, input_size=1, hidden_size=32, num_classes=2):
        super().__init__()
        recurrent_layers = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        if cell_type not in recurrent_layers:
            raise ValueError(f"Unsupported cell type: {cell_type}")
            
        self.cell_type = cell_type
        self.recurrent = recurrent_layers[cell_type](
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, state = self.recurrent(x)
        if self.cell_type == "LSTM":
            hidden_state, _ = state
        else:
            hidden_state = state
            
        final_hidden = hidden_state[-1]
        return self.classifier(final_hidden)

# ==========================================
# 3. EVALUATION & TRAINING LOGIC
# ==========================================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total

def train_model(cell_type):
    torch.manual_seed(SEED)
    model = SequenceClassifier(
        cell_type,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
            
        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    return model, history

# ==========================================
# 4. EXECUTION
# ==========================================
models = {}
histories = {}

print("\nTraining models...")
for cell_type in ("RNN", "LSTM", "GRU"):
    model, history = train_model(cell_type)
    models[cell_type] = model
    histories[cell_type] = history
    print(f"{cell_type:4s} | Best val loss: {min(history['val_loss']):.4f} | Best val acc: {max(history['val_acc']):.3f}")

# ==========================================
# 5. PLOTTING & FINAL RESULTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for cell_type, history in histories.items():
    axes[0].plot(history["train_loss"], label=cell_type)
    axes[1].plot(history["val_acc"], label=cell_type)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()

axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()

criterion = nn.CrossEntropyLoss()
print("\nFinal test evaluation")
for cell_type, model in models.items():
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"{cell_type:4s} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.3f}")