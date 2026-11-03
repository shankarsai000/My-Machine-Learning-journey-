import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = load_breast_cancer()

X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32,
    shuffle=True,
)

class DebugMLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


model = DebugMLP(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)


def debug_training(model, features, labels, logits, loss, optimizer, epoch):
    """
    Debug training by checking inputs, logits, loss, gradients, and parameter updates.
    Reports the first detected failure and includes plots for graphical analysis.
    """
    print(f"\n=== DEBUGGING EPOCH {epoch + 1} ===")

    # Check 1: Input validation
    print("\n[CHECK 1] Input Validation")
    if torch.isnan(features).any():
        print("[FAIL] FAIL: features contains NaN values")
        return False
    if torch.isinf(features).any():
        print("[FAIL] FAIL: features contains Inf values")
        return False
    if torch.isnan(labels).any():
        print("[FAIL] FAIL: labels contains NaN values")
        return False
    if torch.isinf(labels).any():
        print("[FAIL] FAIL: labels contains Inf values")
        return False
    print(f"[PASS] inputs valid (shape: {features.shape}, labels shape: {labels.shape})")

    # Check 2: Logits validation
    print("\n[CHECK 2] Logits Validation")
    if torch.isnan(logits).any():
        print("[FAIL] FAIL: logits contains NaN values")
        return False
    if torch.isinf(logits).any():
        print("[FAIL] FAIL: logits contains Inf values")
        return False
    print(f"[PASS] logits valid (min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f})")

    # Check 3: Loss validation
    print("\n[CHECK 3] Loss Validation")
    if not torch.isfinite(loss):
        print(f"[FAIL] FAIL: loss is {loss.item()}")
        return False
    if loss.item() < 0:
        print(f"[FAIL] FAIL: loss is negative: {loss.item()}")
        return False
    print(f"[PASS] PASS: loss valid (value: {loss.item():.4f})")

    # Check 4: Gradient validation
    print("\n[CHECK 4] Gradient Validation")
    loss.backward()
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if torch.isnan(param.grad).any():
                print(f"[FAIL] FAIL: gradient for {name} contains NaN")
                return False
            if torch.isinf(param.grad).any():
                print(f"[FAIL] FAIL: gradient for {name} contains Inf")
                return False
    if len(grad_norms) > 0:
        print(f"[PASS] PASS: gradients valid (mean norm: {np.mean(grad_norms):.6f}, max norm: {np.max(grad_norms):.6f})")
    else:
        print("[FAIL] FAIL: no gradients computed")
        return False

    # Check 5: Parameter update validation
    print("\n[CHECK 5] Parameter Update Validation")
    param_before = {name: param.clone() for name, param in model.named_parameters()}
    optimizer.step()
    param_after = {name: param.clone() for name, param in model.named_parameters()}

    update_magnitudes = []
    for name in param_before:
        change = (param_after[name] - param_before[name]).abs().sum().item()
        update_magnitudes.append(change)
        if change == 0:
            print(f"[FAIL] FAIL: parameter {name} did not update")
            return False

    print(f"[PASS] PASS: parameters updated (mean change: {np.mean(update_magnitudes):.6f}, max change: {np.max(update_magnitudes):.6f})")

    # Create plots for graphical analysis
    print("\n[GENERATING PLOTS]")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Debug Analysis - Epoch {epoch + 1}', fontsize=16)

    # Plot 1: Feature distribution
    axes[0, 0].hist(features.flatten().cpu().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Feature Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Label distribution
    axes[0, 1].hist(labels.cpu().numpy(), bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Label Distribution')
    axes[0, 1].set_xlabel('Label Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Logits distribution
    axes[0, 2].hist(logits.detach().cpu().numpy(), bins=50, alpha=0.7, color='orange')
    axes[0, 2].set_title('Logits Distribution')
    axes[0, 2].set_xlabel('Logit Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Gradient norms per layer
    layer_names = [name for name, _ in model.named_parameters()]
    axes[1, 0].bar(range(len(grad_norms)), grad_norms, color='red', alpha=0.7)
    axes[1, 0].set_title('Gradient Norms per Parameter')
    axes[1, 0].set_xlabel('Parameter Index')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_xticks(range(len(layer_names)))
    axes[1, 0].set_xticklabels([f'P{i}' for i in range(len(layer_names))], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Parameter update magnitudes
    axes[1, 1].bar(range(len(update_magnitudes)), update_magnitudes, color='purple', alpha=0.7)
    axes[1, 1].set_title('Parameter Update Magnitudes')
    axes[1, 1].set_xlabel('Parameter Index')
    axes[1, 1].set_ylabel('Update Magnitude')
    axes[1, 1].set_xticks(range(len(layer_names)))
    axes[1, 1].set_xticklabels([f'P{i}' for i in range(len(layer_names))], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Loss trajectory (placeholder for now)
    axes[1, 2].plot([loss.item()], 'ro-', markersize=10)
    axes[1, 2].set_title('Current Loss')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Loss Value')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, max(1.0, loss.item() * 1.2))

    plt.tight_layout()
    plt.savefig(f'debug_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
    print(f"[PASS] Plots saved to debug_epoch_{epoch + 1}.png")

    print("\n=== ALL CHECKS PASSED ===\n")
    return True

EPOCHS = 20
DEBUG_MODE = True  # Set to False to disable debugging

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    first_batch = True

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch {epoch + 1}"
            )

        # Run debug checks on first batch if DEBUG_MODE is enabled
        if DEBUG_MODE and first_batch:
            debug_result = debug_training(model, features, labels, logits, loss, optimizer, epoch)
            if not debug_result:
                print(f"Training stopped due to debug failure at epoch {epoch + 1}")
                break
            first_batch = False
            continue  # Skip rest of loop since debug_training already did backward and step

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item() * len(labels)

    average_loss = total_loss / len(train_loader.dataset)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Gradient norm: {grad_norm.item():.4f}"
    )

    