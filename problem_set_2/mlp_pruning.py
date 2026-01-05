import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import numpy as np

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# 2. Dataset (MNIST)
transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transform),
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=False, transform=transform),
    batch_size=1000,
    shuffle=False
)

# 3. MLP Definition (S1 = 80)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # Flatten image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Training Function
def train(model, epochs=25):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

# 5. Evaluation Function
def test_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# 6. Helper Functions for Weight Pruning
def get_all_weights(model):
    """
    Collect all weights (excluding biases) into a single vector.
    """
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.dim() > 1])

def apply_mask(model, mask):
    """
    Apply a binary mask to the model weights.
    """
    idx = 0
    for p in model.parameters():
        if p.dim() > 1:
            numel = p.numel()
            p.data *= mask[idx:idx+numel].view_as(p)
            idx += numel

# 7A. Smallest-Magnitude Pruning
def prune_smallest(model, pct):
    weights = get_all_weights(model)
    k = int(len(weights) * pct)
    threshold = torch.topk(torch.abs(weights), k, largest=False).values.max()
    mask = (torch.abs(weights) > threshold).float()
    apply_mask(model, mask)

# 7B. Random Pruning
def prune_random(model, pct):
    weights = get_all_weights(model)
    k = int(len(weights) * pct)
    mask = torch.ones(len(weights))
    perm = torch.randperm(len(weights))
    mask[perm[:k]] = 0
    apply_mask(model, mask)

# 7C. Largest-Magnitude Pruning
def prune_largest(model, pct):
    weights = get_all_weights(model)
    k = int(len(weights) * pct)
    threshold = torch.topk(torch.abs(weights), k, largest=True).values.min()
    mask = (torch.abs(weights) < threshold).float()
    apply_mask(model, mask)

# 8. Train Baseline Model
baseline_model = MLP()
train(baseline_model)
baseline_acc = test_accuracy(baseline_model)

print(f"\nBaseline (unpruned) accuracy: {baseline_acc:.2f}%")

# 9. Experiment A: Smallest-Magnitude Pruning
print("\nA) Smallest-magnitude pruning")
for pct in np.arange(0.05, 0.26, 0.02):
    model = copy.deepcopy(baseline_model)
    prune_smallest(model, pct)
    acc = test_accuracy(model)
    print(f"Pruned {pct*100:.1f}% -> Accuracy: {acc:.2f}%")

# 10. Experiment B: Random Pruning
print("\nB) Random pruning")
for pct in np.arange(0.05, 0.26, 0.02):
    model = copy.deepcopy(baseline_model)
    prune_random(model, pct)
    acc = test_accuracy(model)
    print(f"Pruned {pct*100:.1f}% -> Accuracy: {acc:.2f}%")

# 11. Experiment C: Largest-Magnitude Pruning
print("\nC) Largest-magnitude pruning")
for pct in np.arange(0.0125, 0.051, 0.0125):
    model = copy.deepcopy(baseline_model)
    prune_largest(model, pct)
    acc = test_accuracy(model)
    print(f"Pruned {pct*100:.2f}% -> Accuracy: {acc:.2f}%")
