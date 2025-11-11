import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split      
# 1) Load data
X = np.load(r'C:\Users\Downloads\attacks_outputs\X_onehot.npy').astype(np.float32)
y = np.load(r'C:\Users\Downloads\attacks_outputs\y.npy').astype(np.int64)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

# 2) Train/val split (80/20)
n = len(y)
n_train = int(0.8 * n)
n_val = n - n_train
dataset = TensorDataset(X, y)
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=512)

# 3) MLP classifier
model = nn.Sequential(
    nn.Linear(X.shape[1], 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Handle potential class imbalance with pos_weight
pos_weight = ( (y==0).sum() / (y==1).sum() ).to(torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4) Training loop
def evaluate():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
    return loss_sum/total, correct/total

for epoch in range(15):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device).float()
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    val_loss, val_acc = evaluate()
    print(f"epoch {epoch+1:02d}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

# 5) Inference
model.eval()
