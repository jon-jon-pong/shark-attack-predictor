# -*- coding: utf-8 -*-
"""
train_fatality_baseline.py
A tiny, commented PyTorch example that uses Year, Type, and Sex to predict fatality.

How to run:
    python train_fatality_baseline.py --csv attacks_encoded_ml.csv

If you're using WSL or Linux/macOS, same command applies.
"""

import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# Small helper Dataset wrapper
# -----------------------------
class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Convert NumPy arrays to tensors for PyTorch
        self.X = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Returns one (features, label) pair
        return self.X[idx], self.y[idx]

# -----------------------------
# A deeper MLP classifier (better than logistic regression)
# -----------------------------
class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for binary classification.
    More powerful than logistic regression for capturing complex patterns.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Output shape: (batch,)
        return self.network(x).squeeze(1)

def main(args):
    # 1) Load your data
    df = pd.read_csv(args.csv, encoding="latin1")

    # Normalize column names for easier matching: lowercase + underscores
    df.columns = [c.strip().replace(" ", "_").replace(".", "_").lower() for c in df.columns]

    # 2) Find the fatality target column robustly
    # It might be "fatal_(y/n)" or "fatal_yn" etc. We just look for "fatal" in the name.
    target_col = None
    for c in df.columns:
        if "fatal" in c:
            target_col = c
            break
    if target_col is None:
        raise RuntimeError("Could not find a fatality target column. Look for a column containing 'fatal'.")

    # Map target values to {0,1}. We try text labels first, then fallback to numeric.
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y = np.where(y_raw.isin(["1", "true", "y", "yes", "fatal"]), 1,
         np.where(y_raw.isin(["0", "false", "n", "no", "non-fatal", "not_fatal", "not fatal"]), 0, np.nan)
    ).astype(float)
    if np.isnan(y).all():
        # Maybe it's already numeric 0/1
        y = pd.to_numeric(df[target_col], errors="coerce").astype(float).to_numpy()

    # 3) Pick all available features for better predictions
    def pick(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    year_col = pick(["year"])
    type_col = pick(["type"])
    country_col = pick(["country"])
    area_col = pick(["area"])
    activity_col = pick(["activity"])
    sex_col  = pick(["sex_", "sex"])
    injury_col = pick(["injury"])

    # We'll use whatever features are available
    feature_map = {
        "year": year_col,
        "type": type_col,
        "country": country_col,
        "area": area_col,
        "activity": activity_col,
        "sex": sex_col,
        "injury": injury_col
    }
    
    available_features = {k: v for k, v in feature_map.items() if v is not None}
    print(f"Using features: {list(available_features.keys())}")
    
    if not available_features:
        raise RuntimeError("No features found in the dataset")

    # Build data dictionary with all available features
    data_dict = {"y": y}
    numeric_cols = []
    categorical_cols = []
    
    # Numeric features
    if "year" in available_features:
        data_dict["year"] = pd.to_numeric(df[year_col], errors="coerce")
        numeric_cols.append("year")
    
    if "injury" in available_features:
        data_dict["injury"] = pd.to_numeric(df[injury_col], errors="coerce")
        numeric_cols.append("injury")
    
    # Categorical features
    if "type" in available_features:
        data_dict["type"] = df[type_col].astype(str).str.strip()
        categorical_cols.append("type")
    
    if "country" in available_features:
        data_dict["country"] = df[country_col].astype(str).str.strip()
        categorical_cols.append("country")
    
    if "area" in available_features:
        data_dict["area"] = df[area_col].astype(str).str.strip()
        categorical_cols.append("area")
    
    if "activity" in available_features:
        data_dict["activity"] = df[activity_col].astype(str).str.strip()
        categorical_cols.append("activity")
    
    if "sex" in available_features:
        data_dict["sex"] = df[sex_col].astype(str).str.strip().str.upper()
        categorical_cols.append("sex")

    # Combine into a working DataFrame and drop rows with missing target
    data = pd.DataFrame(data_dict).dropna(subset=["y"]).reset_index(drop=True)

    # 4) Encode features
    # - Numeric: standardize (zero mean, unit variance) for stability
    # - Categorical: one-hot encode (turn categories into binary columns)
    
    if numeric_cols:
        X_num = data[numeric_cols].astype(float)
        X_num = (X_num - X_num.mean()) / (X_num.std() + 1e-9)
    else:
        X_num = pd.DataFrame()
    
    if categorical_cols:
        X_cat = pd.get_dummies(data[categorical_cols], dummy_na=True)
    else:
        X_cat = pd.DataFrame()
    
    # Combine numeric and categorical features
    if not X_num.empty and not X_cat.empty:
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1).astype(np.float32)
    elif not X_num.empty:
        X = X_num.astype(np.float32)
    else:
        X = X_cat.astype(np.float32)
    
    print(f"Total features after encoding: {X.shape[1]}")

    y = data["y"].astype(int).to_numpy()

    # 5) Train/validation split
    # Use stratify=y to maintain the same class balance in both splits.
    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(), y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Wrap in PyTorch Datasets and DataLoaders
    train_ds = TabularDataset(X_train, y_train)
    val_ds   = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 7) Create model, loss, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(in_features=X.shape[1]).to(device)

    # Helpful bias init: set final layer bias to the base-rate log-odds
    p = y_train.mean()  # prevalence of positive class in training data
    with torch.no_grad():
        # For MLP, initialize the final layer's bias
        final_layer = list(model.network.children())[-1]
        final_layer.bias.fill_(float(math.log(p / (1 - p))))

    # Handle class imbalance via pos_weight in BCEWithLogitsLoss:
    # pos_weight = (#negatives / #positives)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # increased lr from 3e-4

    # Print class distribution for debugging
    print(f"\nClass distribution:")
    print(f"  Training: {(y_train==0).sum()} class 0, {(y_train==1).sum()} class 1 ({y_train.mean()*100:.1f}% positive)")
    print(f"  Validation: {(y_val==0).sum()} class 0, {(y_val==1).sum()} class 1 ({y_val.mean()*100:.1f}% positive)")
    print(f"  pos_weight: {pos_weight.item():.3f}\n")

    # 8) Training/evaluation loops
    def evaluate():
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()  # threshold at 0.5
                correct += (preds == yb.long()).sum().item()
                total += yb.size(0)
                loss_sum += loss.item() * yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Print prediction statistics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        pred_dist = f"preds: {(all_preds==0).sum()} class 0, {(all_preds==1).sum()} class 1"
        prob_stats = f"prob range: [{all_probs.min():.3f}, {all_probs.max():.3f}], mean: {all_probs.mean():.3f}"
        
        return loss_sum / total, correct / total, pred_dist, prob_stats

    epochs = 20  # increased from 10 for better convergence
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, pred_dist, prob_stats = evaluate()
        print(f"epoch {epoch:02d}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  |  {pred_dist}  |  {prob_stats}")

    # 9) Save the trained model
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": X.shape[1],
        "feature_columns": list(X.columns),  # helpful when loading the model later
    }, "pytorch_mlp_model.pt")

    print("Saved model to pytorch_mlp_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help=r"C:\Users\roark\Downloads\attacks_encoded_ml.csv")
    args = parser.parse_args()
    main(args)
