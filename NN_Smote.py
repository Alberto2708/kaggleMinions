# nn_wandb_csv_pytorch.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import wandb
from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------
# 1. Initialize W&B
# -----------------------
wandb.init(project="csv-tf-example", config={
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": 64
})
config = wandb.config

# -----------------------
# 2. Load Dataset
# -----------------------
data = pd.read_csv("Datasets/Base.csv")
target_col = "fraud_bool"


y = data[target_col].values.astype(np.float32)
X = data.drop(target_col, axis=1)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#X = X.drop(categorical_columns, axis=1)  # Drop categorical for simplicity

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(X_scaled, y)


# -----------------------
# 4. Convert to PyTorch tensors
# -----------------------
X_tensor = torch.tensor(x_res, dtype=torch.float32)
y_tensor = torch.tensor(y_res, dtype=torch.float32).unsqueeze(1)  # shape (N,1) for BCE

# -----------------------
# 5. Train/Test split
# -----------------------
dataset = TensorDataset(X_tensor, y_tensor)
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

# -----------------------
# 6. Build Neural Network
# -----------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = Net(X_tensor.shape[1], config.hidden_units)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# -----------------------
# 7. Training loop
# -----------------------
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total

    # -----------------------
    # Validation loop
    # -----------------------
    model.eval()
    val_correct = 0
    val_total = 0
    val_recall = 0
    with torch.no_grad():
        for val_X, val_y in test_loader:
            val_outputs = model(val_X)
            predicted = (val_outputs >= 0.5).float()
            val_correct += (predicted == val_y).sum().item()
            
            # Recall calculation
            true_positive = ((predicted == 1) & (val_y == 1)).sum().item()
            false_negative = ((predicted == 0) & (val_y == 1)).sum().item()
            val_recall += true_positive / max(true_positive + false_negative, 1)
            val_total += 1

    val_acc = val_correct / len(test_dataset)
    val_recall = val_recall / val_total

    print(f"Epoch [{epoch+1}/{config.epochs}] "
          f"Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f} "
          f"Val Recall: {val_recall:.4f}")
    
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_recall": val_recall
    })

# -----------------------
# 8. Final Evaluation
# -----------------------
model.eval()
all_correct = 0
all_recall = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = (outputs >= 0.5).float()
        all_correct += (preds == y_batch).sum().item()
        true_positive = ((preds == 1) & (y_batch == 1)).sum().item()
        false_negative = ((preds == 0) & (y_batch == 1)).sum().item()
        all_recall += true_positive / max(true_positive + false_negative, 1)

test_acc = all_correct / len(test_dataset)
test_recall = all_recall / len(test_loader)

print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Recall: {test_recall:.4f}")
wandb.log({"test_accuracy": test_acc, "test_recall": test_recall})
