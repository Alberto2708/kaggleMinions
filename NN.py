# nn_wandb_csv.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback


wandb.init(project="csv-tf-example", config={
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": 64
})
config = wandb.config


data = pd.read_csv("Datasets/Base.csv")


target_col = "fraud_bool"
y = data[target_col] 


X = data.drop(target_col, axis=1)

# -----------------------
# 3. Encode features
# -----------------------
# One-hot encode categorical columns automatically
X = pd.get_dummies(X, drop_first=True)

# Scale numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# 4. Train/Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

callbacks = [
    WandbMetricsLogger(),  # logs metrics to W&B # optionally save model
]

# -----------------------
# 5. Build Neural Network
# -----------------------
model = keras.Sequential([
    layers.Dense(config.hidden_units, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss="binary_crossentropy",
    metrics=["recall", "accuracy"]
)

# -----------------------
# 6. Train with W&B logging
# -----------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=callbacks
)

# -----------------------
# 7. Evaluate
# -----------------------
loss, recall, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
wandb.log({"test_accuracy": acc, "test_recall": recall})