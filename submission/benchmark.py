import json
import time

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from lightgbm import LGBMClassifier


# 1. Load data
start = time.perf_counter()

df = pd.read_csv("creditcard.csv")

load_time = time.perf_counter() - start

print(f"Data shape: {df.shape}")
print(f"Load time: {load_time:.4f} seconds")


# 2. Split train / test
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Train LightGBM
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
    verbosity=-1,
)

start = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(30, verbose=False)
    ],
)

training_time = time.perf_counter() - start

print(f"Training time: {training_time:.4f} seconds")
print(f"Best iteration: {model.best_iteration_}")


# 4. Evaluation
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"AUC-ROC: {auc:.6f}")
print(f"Accuracy: {accuracy:.6f}")
print(f"F1-Score: {f1:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")


# 5. Inference latency - 1 row
single_row = X_test.iloc[[0]]

start = time.perf_counter()

model.predict_proba(single_row)

latency = (time.perf_counter() - start) * 1000

print(f"Inference latency (1 row): {latency:.4f} ms")


# 6. Inference throughput - 1000 rows
batch = X_test.iloc[:1000]

start = time.perf_counter()

model.predict_proba(batch)

elapsed = time.perf_counter() - start

throughput = 1000 / elapsed

print(f"Inference throughput (1000 rows): {throughput:.2f} rows/sec")


# 7. Save result
result = {
    "load_time_seconds": load_time,
    "training_time_seconds": training_time,
    "best_iteration": int(model.best_iteration_),
    "auc_roc": auc,
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "inference_latency_1_row_ms": latency,
    "inference_throughput_1000_rows_per_sec": throughput,
}

with open("benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nBenchmark result saved to benchmark_result.json")
