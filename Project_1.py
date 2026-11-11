"""
 Human Activity Detection

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# ============================================================
# Load Dataset
# ============================================================

X_train = pd.read_csv("X_train_features.csv")
X_test = pd.read_csv("X_test_features.csv")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

activity_map = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back"
}

labels_sorted = sorted(activity_map.keys())
label_names = [activity_map[i] for i in labels_sorted]

# ============================================================
# Dataset Overview
# ============================================================

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("\nTraining Feature Shape :", X_train.shape)
print("Testing Feature Shape  :", X_test.shape)
print("Training Labels Shape  :", y_train.shape)
print("Testing Labels Shape   :", y_test.shape)

print("\nNumber of Features :", X_train.shape[1])
print("Number of Activities :", len(activity_map))

print("\nFeature Names")
print("-" * 70)
print(list(X_train.columns))

print("\nFirst Five Rows")
print("-" * 70)
print(X_train.head())

print("\nDataset Information")
print("-" * 70)
X_train.info()

print("\nMissing Values")
print("-" * 70)
print(X_train.isnull().sum())

print("\nStatistical Summary")
print("-" * 70)
print(X_train.describe())

# ============================================================
# Activity Distribution
# ============================================================

print("\nActivity Distribution")
print("-" * 70)

activity_counts = (
    pd.Series(y_train)
    .map(activity_map)
    .value_counts()
    .sort_index()
)

print(activity_counts)

plt.figure(figsize=(12,6))
sns.barplot(
    x=activity_counts.index,
    y=activity_counts.values,
    palette="viridis"
)

plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("activity_distribution.png", dpi=120)
plt.close()

print("\nSaved : activity_distribution.png")

# ============================================================
# Data Preparation
# ============================================================

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

print("""
Feature Scaling:
----------------
Machine learning algorithms such as Logistic Regression,
Support Vector Machine and K-Nearest Neighbors depend on
feature magnitude. Therefore StandardScaler is applied
before training these models.

Tree-based algorithms (Decision Tree, Random Forest and
Gradient Boosting) do not require feature scaling, so
the original features are used for them.
""")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling Completed Successfully.")

# ============================================================
# Define Models
# ============================================================

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        True
    ),

    "SVM (RBF)": (
        SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42
        ),
        True
    ),

    "KNN": (
        KNeighborsClassifier(
            n_neighbors=5
        ),
        True
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        False
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            random_state=42
        ),
        False
    ),
}

results = []

print("\nModels Loaded Successfully")
print("-"*70)

for model_name in models.keys():
    print(model_name)

print("\nTotal Models :", len(models))

# ============================================================
# Train, Evaluate and Plot Confusion Matrices
# ============================================================
best_model = None
best_model_name = ""
best_accuracy = 0
best_macro_f1 = 0

for name, (model, needs_scaling) in models.items():

    print("\n")
    print("=" * 70)
    print(f"Training Model : {name}")
    print("=" * 70)

    if needs_scaling:
        Xtr = X_train_scaled
        Xte = X_test_scaled
    else:
        Xtr = X_train
        Xte = X_test

    start_time = time.time()

    model.fit(Xtr, y_train)

    end_time = time.time()
    train_time = end_time - start_time

    y_pred = model.predict(Xte)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=label_names,
            zero_division=0
        )
    )

    print("-" * 70)
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro F1 Score  : {macro_f1:.4f}")
    print(f"Training Time   : {train_time:.2f} seconds")

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Train Time (s)": train_time,
        }
    )

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

    print("\nGenerating Confusion Matrix...")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels_sorted
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names
    )

    plt.xlabel("Predicted Activity")
    plt.ylabel("Actual Activity")
    plt.title(f"{name} - Confusion Matrix")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    filename = (
        "confusion_matrix_"
        + name.replace(" ", "_")
              .replace("(", "")
              .replace(")", "")
        + ".png"
    )

    plt.savefig(filename, dpi=120)
    plt.close()

    print(f"Confusion Matrix Saved : {filename}")

print("\n")
print("=" * 70)
print("All Models Trained Successfully")
print("=" * 70)
# ============================================================
# Model Comparison
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Macro F1", ascending=False)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_results.csv", index=False)

# ============================================================
# Model Comparison Charts
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(
    data=results_df,
    x="Macro F1",
    y="Model",
    hue="Model",
    palette="viridis",
    legend=False,
    ax=axes[0]
)

axes[0].set_title("Macro F1 Score Comparison")
axes[0].set_xlim(0, 1)

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model",
    hue="Model",
    palette="magma",
    legend=False,
    ax=axes[1]
)

axes[1].set_title("Accuracy Comparison")
axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig("model_comparison_chart.png", dpi=120)
plt.close()

print("Saved : model_comparison_chart.png")

# ============================================================
# Training Time Comparison
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df.sort_values("Train Time (s)"),
    x="Train Time (s)",
    y="Model",
    hue="Model",
    palette="coolwarm",
    legend=False
)

plt.title("Training Time Comparison")

plt.tight_layout()

plt.savefig("model_train_time_chart.png", dpi=120)

plt.close()

print("Saved : model_train_time_chart.png")

# ============================================================
# Best Model Summary
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model Name      : {best_model_name}")
print(f"Accuracy        : {best_accuracy:.4f}")
print(f"Macro F1 Score  : {best_macro_f1:.4f}")

# ============================================================
# Project Conclusion
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT CONCLUSION")
print("=" * 70)

print(f"""
A total of {len(models)} machine learning models were trained
and evaluated on the MHEALTH Human Activity Recognition dataset.

Best Performing Model
---------------------
Model        : {best_model_name}
Accuracy     : {best_accuracy:.4f}
Macro F1     : {best_macro_f1:.4f}
""")

print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)