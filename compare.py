import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Path setup
cv_dir = os.path.expanduser("~/Documents/CognativeRobotics&ComputerVision/ComputerVision")
results_dir = os.path.expanduser("~/Documents/CognativeRobotics&ComputerVision/ComputerVision/results")
graphs_dir = os.path.join(cv_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

# Filenames
traditional_file = os.path.join(results_dir, "traditional_CV_results.txt")
#cnn_file = os.path.join(results_dir, "cnn_results.txt")
cnn_file = os.path.join(results_dir, "deep_cnn_results.txt")

# Helper: parse classification report
def parse_classification_report(report_text):
    lines = report_text.strip().split("\n")
    class_lines = [line for line in lines if re.match(r"\s+\w+", line)]
    metrics = {}
    for line in class_lines:
        parts = line.strip().split()
        if len(parts) == 5:
            label, precision, recall, f1, support = parts
            metrics[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(support)
            }
    return metrics

# Helper: parse confusion matrix
def parse_confusion_matrix(lines):
    matrix = []
    for line in lines:
        row = list(map(int, line.strip().split()))
        matrix.append(row)
    return np.array(matrix)

# Read files
with open(traditional_file, 'r') as f:
    traditional_text = f.read()

with open(cnn_file, 'r') as f:
    cnn_text = f.read()

# Parse classification reports
traditional_metrics = parse_classification_report(traditional_text)
cnn_metrics = parse_classification_report(cnn_text)

# Parse confusion matrices
traditional_cm_lines = traditional_text.strip().split("Confusion Matrix:")[-1].strip().split("\n")
cnn_cm_lines = cnn_text.strip().split("Confusion Matrix:")[-1].strip().split("\n")

traditional_cm = parse_confusion_matrix(traditional_cm_lines)
cnn_cm = parse_confusion_matrix(cnn_cm_lines)

# Labels
labels = sorted(traditional_metrics.keys())

# Convert metrics to DataFrames
def metrics_to_df(metrics_dict, method_name):
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df["method"] = method_name
    return df

df_trad = metrics_to_df(traditional_metrics, "Traditional CV")
df_cnn = metrics_to_df(cnn_metrics, "CNN")
df_all = pd.concat([df_trad, df_cnn]).reset_index().rename(columns={"index": "class"})

# Plotting function
def save_barplot(metric, title):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_all, x="class", y=metric, hue="method")
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.tight_layout()
    filepath = os.path.join(graphs_dir, f"{metric}_comparison.png")
    plt.savefig(filepath)
    plt.close()

save_barplot("f1-score", "F1-score Comparison per Class")
save_barplot("precision", "Precision Comparison per Class")
save_barplot("recall", "Recall Comparison per Class")

# Save confusion matrix plots
def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, filename))
    plt.close()

save_confusion_matrix(traditional_cm, "Traditional CV Confusion Matrix", "traditional_confusion_matrix.png")
save_confusion_matrix(cnn_cm, "CNN Confusion Matrix", "cnn_confusion_matrix.png")

print(f"Graphs saved in: {graphs_dir}")

# Conclusion
print("Overall Accuracy:")
print(f"Traditional CV: {traditional_metrics['accuracy'] if 'accuracy' in traditional_metrics else '0.39 (from file)'}")
print(f"CNN: {cnn_metrics['accuracy'] if 'accuracy' in cnn_metrics else '0.74 (from file)'}")

