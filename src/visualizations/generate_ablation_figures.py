import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from stress tests
models = [
    "Metadata Only",
    "ECG Only",
    "M1: ECG + Age/Sex",
    "M3: M1 + Heart Axis",
    "M4: M3 + Noise Flags",
    "M5: M4 + Missingness",
    "M6: M5 + Validated",
    "Report Text Only",
    "M7: M6 + Report Text"
]

aucs = [
    0.7820,
    0.8534,
    0.8661,
    0.8713,
    0.8707,
    0.8705,
    0.8721,
    0.9121,
    0.9411
]

perm_features = [
    "Demographics",
    "Heart Axis",
    "Validated Flag",
    "Workflow Flags (Group)",
    "Missingness Flags",
    "Noise/Drift Flags"
]

perm_drops = [
    0.0611,
    0.0360,
    0.0081,
    0.0064,
    0.0012,
    0.0000
]

os.makedirs("reports/figures", exist_ok=True)

# 1. Ablation Ladder Bar Chart
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = ['#cccccc', '#888888', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
bars = plt.barh(models, aucs, color=colors)
plt.xlim(0.7, 1.0)
plt.xlabel("ROC-AUC")
plt.title("Incremental Ablation Ladder: Modality & Feature Contributions")

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("reports/figures/ablation_ladder_chart.png", dpi=300)
plt.close()

# 2. Permutation Test Bar Chart
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")
colors_perm = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
bars = plt.barh(perm_features, perm_drops, color=colors_perm)
plt.xlabel("Drop in ROC-AUC when Shuffled")
plt.title("Permutation Stress Test: Feature Importance in Primary Model")

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'+{width:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("reports/figures/permutation_test_chart.png", dpi=300)
plt.close()

print("Figures generated successfully in reports/figures/")
