import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

data_df = pd.read_csv("correlation-dataset.csv")

with open("correlation/best_corr_pairs_adjusted.json", 'r') as f:
    best_pairs = json.load(f)

ncols = 4
nrows = int(np.ceil(len(best_pairs) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 15))
axes = axes.flatten()
for i, (col1, col2) in enumerate(best_pairs):
    ax = axes[i]
    ax.scatter(data_df[col1], data_df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}")

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("correlation/best_corr_pairs_adjusted", dpi=500)
plt.show()
plt.close()
