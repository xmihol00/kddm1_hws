import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from scipy.stats import spearmanr, kendalltau, pearsonr
import heapq
import dcor
from sklearn.cross_decomposition import CCA
from minepy import MINE

SHOW_PLOTS = False
TRANSFORMATIONS = True
SPECIAL_CORRELATIONS = True

transformations = [ lambda x: x, lambda x: x**2, lambda x: np.sqrt(np.abs(x)), lambda x: np.log(np.abs(x) + 1e-10), 
                    lambda x: np.log10(np.abs(x) + 1e-10), lambda x: np.exp(x), lambda x: 1.0 / x, lambda x: np.sin(x), lambda x: np.cos(x), 
                    lambda x: np.tan(x), lambda x: np.arcsin(np.tanh(x)), lambda x: np.arccos(np.tanh(x)), lambda x: np.arctan(x), 
                    lambda x: np.tanh(x), lambda x: 1.0 / (1.0 + np.exp(-x)), lambda x: 2**x, lambda x: x + x**2, lambda x: x + x**2 + x**3, 
                    lambda x: x**3, lambda x: x**4, lambda x: x**5, lambda x: x**6, lambda x: x**7, lambda x: x**8, lambda x: x**9, 
                    lambda x: x**10 ]

transformation_names = [ "identity", "square", "square root of the absolute value", "natural logarithm of the absolute value", 
                         "base 10 logarithm of the absolute value", "exponential", "reciprocal", "sine", "cosine", "tangent", 
                         "arkus sine of the hyperbolic tangent", "arkus cosine of the hyperbolic tangent", "arkus tangent", 
                         "hyperbolic tangent", "logistic sigmoid", "2^x", "x + x^2", "x + x^2 + x^3", "x^3", "x^4", "x^5", "x^6", 
                         "x^7", "x^8", "x^9", "x^10" ]

if len(transformations) != len(transformation_names):
    raise ValueError("The number of transformations and transformation names must be the same.")

data_df = pd.read_csv("correlation-dataset.csv")
best_names = []
best_pairs = []
best_correlations = {}

if TRANSFORMATIONS:
    for transformation, name in zip(transformations, transformation_names):
        print(f"Processing '{name}' transformation: {transformation(4)} ...")
        transformed_data_df = data_df.apply(transformation)

        # combine the two data frames horizontally
        combined_data_df = pd.concat([data_df, transformed_data_df], axis=1)
        corr_mat_df = combined_data_df.corr()

        corr_mat_selected_df = corr_mat_df.iloc[:data_df.shape[1], data_df.shape[1]:]
        corr_mat_no_self = corr_mat_selected_df.copy()
        for i in range(corr_mat_no_self.shape[0]):
            corr_mat_no_self.iat[i, i] = 0

        sorted_corr = corr_mat_no_self.unstack().sort_values(ascending=False, key=abs)

        top_pairs_dict = {}
        count = 0
        for idx, value in sorted_corr.items():
            sorted_key_asc = tuple(sorted(idx))
            sorted_key_desc = tuple(sorted(idx, reverse=True))
            key = tuple(idx)
            # check if the same pair is already in the dictionary, if so, it already has the higher correlation, otherwise, insert it
            if sorted_key_asc not in top_pairs_dict and sorted_key_desc not in top_pairs_dict:
                top_pairs_dict[key] = value
                count += 1

            if key in best_correlations:
                last_name, last_value = best_correlations[key]
                if abs(value) > abs(last_value):
                    best_correlations[key] = (name, value)
            elif abs(value) > 0.5:
                best_correlations[key] = (name, value)

            if count >= 10:
                break
            
        columns = []
        rows = []
        for key in top_pairs_dict:
            columns.append(key[0])
            rows.append(key[1])
        columns = sorted(list(dict.fromkeys(columns)))
        rows = sorted(list(dict.fromkeys(rows)))

        top_pairs_df = corr_mat_selected_df.loc[rows, columns]

        plt.figure(figsize=(8, 8))
        sns.heatmap(top_pairs_df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, xticklabels=columns, yticklabels=rows, square=False)
        plt.title(f"Original and transformed data with {name} function")
        plt.tight_layout()
        plt.savefig(f"correlation/{name.replace(' ', '_')}", dpi=500)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    best_correlations_string_keys = {}
    for key in best_correlations:
        best_names.append(key[0])
        best_names.append(key[1])
        best_pairs.append(key)
        best_correlations_string_keys['_'.join(key)] = best_correlations[key]

    with open("correlation/best_corr_transformations.json", 'w') as f:
        json.dump(best_correlations_string_keys, f, indent=4)

if SPECIAL_CORRELATIONS:
    non_linear_correlation_names = [ "Spearman", "Kendall", "Distance", "Canonical" ]
    non_linear_correlations = [ lambda x, y: spearmanr(x, y)[0], lambda x, y: kendalltau(x, y)[0], dcor.distance_correlation,
                                lambda x, y: pearsonr(*list(map(lambda z: z[:, 0], 
                                                                CCA(n_components=1)
                                                                    .fit(x.reshape(-1, 1), y.reshape(-1, 1))
                                                                    .transform(x.reshape(-1, 1), y.reshape(-1, 1)))))[0] ]

    for non_linear_correlation, non_linear_correlation_name in zip(non_linear_correlations, non_linear_correlation_names):
        print(f"Processing '{non_linear_correlation_name}' correlation ...")

        results = []
        for col1 in data_df.columns:
            for col2 in data_df.columns:
                if col1 != col2:
                    x = data_df[col1].to_numpy().astype(np.float64)
                    y = data_df[col2].to_numpy().astype(np.float64)
                    correlation = non_linear_correlation(x, y)
                    results.append((abs(correlation), (col1, col2), correlation))

        top_20_pairs = heapq.nlargest(20, results, key=lambda x: x[0])

        best_correlations_string_keys = {}
        for i in range(0, 20, 2):
            j = i + 1
            _, key1, correlation1 = top_20_pairs[i]
            _, key2, correlation2 = top_20_pairs[j]
            best_names.append(key1[0])
            best_names.append(key1[1])
            best_pairs.append(key1)

            if key1 in best_correlations:
                best_correlations_string_keys['_'.join(key1)] = (non_linear_correlation_name, correlation1)
            elif key2 in best_correlations:
                best_correlations_string_keys['_'.join(key2)] = (non_linear_correlation_name, correlation2)
            else:
                best_correlations_string_keys['_'.join(key1)] = (non_linear_correlation_name, correlation1)

        with open(f"correlation/best_corr_{non_linear_correlation_name.lower()}.json", 'w') as f:
            json.dump(best_correlations_string_keys, f, indent=4)

    best_names = list(set(best_names))
    selected_data_df = data_df[best_names]
    non_linear_correlation_name = "Maximal Information Coefficient"
    print(f"Processing '{non_linear_correlation_name}' correlation ...")

    results = []
    for col1 in selected_data_df.columns:
        for col2 in selected_data_df.columns:
            if col1 != col2:
                x = selected_data_df[col1].to_numpy().astype(np.float64).reshape(-1)
                y = selected_data_df[col2].to_numpy().astype(np.float64).reshape(-1)
                mine = MINE()
                mine.compute_score(x, y)
                correlation = mine.mic()
                results.append((abs(correlation), (col1, col2), correlation))

    top_20_pairs = heapq.nlargest(20, results, key=lambda x: x[0])

    best_correlations_string_keys = {}
    for i in range(0, 20, 2):
        j = i + 1
        _, key1, correlation1 = top_20_pairs[i]
        _, key2, correlation2 = top_20_pairs[j]
        best_names.append(key1[0])
        best_names.append(key1[1])
        best_pairs.append(key1)

        if key1 in best_correlations:
            best_correlations_string_keys['_'.join(key1)] = (non_linear_correlation_name, correlation1)
        elif key2 in best_correlations:
            best_correlations_string_keys['_'.join(key2)] = (non_linear_correlation_name, correlation2)
        else:
            best_correlations_string_keys['_'.join(key1)] = (non_linear_correlation_name, correlation1)

    with open("correlation/best_corr_MIC.json", 'w') as f:
        json.dump(best_correlations_string_keys, f, indent=4)

best_pairs = list(set(best_pairs))
with open("correlation/best_corr_pairs.json", 'w') as f:
    json.dump(best_pairs, f, indent=4)

ncols = 4
nrows = int(np.ceil(len(best_pairs) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 15))
axes = axes.flatten()
for i, (col1, col2) in enumerate(best_pairs):
    ax = axes[i]
    ax.scatter(data_df[col1], data_df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}")

for i in range(i, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("correlation/best_corr_pairs", dpi=500)
if SHOW_PLOTS:
    plt.show()
plt.close()
