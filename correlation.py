import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

transformations = [ lambda x: x, lambda x: x**2, lambda x: np.sqrt(np.abs(x)), lambda x: np.log(np.abs(x) + 1e-10), 
                    lambda x: np.log10(np.abs(x) + 1e-10), lambda x: np.exp(x), lambda x: 1/x, lambda x: np.sin(x), lambda x: np.cos(x), 
                    lambda x: np.tan(x), lambda x: np.arcsin(x), lambda x: x + x**2, lambda x: x + x**2 + x**3, lambda x: x**3, 
                    lambda x: x**4, lambda x: x**5, lambda x: 2**x ]

transformation_names = [ "identity", "square", "square root of the absolute value", "natural logarithm of the absolute value", 
                         "base 10 logarithm of the absolute value", "exponential", "reciprocal", "sine", "cosine", "tangent", 
                         "arcsine", "x + x^2", "x + x^2 + x^3", "x^3", "x^4", "2^x" ]

data_df = pd.read_csv('correlation-dataset.csv')

for transformation, name in zip(transformations, transformation_names):
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
        sorted_key = tuple(sorted(idx))
        key = tuple(idx)
        if sorted_key not in top_pairs_dict and key not in top_pairs_dict:
            top_pairs_dict[key] = value
            count += 1
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

    plt.figure(figsize=(12, 12))
    sns.heatmap(top_pairs_df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, xticklabels=columns, yticklabels=rows, square=False)
    plt.title(f"Original and transformed data with {name} function")
    plt.show()

exit()
