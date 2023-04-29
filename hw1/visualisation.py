import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency
import seaborn as sns

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denominator = min((kcorr-1), (rcorr-1))
    if denominator <= 0:
        return 0
    return np.sqrt(phi2corr / denominator)

# separate column names by their data types
header_df = pd.read_csv("visual-dataset.csv", nrows=2)
continuous_cols = header_df.loc[:, (header_df == "continuous").any()].columns.values.tolist()
nominal_categorical_cols = header_df.loc[:, ((header_df != "continuous") & (header_df != "class") & (header_df.replace(r"^(\d\s*)+$", "ordinal", regex=True) != "ordinal")).all()].columns.values.tolist()
ordinal_categorical_cols = header_df.loc[:, (header_df.replace(r"^(\d\s*)+$", "ordinal", regex=True) == "ordinal").any()].columns.values.tolist()
binary_class_col = header_df.loc[:, (header_df == "class").any()].columns.values.tolist()[0]
categorical_cols = nominal_categorical_cols + ordinal_categorical_cols + [binary_class_col]

# load the dataset without the additional header rows
data_df = pd.read_csv("visual-dataset.csv", skiprows=range(1, 3))
data_df[categorical_cols] = data_df[categorical_cols].astype('category')

# preprocess the dataset, i.e. scale the continuous features and encode the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_cols),
        ("cat_nom", OneHotEncoder(), nominal_categorical_cols),
        ("cat_ord", OrdinalEncoder(), ordinal_categorical_cols)
    ])

X = preprocessor.fit_transform(data_df.drop(binary_class_col, axis=1))
y = data_df[binary_class_col]

# apply PCA, reduce to 2 dimensions
X_pca = PCA(n_components=2).fit_transform(X)

# plot the 2D PCA results in a scatter plot as two subplots, first without class separation and then with the class labels as the color
figure, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].title.set_text("Without class separation")
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c="gray")
axes[0].axis("equal")
axes[0].set_xlabel("First Principal Component")
axes[0].set_ylabel("Second Principal Component")

axes[1].title.set_text("With class separation")
axes[1].scatter(X_pca[y == "Yes", 0], X_pca[y == "Yes", 1], c='m', label="Yes")
axes[1].scatter(X_pca[y == "No", 0], X_pca[y == "No", 1], c='y', label="No", alpha=0.3)
axes[1].axis("equal")
axes[1].set_xlabel("First Principal Component")
axes[1].set_ylabel("Second Principal Component")
axes[1].legend()

plt.subplots_adjust(left=0.05, bottom=0.09, right=0.98, top=0.95)
plt.savefig("visualisation/pca_2D_subplots.png", dpi=500)
plt.show()

# apply PCA, reduce to 3 dimensions
X_pca = PCA(n_components=3).fit_transform(X)

# plot the 3D PCA results in a scatter plot as two subplots, first without class separation and then with the class labels as the color
figure, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
axes[0].title.set_text("Without class separation")
axes[0].scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c="gray")
axes[0].set_xlabel("First Principal Component")
axes[0].set_ylabel("Second Principal Component")
axes[0].set_zlabel("Third Principal Component")

axes[1].title.set_text("With class separation")
axes[1].scatter3D(X_pca[y == "Yes", 0], X_pca[y == "Yes", 1], X_pca[y == "Yes", 2], c='m', label="Yes")
axes[1].scatter3D(X_pca[y == "No", 0], X_pca[y == "No", 1], X_pca[y == "No", 2], c='y', label="No", alpha=0.3)
axes[1].set_xlabel("First Principal Component")
axes[1].set_ylabel("Second Principal Component")
axes[1].set_zlabel("Third Principal Component")
axes[1].legend()

plt.subplots_adjust(left=0.0, bottom=0.02, right=0.96, top=0.96, wspace=0.1)
plt.savefig("visualisation/pca_3D_subplots.png", dpi=500)
plt.show()

# apply LDA, reduce to 1 dimension
X_lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)

# plot the 1D LDA results in a scatter plot as two subplots, first without class separation and then with the class labels as the color
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].title.set_text("Without class separation")
axes[0].scatter(X_lda, [0] * len(X_lda), c="gray")
axes[0].set_xlabel("Linear Discriminant")
axes[0].set_yticks([])
axes[0].set_ylim(-0.2, 0.2)

axes[1].title.set_text("With class separation")
axes[1].scatter(X_lda[y == "Yes"], [0.1] * len(X_lda[y == "Yes"]), c='m', label="Yes")
axes[1].scatter(X_lda[y == "No"], [-0.1] * len(X_lda[y == "No"]), c='y', label="No")
axes[1].set_xlabel("Linear Discriminant")
axes[1].set_yticks([])
axes[1].set_ylim(-0.2, 0.2)
axes[1].legend()

plt.subplots_adjust(left=0.05, bottom=0.12, right=0.95, top=0.91)
plt.savefig("visualisation/lda_1D_subplots.png", dpi=500)
plt.show()

# calculate correlation matrix
cols = continuous_cols + categorical_cols
correlation_matrix_df = pd.DataFrame(index=cols, columns=cols)

for col1 in cols:
    for col2 in cols:
        if col1 in continuous_cols and col2 in continuous_cols:
            # Pearson's correlation for continuous-continuous pairs
            if data_df[col1].nunique() == 1 or data_df[col2].nunique() == 1:
                correlation_matrix_df.loc[col1, col2] = np.nan
            else:
                correlation_matrix_df.loc[col1, col2] = data_df[col1].corr(data_df[col2])
        elif col1 in categorical_cols and col2 in categorical_cols:
            # Cramér's V for categorical-categorical pairs
            correlation_matrix_df.loc[col1, col2] = cramers_v(data_df[col1], data_df[col2])
        else:
            # Point-biserial correlation for continuous-categorical pairs
            continuous_var, categorical_var = (col1, col2) if col1 in continuous_cols else (col2, col1)
            if data_df[continuous_var].nunique() == 1 or data_df[categorical_var].nunique() == 1:
                correlation_matrix_df.loc[col1, col2] = np.nan
            else:
                correlation_matrix_df.loc[col1, col2] = pointbiserialr(data_df[categorical_var].cat.codes, data_df[continuous_var])[0]


correlation_matrix_df = correlation_matrix_df.astype(float)
# find the 3 most correlated features with the 'Attrition' column
top_3_columns = correlation_matrix_df["Attrition"].apply(abs).sort_values(ascending=False).head(4)
# remove the 'Attrition' itself from the list
top_3_columns = list(top_3_columns.drop("Attrition").index)

correlation_matrix_df = correlation_matrix_df.round(2)

# create correlation heatmap
plt.figure(figsize=(17, 11))
sns.heatmap(correlation_matrix_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 9})
plt.tight_layout()
plt.savefig("visualisation/correlation_heatmap.png", dpi=500)
plt.show()

# convert the categorical columns to numerical codes
data_df_numerical = data_df[top_3_columns].apply(lambda x: x.cat.codes)

# plot the top 3 most important columns in a 3D scatter plot as two subplots, 
# first without class separation and then with the class labels as the color
figure, axes = plt.subplots(1, 2, figsize=(12, 6.5), subplot_kw={"projection": "3d"})

# get unique categories for each of the top 3 columns
x_categories = data_df[top_3_columns[0]].cat.categories
y_categories = data_df[top_3_columns[1]].cat.categories
z_categories = data_df[top_3_columns[2]].cat.categories

# get the tick positions for each axis
x_ticks = range(len(x_categories))
y_ticks = range(len(y_categories))
z_ticks = range(len(z_categories))

# set tick positions and labels for each axis
for ax in axes:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_categories, rotation=45)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_categories, rotation=90, va="center")

    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_categories)

axes[0].title.set_text("Without class separation")
axes[0].scatter3D(data_df_numerical[top_3_columns[0]], data_df_numerical[top_3_columns[1]], data_df_numerical[top_3_columns[2]], c="gray")
axes[0].set_xlabel(top_3_columns[0])
axes[0].set_ylabel(top_3_columns[1])
axes[0].set_zlabel(top_3_columns[2])

axes[1].title.set_text("With class separation")
axes[1].scatter3D(data_df_numerical[top_3_columns[0]][y == "Yes"], data_df_numerical[top_3_columns[1]][y == "Yes"], data_df_numerical[top_3_columns[2]][y == "Yes"], c='m', label="Yes", s=80)
axes[1].scatter3D(data_df_numerical[top_3_columns[0]][y == "No"], data_df_numerical[top_3_columns[1]][y == "No"], data_df_numerical[top_3_columns[2]][y == "No"], c='y', label="No")
axes[1].set_xlabel(top_3_columns[0])
axes[1].set_ylabel(top_3_columns[1])
axes[1].set_zlabel(top_3_columns[2])
axes[1].legend()

plt.subplots_adjust(left=0.0, bottom=0.1, right=0.96, top=0.96, wspace=0.1)
plt.savefig("visualisation/feature_importance_3D_subplots.png", dpi=500)
plt.show()

# normalize continuous columns except "StandardHours", which is all 80
continuous_cols_cleaned = [col for col in continuous_cols if col != "StandardHours"]
data_df[continuous_cols_cleaned] = ((data_df[continuous_cols_cleaned] - data_df[continuous_cols_cleaned].min()) / 
                                    (data_df[continuous_cols_cleaned].max() - data_df[continuous_cols_cleaned].min()))
# normalize "StandardHours" to 1
data_df["StandardHours"] = data_df["StandardHours"] / data_df["StandardHours"].max()

# prepare subplots
total_plots = len(continuous_cols) + len(nominal_categorical_cols) + len(ordinal_categorical_cols)
ncols = 4
nrows = int(np.ceil(total_plots / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 15))
axes = axes.flatten()

# create bar plots for categorical columns
plot_index = 0
for cat_col in nominal_categorical_cols + ordinal_categorical_cols:
    counts = data_df.groupby([cat_col, binary_class_col]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[plot_index])
    axes[plot_index].set_title(cat_col, fontsize=6)
    axes[plot_index].set_xlabel("")
    axes[plot_index].set_ylabel("Count", fontsize=6)
    axes[plot_index].legend(fontsize=6)
    axes[plot_index].tick_params(axis="both", labelsize=5, rotation=0)
    plot_index += 1

# create bar plots for continuous columns with intervals
interval_labels = ["0.0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
for cont_col in continuous_cols:
    data_df[f"{cont_col}_interval"] = pd.cut(data_df[cont_col], bins=[0, 0.25, 0.5, 0.75, 1], labels=interval_labels, include_lowest=True)
    counts = data_df.groupby([f"{cont_col}_interval", binary_class_col]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[plot_index])
    axes[plot_index].set_title(cont_col, fontsize=6)
    axes[plot_index].set_xlabel("")
    axes[plot_index].set_ylabel("Count", fontsize=6)
    axes[plot_index].legend(fontsize=6)
    axes[plot_index].tick_params(axis="both", labelsize=5, rotation=0)
    plot_index += 1

# remove any extra subplots
for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])

# adjust spacing between subplots to avoid overlapping
plt.subplots_adjust(left=0.05, bottom=0.02, right=0.99, top=0.98, wspace=0.25, hspace=0.3)
plt.savefig("visualisation/bar_plots.png", dpi=500)
plt.show()
