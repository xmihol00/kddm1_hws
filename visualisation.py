import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# separate column names by their data types
header_df = pd.read_csv("visual-dataset.csv", nrows=2)
continuous_cols = header_df.loc[:, (header_df == "continuous").any()].columns.values.tolist()
nominal_categorical_cols = header_df.loc[:, ((header_df != "continuous") & (header_df != "class") & (header_df.replace(r"^(\d\s*)+$", "ordinal", regex=True) != "ordinal")).all()].columns.values.tolist()
ordinal_categorical_cols = header_df.loc[:, (header_df.replace(r"^(\d\s*)+$", "ordinal", regex=True) == "ordinal").any()].columns.values.tolist()
binary_class_col = header_df.loc[:, (header_df == "class").any()].columns.values.tolist()[0]

# load the dataset without the additional header rows
data_df = pd.read_csv("visual-dataset.csv", skiprows=range(1, 3))

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

# plot the PCA results in a 2D scatter plot without class separation
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c="gray")
plt.axis("equal")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.tight_layout()
plt.savefig("visualisation_pca_2D.png", dpi=500)
plt.show()

# plot the PCA results in a scatter plot with the class labels as the color
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[y == "Yes", 0], X_pca[y == "Yes", 1], c='m', label="Yes")
plt.scatter(X_pca[y == "No", 0], X_pca[y == "No", 1], c='y', label="No", alpha=0.3)
plt.axis("equal")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.tight_layout()
plt.savefig("visualisation_pca_2D_classes.png", dpi=500)
plt.show()

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
plt.savefig("visualisation_pca_2D_subplots.png", dpi=500)
plt.show()

# apply PCA, reduce to 3 dimensions
X_pca = PCA(n_components=3).fit_transform(X)

# plot the PCA results in a 3D scatter plot without class separation
plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d")
ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c="gray")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
plt.tight_layout()
plt.savefig("visualisation_pca_3D.png", dpi=500)
plt.show()

# plot the PCA results in a 3D scatter plot with the class labels as the color
plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d")
ax.scatter3D(X_pca[y == "Yes", 0], X_pca[y == "Yes", 1], X_pca[y == "Yes", 2], c='m', label="Yes")
ax.scatter3D(X_pca[y == "No", 0], X_pca[y == "No", 1], X_pca[y == "No", 2], c='y', label="No", alpha=0.3)
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
plt.legend()
plt.tight_layout()
plt.savefig("visualisation_pca_3D_classes.png", dpi=500)
plt.show()

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
plt.savefig("visualisation_pca_3D_subplots.png", dpi=500)
plt.show()

