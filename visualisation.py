import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px

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
plt.savefig("visualisation_lda_1D_subplots.png", dpi=500)
plt.show()

## select only the continuous features
#continous_data_df = data_df[continuous_cols + [binary_class_col]].copy()
## normalize the continuous features to the range [0, 1]
#continous_data_df[continuous_cols] = ((continous_data_df[continuous_cols] - continous_data_df[continuous_cols].min()) / 
#                                      (continous_data_df[continuous_cols].max() - continous_data_df[continuous_cols].min()))
#
#continous_data_df = continous_data_df.replace("Yes", 1)
#continous_data_df = continous_data_df.replace("No", 0)

## create the Parallel Coordinates plot
#fig = px.parallel_coordinates(
#    continous_data_df,
#    color=binary_class_col,
#    labels=dict(zip(continous_data_df.columns, continous_data_df.columns)),  # Custom labels can be provided here if needed
#    color_continuous_scale=px.colors.diverging.Tealrose,  # Choose the color scale
#    color_continuous_midpoint=0.5  # Set the color midpoint (adjust if needed)
#)

#fig.show()

# get feature importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# combine feature importances with their column names
columns_encoded = continuous_cols + preprocessor.named_transformers_["cat_nom"].get_feature_names_out(nominal_categorical_cols).tolist() + ordinal_categorical_cols
feature_importances = pd.DataFrame({"column": columns_encoded, "importance": rf.feature_importances_})

# select top 3 most important columns
top_3_columns = feature_importances.nlargest(3, "importance")["column"].tolist()

# plot the top 3 most important columns in a 3D scatter plot as two subplots, 
# first without class separation and then with the class labels as the color
figure, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
axes[0].title.set_text("Without class separation")
axes[0].scatter3D(data_df[top_3_columns[0]], data_df[top_3_columns[1]], data_df[top_3_columns[2]], c="gray")
axes[0].set_xlabel(top_3_columns[0])
axes[0].set_ylabel(top_3_columns[1])
axes[0].set_zlabel(top_3_columns[2])

axes[1].title.set_text("With class separation")
axes[1].scatter3D(data_df[top_3_columns[0]][y == "Yes"], data_df[top_3_columns[1]][y == "Yes"], data_df[top_3_columns[2]][y == "Yes"], c='m', label="Yes")
axes[1].scatter3D(data_df[top_3_columns[0]][y == "No"], data_df[top_3_columns[1]][y == "No"], data_df[top_3_columns[2]][y == "No"], c='y', label="No", alpha=0.3)
axes[1].set_xlabel(top_3_columns[0])
axes[1].set_ylabel(top_3_columns[1])
axes[1].set_zlabel(top_3_columns[2])
axes[1].legend()

plt.subplots_adjust(left=0.0, bottom=0.02, right=0.96, top=0.96, wspace=0.1)
plt.savefig("visualisation_feature_importance_3D_subplots.png", dpi=500)
plt.show()
