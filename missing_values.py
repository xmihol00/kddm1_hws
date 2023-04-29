import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, column, bin_width=3, title=None, ax=None, num_bins=None):
    if ax:
        plt.sca(ax)
    else:
        plt.figure()
    if num_bins:
        plt.hist(data[column].dropna(), bins=num_bins)
    else:
        plt.hist(data[column].dropna(), bins=range(int(data[column].min()), int(data[column].max()) + bin_width, bin_width))
    plt.xlabel(column)
    plt.ylabel("Frequency")

    if title:
        plt.title(title)
    else:
        plt.title(f"Histogram of {column} (bin width = {bin_width})")

    if ax is None:
        plt.show()

data_df = pd.read_csv("missing-values-dataset.csv")

missing_values = data_df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)

plt.figure(figsize=(13, 10))
sns.heatmap(data_df.corr().round(2), annot=True, square=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig("missing_values/correlation_heatmap", dpi=500)
plt.show()

# plot distributions of columns with missing values
figure, axes = plt.subplots(1, 1, figsize=(6, 5))
bin_width_height = 5
plot_histogram(data_df, "height", bin_width_height, "Overall height distribution", axes)
plt.tight_layout()
plt.savefig("missing_values/height_distribution", dpi=500)
plt.show()
bin_width_english_skills = 3
plot_histogram(data_df, "english_skills", bin_width_english_skills)
plot_histogram(data_df, "likes_ananas_on_pizza", num_bins=20)
plot_histogram(data_df, "semester", 1)
plot_histogram(data_df, "books_per_year", 1)

missing_english_skills = data_df[data_df["english_skills"].isnull()]
missing_semester = data_df[data_df["semester"].isnull()]
missing_height = data_df[data_df["height"].isnull()]
missing_books_per_year = data_df[data_df["books_per_year"].isnull()]

# plot age distributions
bin_width_age = 1
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_histogram(data_df, "age", bin_width_age, "Overall age distribution", axes[0])
plot_histogram(missing_english_skills, "age", bin_width_age, "Age distribution when English skills are missing", axes[1])
plot_histogram(missing_semester, "age", bin_width_age, "Age distribution when semester is missing", axes[2])
plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.93, wspace=0.15)
plt.savefig("missing_values/age_distribution", dpi=500)
plt.show()

# plot age distributions
bin_width_gender = 0
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_histogram(data_df, "gender", title="Overall gender distribution", ax=axes[0], num_bins=2)
plot_histogram(missing_height, "gender", title="Gender distribution when height is missing", ax=axes[1], num_bins=2)
plot_histogram(missing_books_per_year, "gender", title="Gender distribution when books per year are missing", ax=axes[2], num_bins=2)
plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.93, wspace=0.15)
plt.savefig("missing_values/gender_distribution", dpi=500)
plt.show()

# print some statistics
print("age arithmetic mean:", data_df["age"].mean())
print("age median:", data_df["age"].median())
print("semester arithmetic mean:", data_df["semester"].mean())
print("semester median:", data_df["semester"].median())
print("height arithmetic mean:", data_df["height"].mean())
print("english skills arithmetic mean:", data_df["english_skills"].mean())
print("english skills median:", data_df["english_skills"].median())
print("likes ananas on pizza arithmetic mean:", data_df["likes_ananas_on_pizza"].mean())
print("likes ananas on pizza median:", data_df["likes_ananas_on_pizza"].median())
print("likes chocolate arithmetic mean:", data_df["likes_chocolate"].mean())
print("likes chocolate median:", data_df["likes_chocolate"].median())
print("gender arithmetic mean", data_df["gender"].mean())
print("books per year arithmetic mean", data_df["books_per_year"].mean())
print("books per year median", data_df["books_per_year"].median())
print("books per year mode", data_df["books_per_year"].mode()[0])

total_count = len(data_df)
male_count = len(data_df[data_df["gender"] == 0])
female_count = len(data_df[data_df["gender"] == 1])

male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f"Male percentage: {male_percentage:.2f} %")
print(f"Female percentage: {female_percentage:.2f} %")

# fill missing values
data_df["semester"].fillna(data_df["semester"].median(), inplace=True)
data_df["english_skills"].fillna(data_df["english_skills"].mean(), inplace=True)
data_df.loc[data_df["gender"] == 0, "height"] = data_df.loc[data_df["gender"] == 0, "height"].fillna(171) # fill with the world-wide average for men
data_df.loc[data_df["gender"] == 1, "height"] = data_df.loc[data_df["gender"] == 1, "height"].fillna(159) # fill with the world-wide average for woman
data_df["books_per_year"].fillna(data_df["books_per_year"].median(), inplace=True)
data_df["likes_ananas_on_pizza"].fillna(0.5, inplace=True)

# print the statistics again with imputed values
print()
print("imputed age arithmetic mean:", data_df["age"].mean())
print("imputed age median:", data_df["age"].median())
print("imputed semester arithmetic mean:", data_df["semester"].mean())
print("imputed semester median:", data_df["semester"].median())
print("imputed height arithmetic mean:", data_df["height"].mean())
print("imputed english skills arithmetic mean:", data_df["english_skills"].mean())
print("imputed likes ananas on pizza arithmetic mean:", data_df["likes_ananas_on_pizza"].mean())
print("imputed likes chocolate arithmetic mean:", data_df["likes_chocolate"].mean())
print("imputed gender arithmetic mean", data_df["gender"].mean())
print("imputed books per year arithmetic mean", data_df["books_per_year"].mean())
print("imputed books per year median", data_df["books_per_year"].median())
