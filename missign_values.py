import pandas as pd
import matplotlib.pyplot as plt

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

data = pd.read_csv("missing-values-dataset.csv")

missing_values = data.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)

rows_with_missing_values = data[data.isnull().any(axis=1)]
print("\nRows with missing values:")
print(rows_with_missing_values)

missing_height = data[~data["height"].isnull()]
print("\nHeight distribution when height is not missing:")
print(missing_height["height"].describe())

missing_english_skills = data[~data["english_skills"].isnull()]
print("\nEnglish skills distribution when English skills are missing:")
print(missing_english_skills["english_skills"].describe())

min_height = data["height"].min()
min_english_skills = data["english_skills"].min()

print("\nSmallest recorded height:", min_height)
print("Smallest recorded English skills:", min_english_skills)

bin_width_height = 5
bin_width_english_skills = 3

plot_histogram(data, "height", bin_width_height)
plot_histogram(data, "english_skills", bin_width_english_skills)
plot_histogram(data, "likes_ananas_on_pizza", num_bins=20)
plot_histogram(data, "semester", 1)

missing_english_skills = data[data["english_skills"].isnull()]
missing_semester = data[data["semester"].isnull()]

missing_height = data[data["height"].isnull()]
missing_books_per_year = data[data["books_per_year"].isnull()]

bin_width_age = 1
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_histogram(data, "age", bin_width_age, "Overall age distribution", axes[0])
plot_histogram(missing_english_skills, "age", bin_width_age, "Age distribution when English skills are missing", axes[1])
plot_histogram(missing_semester, "age", bin_width_age, "Age distribution when semester is missing", axes[2])
plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.93, wspace=0.15)
plt.savefig("missing_values/age_distribution", dpi=500)
plt.show()

bin_width_gender = 0
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_histogram(data, "gender", title="Overall gender distribution", ax=axes[0], num_bins=2)
plot_histogram(missing_height, "gender", title="Gender distribution when height is missing", ax=axes[1], num_bins=2)
plot_histogram(missing_books_per_year, "gender", title="Gender distribution when books per year are missing", ax=axes[2], num_bins=2)
plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.93, wspace=0.15)
plt.savefig("missing_values/gender_distribution", dpi=500)
plt.show()

print("age arithmetic mean:", data["age"].mean())
print("age median:", data["age"].median())
print("semester arithmetic mean:", data["semester"].mean())
print("semester median:", data["semester"].median())
print("height arithmetic mean:", data["height"].mean())
print("english skills arithmetic mean:", data["english_skills"].mean())
print("likes ananas on pizza arithmetic mean:", data["likes_ananas_on_pizza"].mean())
print("likes chocolate arithmetic mean:", data["likes_chocolate"].mean())
print("gender arithmetic mean", data["gender"].mean())
print("books per year arithmetic mean", data["books_per_year"].mean())

total_count = len(data)
male_count = len(data[data["gender"] == 0])
female_count = len(data[data["gender"] == 1])

male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f"Male percentage: {male_percentage:.2f} %")
print(f"Female percentage: {female_percentage:.2f} %")
