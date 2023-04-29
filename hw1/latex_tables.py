import json
import os
import pandas as pd

def json_to_list(data):
    data_list = []
    for key, value in data.items():
        variable1, variable2 = key.split("_")
        method, correlation = value
        data_list.append({"Variable A": variable1, "Variable B": variable2, method: round(correlation, 3)})
    return data_list

all_data_lists = []

with open("correlation/best_corr_transformations.json", "r") as f:
    data = json.load(f)
    for key, value in data.items():
        variable1, variable2 = key.split("_")
        method, correlation = value
        all_data_lists.append({"Variable A": variable1, "Variable B": variable2, "Transformation of variable B": method, 
                               "Pearson of Variable A and transformed Variable B": round(correlation, 3)})

for file_name in [ "best_corr_canonical.json", "best_corr_distance.json", "best_corr_kendall.json", "best_corr_MIC.json", "best_corr_spearman.json"]:
    file_name = os.path.join("correlation", file_name)
    with open(file_name, "r") as f:
        data = json.load(f)
        data_list = json_to_list(data)
        all_data_lists.extend(data_list)

df = pd.DataFrame(all_data_lists)
df_agg = df.groupby(["Variable A", "Variable B"], as_index=False).agg("first")

print(df_agg.to_latex(index=False))

exit()
with open("correlation/best_corr_transformations.json") as f:
    best_correlations = json.load(f)

corr_table = pd.DataFrame(columns=["Original variable", "Transformed variable", "Correlation", "Transformation (relationship)"])
for key, value in best_correlations.items():
    corr_table = corr_table.append({"Original variable": key.split("_")[0], 
                          "Transformed variable": key.split("_")[1],
                          "Correlation": value[1],
                          "Transformation (relationship)": value[0]}, ignore_index=True)

# sort by the absolute value of correlation
corr_table = corr_table.reindex(corr_table["Correlation"].abs().sort_values(ascending=False).index)
#print(table.to_latex(index=False))

special_corr_table = pd.DataFrame(columns=["Original variable", "Transformed variable", "Correlation (transformed)", 
                                           "Transformation (relationship)", "Correlation (non-linear method)", "Non-linear method" ])

with open("correlation/best_corr_spearman_kendall.json", "r") as f:
    best_correlations = json.load(f)

for key, value in best_correlations.items():
    selected_rows = corr_table.loc[corr_table["Original variable"] == key.split("_")[0]]
    selected_rows = selected_rows.loc[selected_rows["Transformed variable"] == key.split("_")[1]]
    if not selected_rows.empty and abs(selected_rows["Correlation"].values[0]) < (value[1]):
        special_corr_table = special_corr_table.append({"Original variable": key.split("_")[0], 
                          "Transformed variable": key.split("_")[1],
                          "Correlation (transformed)": selected_rows["Correlation"].values[0],
                          "Transformation (relationship)": selected_rows["Transformation (relationship)"].values[0],
                          "Correlation (non-linear method)": value[1],
                          "Non-linear method": value[0] }, ignore_index=True)
        
with open("correlation/best_corr_0_to_1.json", "r") as f:
    best_correlations = json.load(f)

for key, value in best_correlations.items():
    selected_rows = corr_table.loc[corr_table["Original variable"] == key.split("_")[0]]
    selected_rows = selected_rows.loc[selected_rows["Transformed variable"] == key.split("_")[1]]
    if not selected_rows.empty and abs(selected_rows["Correlation"].values[0]) < abs(value[1]):
        special_corr_table = special_corr_table.append({"Original variable": key.split("_")[0], 
                          "Transformed variable": key.split("_")[1],
                          "Correlation (transformed)": selected_rows["Correlation"].values[0],
                          "Transformation (relationship)": selected_rows["Transformation (relationship)"].values[0],
                          "Correlation (non-linear method)": value[1],
                          "Non-linear method": value[0] }, ignore_index=True)

special_corr_table = special_corr_table.reindex(special_corr_table["Correlation (non-linear method)"].abs().sort_values(ascending=False).index)
print(special_corr_table.to_latex(index=False))