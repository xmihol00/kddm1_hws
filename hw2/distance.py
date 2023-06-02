import pandas as pd
import numpy as np

# header: Temperature, Wind, Speed, Wind, Angle, Date, Precipitation, Traffic, Pollution

df = pd.read_csv("distance-function-dataset.csv")

# convert date to number
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].astype(np.int64)

# standardize all columns between 0 and 1
df = (df - df.min()) / (df.max() - df.min())

# save to csv
df.to_csv("distance-function-dataset-standardized.csv", index=False)
