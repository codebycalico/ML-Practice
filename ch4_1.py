import pandas as pd
from io import StringIO

# ***Identifying missing values in tabular data***

# creating and assigning a text file, comma separated values (csv)
# if there is an empty space (', ,' instead of ',,'), pd won't detect it
# as a missing value
csv_data = \
'''A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0
10.0, 11.0, 12.0,'''

# create a Pandas DataFrame from csv
df = pd.read_csv(StringIO(csv_data))
#print(df)

# summation of missing values in the DataFrame
df.isnull().sum()
#print(df.isnull().sum())

# convert the DataFrame into a NumPy array
#print(df.values)
df.values

# *** Eliminating training examples or features with missing values ***
# drop all rows that contain at least one missing value, applies to the row (axis = 0)
# if axis = 1, it will drop all columns that contain at least one missing value.
#print(df.dropna(axis=0))
df.dropna(axis=0)
df.dropna(axis=1)