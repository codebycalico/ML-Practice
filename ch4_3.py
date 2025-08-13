import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])

# rename the index number (0, 1, 2) to the class label
df.columns = ['color', 'size', 'price', 'classlabel']
#print(df)

# map ordinal features, put them into the correct order that we want to see them in
size_mapping = { 'XL': 3,
                'L': 2,
                'M': 1}

# replace XL, L, and M with 3, 2, 1
df['size'] = df['size'].map(size_mapping)
#print(df)

# ***Encoding class labels***
# map categorical labels to numerical values
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
#print(class_mapping)

# replace classlabel1 and classlabel2 with 0 and 1
df['classlabel'] = df['classlabel'].map(class_mapping)
#print(df)

# reverse the mapping of numerical values back to their original categorical labels
inverse_class_mapping = {idx: label for label, idx in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inverse_class_mapping)
#print(df)

# *** One-hot encoding on nominal features ***

# array from values
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()

# select all contents of the rows, only the first column
# replace with the transformed value
# replace green, red, blue with 1, 2, 3
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)