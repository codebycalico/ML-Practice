import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Partioning a dataset into separate training and test datasets
# Load the wine data from the UCI ML Repo
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#print(df_wine)

# Label the columns
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic Acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

#print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()
#print(df_wine)

# Perform the data splitting
# using the colon : selects all rows, the following number says what column to start from
# when gathering x data, start from column 1 to exclude the class label column,
# start from the second column [1] and go to the last
x = df_wine.iloc[:, 1:].values
#print(x)
y = df_wine.iloc[:, 0].values
#print(y)

# separate x and y from df_wine
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
#print(x_train, x_test, y_train, y_test)

# Bringing features onto the same scale

# perform MinMax normalization
# using normalization approach where each value will be sutracted by the minimum value of the column,
# divided by the range, which is the maximum value of the column minus the minimum value of the column
mms = MinMaxScaler()

# normalizes the data (now will be range from 0 - 1)
# normalized column-wise
x_train_norm = mms.fit_transform(x_train)
x_test_norm = mms.transform(x_test)

#print(x_train_norm)
#print(x_test_norm)

#view as a data frame
#print(pd.DataFrame(x_train_norm).describe())

# perform standardization
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

#print(pd.DataFrame(x_train_std).describe())
#print(pd.DataFrame(x_test_std).describe())

# want to be able to extract the important features from the vast initial features

# apply the SBS feature selection implemented by the authors
# https://github.com/rasbt/machine-learning-book/blob/main/ch04/ch04.ipynb
# Sequential feature selection algorithms

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
# apply the SBS feature selection
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(x_train_std, y_train)

# plotting performane of feature subsets
k_feat = [len(k) for k in sbs.subsets]

# show the graph
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
#plt.savefig('figures/04_09.png', dpi=300)
plt.show()

# conclusion: three features gives an accurate model (100% / 1.00 score)

# assessing feature importance with random forests

# build a random forest model
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(x_train, y_train)

feat_labels = df_wine.columns[1:]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title("Feature importance")
plt.bar(range(x_train.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
#plt.savefig('figures/04_10.png', dpi=300)
plt.show()

# conclusions: the higher the magnitude, the more important the feature is to the model

# select the top five important features
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
x_selected = sfm.transform(x_train)
print("Number of features that meet the specified threshold", 'criterion:', x_selected.shape[1])

# printing out the top five features
for f in range(x_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# possible next step for ML: build a second model using the top five features