# Repurpose code from Chapter 4 for the solubility dataset from DataProfessor

import pandas as pd

# dataset / dataframe is pre-labeled
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
#print(df)

# data pre-processing
# separate x and y from df dataframe
# y value will be last value / column
# x value will be second column to the last
# select columns but not including the last
x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
#print(x, y)

# data splitting
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#print(x_train, x_test, y_train, y_test)

# feature scaling (no normalization, just standardization)
# scale features via standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)
print(pd.DataFrame(x_train_std).describe())

# sequential feature selection algorithm
# apply the SBS feature selection implemented by Raschka et al. (2022).
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
    
#import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
#sbs = SBS(knn, k_features=1)
#sbs.fit(x_train_std, y_train)

# plotting erformance of feature subsets
#k_feat = [len(k) for k in sbs.subsets]

#plt.plot(k_feat, sbs.scores_, marker = 'o')
#plt.ylim([0.7, 1.02])
#plt.ylabel('')
#plt.xlabel('')
#plt.grid()
#plt.tight_layout()
#plt.show()

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=500, random_state=1)

forest.fit(x_train, y_train)

# everything except for the last column
feat_labels = df.columns[:-1]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature importance')
plt.bar(range(x_train.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()