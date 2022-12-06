#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Global imports

# %%
import pandas as pd
import numpy as np
from collections import Counter

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Wrapper around sklearn.GridSearchCV
from qksvm.scores import grid_search_cv

seed = 12345
np.random.seed(seed)

# %% [markdown]
# # Dataset

# %%
# Load and rescale data
X, y = load_iris(return_X_y=True)

xmin, xmax = -1, 1
X = MinMaxScaler(feature_range=(xmin, xmax)).fit_transform(X)

print('X.shape:', X.shape)
print('Data balance: ', Counter(y))


# %% [markdown]
# # RBF-SVM

# %%
# Perform the model hyperparameter search and compute the classification scores
param_grid = {
    "gamma": [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0, 10.0],
    "C": [1, 2, 4, 6, 8, 10, 100],
}

gs = grid_search_cv(
    SVC(kernel="rbf", random_state=seed, probability=True, decision_function_shape='ovr'),
    param_grid,
    X,
    y,
    n_splits=5,
    train_size=0.8,
    test_size=0.2,
    seed=seed,
)

# %% [raw]
# # Inspect other solutions
# v = gs.cv_results_['mean_test_score']
# s = gs.cv_results_['std_test_score']
# idxs = np.where((v > 0.97) & (s < 0.03))[0]
# print(idxs)
# for i in idxs:
#     print(i, v[i], s[i])
#     print(gs.cv_results_['params'][i])

# %%
