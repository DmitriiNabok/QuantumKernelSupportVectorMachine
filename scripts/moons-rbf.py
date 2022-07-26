# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_moons

# Quantum Kernel SVM module imports
from qksvm import scores

# Other modules
import numpy as np
from collections import Counter
seed = 12345
np.random.seed(seed)

####################
# Prepare dataset
####################

# Total number of the dataset points
n_samples = 200

X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)

# Data point rescaling
xmin = -1; xmax = 1
X = MinMaxScaler(feature_range=(xmin, xmax), copy=False).fit_transform(X)
y = 2*y-1 # modify binary labels from (0,1) to (-1,1)

# Train/Test split
train_size = 15 
test_size = 15 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=train_size, 
                                                    test_size=test_size,
                                                    stratify=y,
                                                    random_state=seed)

print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")

##########################
# Model Cross-Validation
##########################
print('\n===== ShuffleSplit Cross-Validation scores ====\n')
print('Train set size: ', train_size)
print('Test set size: ', test_size)

scoring = 0 # 0=acc, 1=f1, 2=roc
best_score = 0.0
scores_tr = []
scores_tt = []

train_size = 10
test_size = 10
n_splits = 10

cv = StratifiedShuffleSplit(
    n_splits=n_splits, 
    train_size=train_size, test_size=test_size, 
    random_state=seed,
)
# cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

param_grid = {
    "gamma": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    "C": [1, 5, 10, 15, 20, 50, 100, 1000],
}
model = SVC(kernel='rbf')

i = 0
for train, test in cv.split(X, y):
    i += 1
    print(f"Train/Test CV Subset {i}/{n_splits}", end='\r')
    
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        n_jobs=4,
        refit=True,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=seed),
    )
    clf.fit(X[train,:], y[train])
    
    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_clf = clf.best_estimator_
    
    scores_tr.append(scores.get_scores(clf.best_estimator_, X[train,:], y[train]))
    scores_tt.append(scores.get_scores(clf.best_estimator_, X[test,:],  y[test]))    

# print("\nBest train score: ", best_score)
scores.print_cv_scores(scores_tr, title='Train set:')
scores.print_cv_scores(scores_tt, title='Test set:')

scores_ = scores.get_scores(best_clf, X, y)
scores.print_scores(scores_, title='Best Model / Entire set:')
