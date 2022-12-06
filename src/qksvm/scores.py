# ------------------------------------------------------------
# Collection of custom routines to compute the model scores
# ------------------------------------------------------------

import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn import metrics


def get_scores(model, X, y):
    y_pred = model.predict(X)
    acc = metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred)
    f1 = metrics.f1_score(y_true=y, y_pred=y_pred, average="weighted")
    labels = np.unique(y)
    if len(labels) > 2:
        roc = metrics.roc_auc_score(y, model.predict_proba(X), multi_class="ovr")
    else:
        roc = metrics.roc_auc_score(y, y_pred, average="weighted")
    mcc = metrics.matthews_corrcoef(y_true=y, y_pred=y_pred)
    return [acc, f1, roc, mcc]


def print_scores(scores, title=None):
    if title is not None:
        print("")
        print(title)
    print(f"{'  Balanced accuracy: ':>22} {scores[0]:>.2f}")
    print(f"{'  F1: ':>22} {scores[1]:>.2f}")
    print(f"{'  ROC_AUC: ':>22} {scores[2]:>.2f}")
    print(f"{'  MCC: ':>22} {scores[3]:>.2f}")


def print_cv_scores(scores, title=None):
    if title is not None:
        print("")
        print(title)
    scores_ = np.array(scores)
    print(
        f"{'  Balanced accuracy: ':>22} {np.mean(scores_[:,0]):>.2f} +- {np.std(scores_[:,0]):>.2f}"
    )
    print(f"{'  F1: ':>22} {np.mean(scores_[:,1]):>.2f} +- {np.std(scores_[:,1]):>.2f}")
    print(
        f"{'  ROC_AUC: ':>22} {np.mean(scores_[:,2]):>.2f} +- {np.std(scores_[:,2]):>.2f}"
    )
    print(
        f"{'  MCC: ':>22} {np.mean(scores_[:,3]):>.2f} +- {np.std(scores_[:,3]):>.2f}"
    )


def print_cross_validate_scores(scores, title=None):
    """Pretty print of scores computed with sklearn.cross_validate."""
    if title:
        print(title)
    for key in scores.keys():
        if key[:4] == "test":
            print(
                f"{key[5:]:>22}:  {np.mean(scores[key]):>.2f} +- {np.std(scores[key]):>.2f}"
            )
    return


def grid_search_cv(
    estimator, param_grid, X, y, train_size=0.8, n_splits=5, test_size=0.2, seed=None
):
    """
    Wrapper around the sklearn.GridSearchCV to find the model hyperparameters and compute the classification scores.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        n_jobs=1,
        refit=True,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
    )
    gs.fit(X_train, y_train)

    print("Best grid search parameters:", gs.best_params_)
    print("Best training score:", gs.best_score_)
    clf = gs.best_estimator_

    train_scores = get_scores(clf, X_train, y_train)
    print_scores(train_scores, title="Train scores:")
    y_pred = clf.predict(X_train)
    print(metrics.classification_report(y_true=y_train, y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=y_train, y_pred=y_pred))

    test_scores = get_scores(clf, X_test, y_test)
    print_scores(test_scores, title="Test scores:")
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

    # cross-validation scores for the best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf,
        X,
        y,
        scoring=[
            "balanced_accuracy",
            "f1_weighted",
            "roc_auc_ovr",
            "matthews_corrcoef",
        ],
        n_jobs=1,
        verbose=0,
        cv=cv,
    )
    print_cross_validate_scores(scores, title="\nCross-validation scores:")

    # --------------------------
    # inspect other solutions
    # --------------------------

    # print("\nModels ranking (10 first best models):")
    # print("Best model index: ", gs.best_index_)
    # for i in np.arange(1, 11):
    #     idx = np.argsort(gs.cv_results_['mean_test_score'])
    #     j = idx[-i]
    #     print('Model', j)
    #     print(gs.cv_results_['mean_test_score'][j], ' +- ',  gs.cv_results_['std_test_score'][j])
    #     print('params', gs.cv_results_['params'][j])

    # v = gs.cv_results_['mean_test_score']
    # s = gs.cv_results_['std_test_score']
    # idxs = np.where((v > 0.98) & (s < 0.4))[0]
    # # print(idxs)
    # for i in idxs:
    #     print(i, v[i], s[i])
    #     print(gs.cv_results_['params'][i])

    return gs


def cross_validate_split(model, X, y, train_size=0.8, test_size=0.2, seed=None):
    """
    Computes the classification scores by taking different train/test combinations.
    """
    np.random.seed(seed)

    scores_tr = []
    scores_tt = []

    n_splits = 5

    for _seed in np.random.randint(2**16 - 1, size=5):

        cv = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            random_state=_seed,
        )

        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            train_scores = get_scores(model, X[train], y[train])
            test_scores = get_scores(model, X[test], y[test])
            scores_tr.append(train_scores)
            scores_tt.append(test_scores)

    print("")
    print("==== StratifiedShuffleSplit Scores ====")
    print_cv_scores(scores_tr, title="Train set:")
    print_cv_scores(scores_tt, title="Test set:")
    return
