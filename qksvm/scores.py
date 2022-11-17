import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV

def get_scores(model, X, y, average='weighted'):
    y_pred = model.predict(X)
    from sklearn import metrics
    acc = metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred)
    f1  = metrics.f1_score(y_true=y, y_pred=y_pred, average=average)
    roc = metrics.roc_auc_score(y, model.decision_function(X), average=average)
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
    from numpy import array, mean, std
    if title is not None:
        print("")
        print(title)
    scores_ = array(scores)
    print(f"{'  Balanced accuracy: ':>22} {mean(scores_[:,0]):>.2f} +- {std(scores_[:,0]):>.2f}")
    print(f"{'  F1: ':>22} {mean(scores_[:,1]):>.2f} +- {std(scores_[:,1]):>.2f}")
    print(f"{'  ROC_AUC: ':>22} {mean(scores_[:,2]):>.2f} +- {std(scores_[:,2]):>.2f}")
    print(f"{'  MCC: ':>22} {mean(scores_[:,3]):>.2f} +- {std(scores_[:,3]):>.2f}")


def grid_search_cv(estimator, param_grid, X, y, train_size=0.8, test_size=0.2, seed=None):
    """ Wrapper for the GridSearchCV routine. """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size, test_size=test_size,
        stratify=y, random_state=seed,
    )

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        n_jobs=1,
        refit=True,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
    )
    gs.fit(X_train, y_train)

    print('Best grid search parameters:', gs.best_params_)
    print('Best training score:', gs.best_score_)
    clf = gs.best_estimator_

    train_scores = get_scores(clf, X_train, y_train)
    print_scores(train_scores, title='Train scores:')
    y_pred = clf.predict(X_train)
    print(metrics.classification_report(y_true=y_train, y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=y_train, y_pred=y_pred))

    test_scores  = get_scores(clf, X_test,  y_test)
    print_scores(test_scores, title='Test scores:')
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    
    # cross-validation scores for the best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf, X, y, scoring=["balanced_accuracy", "f1_macro", "matthews_corrcoef"], n_jobs=1, verbose=0, cv=cv,
    )
    print("\nBest model cross-calidation scores:")
    print('Accuracy: {:.2f} +- {:.2f}'.format(
        np.mean(scores['test_balanced_accuracy']), 
        np.std(scores['test_balanced_accuracy']), )
    )
    print('      F1: {:.2f} +- {:.2f}'.format(
        np.mean(scores['test_f1_macro']), 
        np.std(scores['test_f1_macro']), )
    )
    print('     MCC: {:.2f} +- {:.2f}'.format(
        np.mean(scores['test_matthews_corrcoef']), 
        np.std(scores['test_matthews_corrcoef']), )
    )
    
    # inspect other solutions
    print("\nModels ranking:")
    for i in np.arange(2, 8):
        idx = np.argsort(gs.cv_results_['mean_test_score'])
        j = idx[-i]
        print('Model', j)
        print(gs.cv_results_['mean_test_score'][j], ' +- ',  gs.cv_results_['std_test_score'][j])
        print('params', gs.cv_results_['params'][j])
    
    # v = gs.cv_results_['mean_test_score']
    # s = gs.cv_results_['std_test_score']
    # idxs = np.where((v > 0.98) & (s < 0.4))[0]
    # # print(idxs)
    # for i in idxs:
    #     print(i, v[i], s[i])
    #     print(gs.cv_results_['params'][i])
    
    return clf


def cross_validate_split(model, X, y, train_size=0.8, test_size=0.2, seed=None):
    """ Custom version of the model cross-validation procedure. """
    np.random.seed(seed)

    scores_tr = []
    scores_tt = []

    n_splits = 5
    
    for _seed in np.random.randint(2**16-1, size=5):

        cv = StratifiedShuffleSplit(
            n_splits=n_splits, 
            train_size=train_size, test_size=test_size, 
            random_state=_seed
        )

        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            train_scores = get_scores(model, X[train], y[train])
            test_scores = get_scores(model, X[test], y[test])
            scores_tr.append(train_scores)
            scores_tt.append(test_scores)

    print('')
    print('==== CV.SPLIT Cross-Validation Scores ====')
    print_cv_scores(scores_tr, title='Train set:')
    print_cv_scores(scores_tt, title='Test set:')