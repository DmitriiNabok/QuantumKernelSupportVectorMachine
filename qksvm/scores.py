
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