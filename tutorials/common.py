def set_figure_params():
    """Set output figure parameters"""
    from matplotlib import pyplot as plt
    
    # plt.style.use('seaborn')
    plt.rcParams.update({
        'xtick.labelsize' : 20,
        'ytick.labelsize' : 20,
        'lines.linewidth': 2,
        'axes.titlesize' : 24,
        'xtick.labelsize' : 16,
        'ytick.labelsize' : 16,
        'lines.markersize' : 10,
    })

    
def generate_moons_dataset(train_size=10, test_size=10, plot=True):
    """Generate and visualize the SKLearn moons dataset"""
    import numpy as np

    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    
    assert train_size > 0, "Illegal number of training points"
    X_train, y_train = make_moons(train_size, noise=0.05, random_state=0)
    assert test_size > 0, "Illegal number of testing points"
    X_test, y_test = make_moons(test_size, noise=0.15, random_state=30)

    # rescaling
    X_train = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(X_train)
    X_test = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(X_test)

    if plot:
        # Visualize respectively the training and testing set
        from matplotlib import pyplot as plt
        set_figure_params()
        fig, ax = plt.subplots(1, 2, figsize=[11, 5])
        ax[0].scatter(X_train[:,0], X_train[:,1], 
                      marker='o', c = plt.cm.coolwarm(np.array(y_train, dtype=np.float32)))
        ax[0].set_title('Train')    
        ax[1].set_title('Test')
        ax[1].scatter(X_test[:,0], X_test[:,1], marker='v', c = plt.cm.coolwarm(np.array(y_test, dtype=np.float32)))
        for i in range(2):
            ax[i].grid(True)
            ax[i].set_xlabel(r"$x_1$", fontsize=24)
            ax[i].set_ylabel(r"$x_2$", fontsize=24, rotation=0)
        plt.tight_layout()
        plt.show()
    
    return X_train, y_train, X_test, y_test
    
    
def visualize_decision_boundaries(clf, X_train, y_train, X_test, y_test):
    """Visualize the decision function, boundary, and margins of +- 0.2"""
    import numpy as np
    
    # Create a 10x10 mesh in the data plan 
    x_min, x_max = X_train[:,0].min(), X_train[:,0].max()
    y_min, y_max = X_train[:,1].min(), X_train[:,1].max()
    margin = 0.2
    XX, YY = np.meshgrid(np.linspace(x_min-margin, x_max+margin, 10), 
                         np.linspace(y_min-margin, y_max+margin, 10))
    
    # Calculate the decision function value on the 10x10 mesh
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z_qke = Z.reshape(XX.shape)
    
    # visualize the decision function and boundary
    from matplotlib import pyplot as plt
    set_figure_params()
    plt.figure(figsize=(7, 7))
    plt.contourf(XX, YY, Z_qke ,vmin=-1., vmax=1., levels=20,
                 cmap=plt.cm.coolwarm, alpha=1)
    plt.scatter(X_train[:,0], X_train[:,1], marker='o', s=200,
                c = plt.cm.coolwarm(np.array(y_train, dtype=np.float32)),
                edgecolor='k')
    plt.scatter(X_test[:,0], X_test[:,1], marker='v', s=200,
                c = plt.cm.coolwarm(np.array(y_test, dtype=np.float32)),
                edgecolor='k')
    plt.contour(XX, YY, Z_qke, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.2, 0, .2])
    plt.xlabel(r"$x_1$", fontsize=24)
    plt.ylabel(r"$x_2$", fontsize=24, rotation=0)
    plt.title("SVC decision boundaries", fontsize=28)
    plt.tight_layout()
    plt.show()
    
    
def print_scores(clf, X_train, y_train, X_test, y_test):
    """Computes and prints Accuracy and ROC_AUC scores for the training and test datasets"""
    from sklearn import metrics

    y_pred = clf.predict(X_train)
    acc_tr = metrics.balanced_accuracy_score(y_train, y_pred)
    roc_tr = metrics.roc_auc_score(y_train, y_pred)

    y_pred = clf.predict(X_test)
    acc_te = metrics.balanced_accuracy_score(y_test, y_pred)
    roc_te = metrics.roc_auc_score(y_test, y_pred)

    print("\nPrediction Scores:\n")
    print("\t\tTrain\tTest")
    print(f"Accuracy:\t{acc_tr:.2f}\t{acc_te:.2f}")
    print(f" ROC_AUC:\t{roc_tr:.2f}\t{roc_te:.2f}")
    print("")