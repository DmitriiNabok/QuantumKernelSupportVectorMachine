import pytest

import numpy as np
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qksvm.QKSVR import QKSVR
from sklearn import metrics

seed = 12345

# Test dataset
X = [[0.0, 0.0], [1.0, 1.0]]
y = [-1.0, 1.0]

X_train = np.array(
    [
        [2.04318707],
        [6.09207989],
        [4.78015344],
        [6.2007935],
        [3.28442723],
        [2.2523066],
        [0.03469649],
        [0.98014248],
        [0.76678884],
        [6.09412333],
        [0.29185655],
        [0.39934889],
        [2.07909361],
        [5.71342859],
        [3.72225051],
        [1.83559896],
        [2.30191935],
        [0.75143281],
        [4.58425082],
        [3.84438512],
    ]
)
y_train = np.array(
    [
        1.83077283,
        0.44080133,
        -0.99770487,
        -0.08229862,
        -0.1423494,
        0.77662218,
        0.03468952,
        0.42134549,
        1.50743183,
        -0.18793767,
        0.28773076,
        0.38881855,
        0.87357447,
        -0.53942721,
        -0.54857409,
        0.96514417,
        0.74442502,
        -0.49499057,
        -0.99180153,
        -1.86905039,
    ]
)

X_test = np.array(
    [
        [0.61369199],
        [5.20711134],
        [1.25458737],
        [3.11128829],
        [5.07931034],
        [4.93340606],
        [1.91394476],
        [5.96202367],
        [0.46523627],
        [0.87646578],
    ]
)
y_test = np.array(
    [
        0.57588968,
        -0.88010056,
        0.95042113,
        0.03029973,
        -1.70008963,
        -0.22355167,
        0.94170003,
        -0.31566902,
        0.44863397,
        0.76848223,
    ]
)

# QKSVM hyperparameters
n_features = len(X[0])
n_qubits = 1
n_layers = 1
alpha = 1.0
C = 1.0


def test_ini_1():
    """Initialization with an explicit feature map"""

    fm = QuantumFeatureMap(
        num_features=n_features,
        num_qubits=n_qubits,
        num_layers=n_layers,
        gates=["RY", "CX"],
        entanglement="linear",
    )

    # initialize the QKSVM object
    qsvr = QKSVR(feature_map=fm, alpha=alpha, C=C, random_state=seed)

    assert isinstance(qsvr.feature_map, QuantumCircuit)
    assert qsvr.feature_map.num_features == n_features
    assert qsvr.feature_map.num_qubits == n_qubits
    assert qsvr.feature_map.num_layers == n_layers
    assert qsvr.alpha == alpha
    assert qsvr.C == C
    assert qsvr.random_state == seed
    assert isinstance(qsvr.backend, QuantumInstance)


def test_ini_2():
    """Initialization with an implicit feature map"""

    qsvr = QKSVR(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["RY", "CX"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    assert qsvr.n_qubits == n_qubits
    assert qsvr.n_layers == n_layers
    assert qsvr.alpha == alpha
    assert qsvr.C == C
    assert isinstance(qsvr.feature_map, list)
    assert qsvr.random_state == seed
    assert isinstance(qsvr.backend, QuantumInstance)


def test_kernel():
    qsvr = QKSVR(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["RY", "CX"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    qsvr.fit(X_train, y_train)

    x1 = np.array([[0.0]])
    x2 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    print(x1.shape, x2.shape)
    k12 = [
        [1.0, 0.77015115, 0.29192658, 0.00500375, 0.17317819, 0.64183109, 0.98008514]
    ]

    k12_ = qsvr.kernel(x1, x2)
    print(k12_)
    assert_allclose(k12_, k12, rtol=1e-6)


def test_predict():
    qsvr = QKSVR(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["RY", "CX"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    qsvr.fit(X_train, y_train)

    y_pred_ref = np.array([ 0.573004, -0.768857,  0.886231, -0.020526, -0.823571, -0.869459,
               0.843527, -0.230341,  0.461425,  0.738521])
    y_pred = qsvr.predict(X_test)
    assert_allclose(y_pred, y_pred_ref, rtol=1e-4)


def compute_scores(y_true, y_pred):
    return (
        metrics.r2_score(y_true, y_pred),
        metrics.mean_squared_error(y_true, y_pred),
        metrics.median_absolute_error(y_true, y_pred),
    )


def test_scores():
    qsvr = QKSVR(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["RY", "CX"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    qsvr.fit(X_train, y_train)

    train_scores = np.array([0.656959, 0.26007, 0.09846])
    scores = compute_scores(y_train, qsvr.predict(X_train))
    assert_allclose(scores, train_scores, rtol=1e-4)

    test_scores = [0.817352, 0.122255, 0.074759]
    scores = compute_scores(y_test, qsvr.predict(X_test))
    assert_allclose(scores, test_scores, rtol=1e-4)


if __name__ == "__main__":
    test_kernel()