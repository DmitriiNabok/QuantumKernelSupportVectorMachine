import pytest

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qksvm.QKSVC import QKSVC

seed = 12345

# Test dataset
X = [[0.0, 0.0], [1.0, 1.0]]
y = [-1.0, 1.0]

# QKSVM hyperparameters
n_features = len(X[0])
n_qubits = 2
n_layers = 2
alpha = 2.0
C = 1.0


def test_ini_1():
    """Initialization with an explicit feature map"""

    fm = QuantumFeatureMap(
        num_features=n_features,
        num_qubits=n_qubits,
        num_layers=n_layers,
        gates=["H", "RZ", "CZ"],
        entanglement="linear",
    )

    # initialize the QKSVM object
    qsvc = QKSVC(feature_map=fm, alpha=alpha, C=C, random_state=seed)

    assert isinstance(qsvc.feature_map, QuantumCircuit)
    assert qsvc.feature_map.num_features == n_features
    assert qsvc.feature_map.num_qubits == n_qubits
    assert qsvc.feature_map.num_layers == n_layers
    assert qsvc.alpha == alpha
    assert qsvc.C == C
    assert qsvc.random_state == seed
    assert isinstance(qsvc.backend, QuantumInstance)


def test_ini_2():
    """Initialization with an implicit feature map"""

    qsvc = QKSVC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["H", "RZ", "CZ"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    assert qsvc.n_qubits == n_qubits
    assert qsvc.n_layers == n_layers
    assert qsvc.alpha == alpha
    assert qsvc.C == C
    assert isinstance(qsvc.feature_map, list)
    assert isinstance(qsvc.backend, QuantumInstance)
    assert qsvc.random_state == seed


def test_kernel():
    qsvc = QKSVC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["H", "RZ", "CZ"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    qsvc.fit(X, y)

    x1 = [0.0, 0.0]
    x2 = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]
    k12 = [1.0, 0.98016591, 0.92261884, 0.83296253, 0.71970341, 0.5931328]
    for i, xi in enumerate(x2):
        print(i, xi)
        assert pytest.approx(qsvc.kernel(x1, xi), 1e-8) == k12[i]


def test_predict():
    qsvc = QKSVC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=["H", "RZ", "CZ"],
        alpha=alpha,
        C=C,
        random_state=seed,
    )
    qsvc.fit(X, y)

    x_test = [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]]
    y_test = [-1, -1, -1, 1, 1, 1]
    for i, xi in enumerate(x_test):
        assert qsvc.predict(xi) == y_test[i]
