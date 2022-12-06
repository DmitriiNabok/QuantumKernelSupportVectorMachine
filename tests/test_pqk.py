import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose

from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qksvm.ProjectedQuantumKernel import ProjectedQuantumKernel
from qiskit.utils import QuantumInstance

seed = 12345

n_features = 2
n_qubits = 2
n_layers = 1

fm = QuantumFeatureMap(
    num_features=n_features,
    num_qubits=n_qubits,
    num_layers=n_layers,
    gates=["RX", "CZ"],
    entanglement="linear",
)
fm = fm.assign_parameters({fm.alpha: 2.0})

X_train = np.array(
    [
        [-1.00000000, -0.38801645],
        [0.15313461, 0.61126768],
        [-0.24455475, -0.45724891],
        [-0.42649158, 0.36185767],
        [-0.85614803, 0.41409787],
    ]
)
y_train = np.array([-1, -1, 1, 1, -1])

X_test = np.array(
    [
        [0.25870690, -0.06222924],
        [0.30934215, -1.00000000],
        [-0.33350839, 1.00000000],
        [1.00000000, 0.34463767],
        [0.72295777, -0.42731420],
    ]
)
y_test = np.array([-1, 1, -1, 1, 1])

px_train = np.array(
    [
        0.23281684,
        0.42220636,
        0.85057282,
        -0.55496256,
        1.16944256,
        1.30955399,
        1.22218437,
        0.31381347,
        0.52841979,
        0.78010319,
    ]
)

px_test = np.array(
    [
        0.37829316,
        1.10015303,
        1.05599234,
        0.32460524,
        0.52822359,
        -1.13055519,
        -1.1178581,
        1.03636737,
        -0.52682994,
        0.75045516,
    ]
)

pkernel_train = np.array(
    [
        [1.00000000e00, 6.90473277e-02, 3.58179138e-02, 1.37904449e-01, 6.49896283e-01],
        [6.90473277e-02, 1.00000000e00, 7.80013140e-04, 1.67673965e-01, 2.29975804e-02],
        [3.58179138e-02, 7.80013140e-04, 1.00000000e00, 1.36892166e-01, 2.50961263e-01],
        [1.37904449e-01, 1.67673965e-01, 1.36892166e-01, 1.00000000e00, 2.47221454e-01],
        [6.49896283e-01, 2.29975804e-02, 2.50961263e-01, 2.47221454e-01, 1.00000000e00],
    ]
)

pkernel_test = np.array(
    [
        [1.00000000e00, 1.19851584e-01, 4.55328448e-05, 1.12760482e-02, 1.52121936e-01],
        [1.19851584e-01, 1.00000000e00, 8.29532257e-03, 2.85326322e-05, 4.63857046e-03],
        [4.55328448e-05, 8.29532257e-03, 1.00000000e00, 3.69809927e-07, 9.11799255e-05],
        [1.12760482e-02, 2.85326322e-05, 3.69809927e-07, 1.00000000e00, 4.22265862e-01],
        [1.52121936e-01, 4.63857046e-03, 9.11799255e-05, 4.22265862e-01, 1.00000000e00],
    ]
)


def test_init():
    """General initialization test"""
    pqk = ProjectedQuantumKernel(
        fm,
        gamma=2.0,
        projection="xyz",
        method="statevector",
        backend=None,
        random_state=seed,
    )

    assert pqk.fm.num_features == n_features
    assert pqk.fm.num_qubits == n_qubits
    assert pqk.fm.num_layers == n_layers
    assert_almost_equal(pqk.gamma, 2.0)
    assert pqk.projection == "xyz"
    assert pqk.method == "statevector"
    assert pqk.seed == seed
    assert isinstance(pqk.backend, QuantumInstance)


def test_proj_ops():
    """Test projection operators generator"""
    # X
    pqk = ProjectedQuantumKernel(fm, projection="x")
    assert pqk.proj_ops == [[("IX", 1.0)], [("XI", 1.0)]]
    # Y
    pqk = ProjectedQuantumKernel(fm, projection="y")
    assert pqk.proj_ops == [[("IY", 1.0)], [("YI", 1.0)]]
    # Z
    pqk = ProjectedQuantumKernel(fm, projection="z")
    assert pqk.proj_ops == [[("IZ", 1.0)], [("ZI", 1.0)]]
    # X+Y+Z
    pqk = ProjectedQuantumKernel(fm, projection="xyz_sum")
    assert pqk.proj_ops == [
        [("IX", 1.0), ("IY", 1.0), ("IZ", 1.0)],
        [("XI", 1.0), ("YI", 1.0), ("ZI", 1.0)],
    ]
    # X, Y, Z
    pqk = ProjectedQuantumKernel(fm, projection="xyz")
    assert pqk.proj_ops == [
        [("IX", 1.0)],
        [("IY", 1.0)],
        [("IZ", 1.0)],
        [("XI", 1.0)],
        [("YI", 1.0)],
        [("ZI", 1.0)],
    ]


def test_projected_feature_map():
    """Custom (xyz_sum) projected feature map check"""
    pqk = ProjectedQuantumKernel(fm, gamma=2.0, projection="xyz_sum", random_state=seed)
    # train set
    _x = pqk.projected_feature_map(X_train)
    assert_allclose(_x, px_train)
    # test set
    _x = pqk.projected_feature_map(X_test)
    assert_allclose(_x, px_test)


def test_evaluate():
    """Projected kernel matrix evaluator"""
    pqk = ProjectedQuantumKernel(fm, gamma=2.0, projection="xyz_sum", random_state=seed)
    # train set
    _kernel = pqk.evaluate(X_train)
    assert_allclose(_kernel, pkernel_train)
    # test set
    _kernel = pqk.evaluate(X_test)
    assert_allclose(_kernel, pkernel_test)
