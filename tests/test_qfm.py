import pytest
from numpy.testing import assert_allclose

from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression

n_features = 2
n_qubits = 2
n_layers = 2
repeat = True
scale = False


def test_init():
    """Initialization test"""

    fm = QuantumFeatureMap(
        num_features=n_features,
        num_qubits=n_qubits,
        num_layers=n_layers,
        gates=["ry", "cz", "RX"],
        entanglement="linear",
        repeat=repeat,
        scale=scale,
    )

    assert isinstance(fm, QuantumCircuit)
    assert fm.num_features == n_features
    assert fm.num_qubits == n_qubits
    assert fm.num_layers == n_layers
    assert fm.repeat == repeat
    assert fm.scale == scale
    assert isinstance(fm.alpha, Parameter)
    assert_allclose(fm.entanglement, [[0, 1]])
    for item in fm.encod_params:
        assert isinstance(item, ParameterExpression)
    assert len(fm.encod_params) == 2
    assert isinstance(fm.train_params, ParameterVector)
    assert len(fm.train_params) == 4


def test_setup_parameters():
    """Test initialization of the encoding and training parameters"""

    fm = QuantumFeatureMap(num_features=1, num_qubits=1, num_layers=1, gates=["RX"])
    assert len(fm.encod_params) == 1
    fm = QuantumFeatureMap(num_features=10, num_qubits=1, num_layers=1, gates=["RX"])
    assert len(fm.encod_params) == 10

    fm = QuantumFeatureMap(
        num_features=1, num_qubits=1, num_layers=1, gates=["RX", "ry"]
    )
    assert len(fm.train_params) == 1
    fm = QuantumFeatureMap(
        num_features=1, num_qubits=1, num_layers=10, gates=["RX", "ry"]
    )
    assert len(fm.train_params) == 10
    fm = QuantumFeatureMap(
        num_features=1, num_qubits=10, num_layers=1, gates=["RX", "ry"]
    )
    assert len(fm.train_params) == 10
    fm = QuantumFeatureMap(
        num_features=1, num_qubits=2, num_layers=2, gates=["RX", "ry", "crz"]
    )
    assert len(fm.train_params) == 6
    fm = QuantumFeatureMap(
        num_features=1,
        num_qubits=2,
        num_layers=2,
        gates=["RX", "ry", "crz", "rx", "rzz"],
    )
    assert len(fm.train_params) == 12


def test_generate_map():
    """Test the entanglement map generator"""
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="linear",
    )
    assert_allclose(fm.entanglement, [[0, 1], [1, 2], [2, 3]])
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="linear_",
    )
    assert_allclose(fm.entanglement, [[0, 1], [2, 3], [1, 2]])
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="ring",
    )
    assert_allclose(fm.entanglement, [[0, 1], [1, 2], [2, 3], [3, 0]])
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="ring_",
    )
    assert_allclose(fm.entanglement, [[0, 1], [2, 3], [1, 2], [3, 0]])
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="full",
    )
    assert_allclose(
        fm.entanglement,
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ],
    )
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RX", "CX"],
        entanglement="full_",
    )
    assert_allclose(fm.entanglement, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def build_circuit():
    """Test for the build_circuit method"""
    fm = QuantumFeatureMap(
        num_features=2, num_qubits=2, num_layers=1, gates=[], entanglement="linear"
    )
    # Test 1
    _fm = fm.copy()
    j = _fm.build_circuit("rx", fm.encod_params, j0=0)
    assert j == 2
    ops = _fm.count_ops()
    assert ops["rx"] == 2
    assert ops["barrier"] == 1
    j = _fm.build_circuit("rx", fm.encod_params, j0=1)
    assert j == 3
    ops = _fm.count_ops()
    assert ops["rx"] == 4
    assert ops["barrier"] == 2
    # Test 2
    _fm = fm.copy()
    # +H
    j = _fm.build_circuit("h", fm.encod_params)
    assert j == 0
    ops = _fm.count_ops()
    assert ops["h"] == 2
    # +RZ
    j = _fm.build_circuit("rz", fm.encod_params)
    assert j == 2
    ops = _fm.count_ops()
    assert (ops["h"] == 2) and (ops["rz"] == 2)
    # + CX
    j = _fm.build_circuit("cx", fm.encod_params)
    assert j == 0
    ops = _fm.count_ops()
    assert (ops["h"] == 2) and (ops["rz"] == 2) and (ops["cx"] == 2)
    # + RZZ
    j = _fm.build_circuit("rzz", fm.encod_params)
    assert j == 2
    ops = _fm.count_ops()
    assert (
        (ops["h"] == 2) and (ops["rz"] == 2) and (ops["cx"] == 2) and (ops["rzz"] == 2)
    )
    # + X
    j = _fm.build_circuit("x", fm.encod_params)
    assert j == 0
    ops = _fm.count_ops()
    assert (
        (ops["h"] == 2)
        and (ops["rz"] == 2)
        and (ops["cx"] == 2)
        and (ops["rzz"] == 2)
        and (ops["x"] == 2)
    )


def test_repeat():
    """Test the repeated encoding scheme for the case n_qubits > n_features"""
    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RY"],
        entanglement="linear",
        repeat=True,
    )
    ops = fm.count_ops()
    assert ops["ry"] == 4

    fm = QuantumFeatureMap(
        num_features=2,
        num_qubits=4,
        num_layers=1,
        gates=["RY"],
        entanglement="linear",
        repeat=False,
    )
    ops = fm.count_ops()
    assert ops["ry"] == 2
