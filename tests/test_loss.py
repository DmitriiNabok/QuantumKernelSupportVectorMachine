import pytest
from numpy.testing import assert_allclose

import numpy as np
from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qksvm.LossFunctions import SVCLoss, KTALoss
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel


seed = 12345
np.random.seed(seed=seed)

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

# Quantum backend
algorithm_globals.random_seed = seed
backend = QuantumInstance(
    AerSimulator(method="statevector"),
    seed_simulator=seed,
    seed_transpiler=seed,
)

# Specify Feature Map
fm = QuantumFeatureMap(
    num_features=len(X_train[0]),
    num_qubits=2,
    num_layers=2,
    gates=["ry", "cz", "RX"],
    entanglement="linear",
    repeat=True,
    scale=True,
)

# Quantum kernel
quantum_kernel = QuantumKernel(
    fm, user_parameters=fm.train_params, quantum_instance=backend
)

init_params = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

ref_svcloss = np.array(
    [
        4.0,
        3.98301873,
        3.89871795,
        3.70478986,
        3.46732802,
        3.27837335,
        3.1591091,
        3.13680101,
        3.20100086,
        3.26514626,
        3.30179542,
        3.33723478,
        3.36188814,
        3.23097761,
        2.943778,
        2.63777912,
        2.56215205,
        2.78038396,
        2.98194134,
        3.18146218,
    ]
)

ref_ktaloss = np.array(
    [
        -1.33226763e-17,
        -6.31407399e-03,
        -2.70756064e-02,
        -6.42821421e-02,
        -1.12109192e-01,
        -1.57554413e-01,
        -1.90789822e-01,
        -2.06556157e-01,
        -2.02276369e-01,
        -1.80299287e-01,
        -1.51183692e-01,
        -1.32573525e-01,
        -1.37426301e-01,
        -1.65305931e-01,
        -2.05644091e-01,
        -2.44013233e-01,
        -2.65133118e-01,
        -2.59314770e-01,
        -2.29692785e-01,
        -1.87650137e-01,
    ]
)


def test_SVCLoss():
    """Testing the SVCLoss"""
    loss = SVCLoss(C=1.0)
    values = loss.plot(
        quantum_kernel, X_train, y_train, init_params, grid=[0, 8, 20], show=False
    )
    assert_allclose(values, ref_svcloss)


def test_KTALoss():
    """Testing the KTALoss"""
    loss = KTALoss()
    values = loss.plot(
        quantum_kernel, X_train, y_train, init_params, grid=[0, 8, 20], show=False
    )
    assert_allclose(values, ref_ktaloss)
