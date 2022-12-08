import pytest
from numpy.testing import assert_allclose

import numpy as np
from qksvm.QuantumFeatureMap import QuantumFeatureMap
from qksvm.QuantumKernelTraining import QuantumKernelTraining
from sklearn.svm import SVC
from qksvm.scores import get_scores
from qksvm.QuantumKernelTraining import TerminationChecker
from qiskit.algorithms.optimizers import SPSA
from qksvm.LossFunctions import SVCLoss
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.utils import algorithm_globals


seed = 12345
np.random.seed(seed=seed)

# Quantum backend
algorithm_globals.random_seed = seed
backend = QuantumInstance(
    AerSimulator(
        method="statevector",
        max_parallel_threads=0,
    ),
    seed_simulator=seed,
    seed_transpiler=seed,
)

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

optimal_value = 2.850241829816053
optimal_point = np.array([2.69474686, 0.38553724, 2.91238291, 2.20079257, 0.54474953])

train_scores = [0.75, 0.7809523809523808, 0.75, 0.6123724356957946]
test_scores = [0.5, 0.2285714285714286, 0.5, 0.0]

n_features = len(X_train[0])
C = 1.0

init_params = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


def test_moons():

    # Specify Feature Map
    fm = QuantumFeatureMap(
        num_features=n_features,
        num_qubits=2,
        num_layers=2,
        gates=["ry", "cz", "RX"],
        entanglement="linear",
        repeat=True,
        scale=True,
    )

    # Choose an optimizer
    optimizer = SPSA(
        maxiter=10,
        learning_rate=None,
        perturbation=None,
        # termination_checker=TerminationChecker(0.001, N=10),
        perturbation_dims=None,
    )

    # Optimization loss function
    loss = SVCLoss(C=C)

    # Apply kernel target alignement
    qkt = QuantumKernelTraining(
        fm,
        X_train,
        y_train,
        init_params,
        optimizer=optimizer,
        loss=loss,
        backend=backend,
        seed=seed,
        plot=False,
    )
    assert qkt.optimal_value == optimal_value
    assert_allclose(actual=qkt.optimal_point, desired=optimal_point)

    # Model training
    qsvc = SVC(
        kernel=qkt.quantum_kernel.evaluate,
        C=C,
        random_state=seed,
    )
    qsvc.fit(X_train, y_train)

    scores = get_scores(qsvc, X_train, y_train)
    assert_allclose(scores, train_scores)

    scores = get_scores(qsvc, X_test, y_test)
    assert_allclose(scores, test_scores)
