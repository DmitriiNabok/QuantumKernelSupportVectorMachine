# External libraries
import numpy as np
import copy

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import algorithm_globals

# SciKit-Learn imports
from sklearn.svm import SVR

# Feature Map builder
from qksvm.QuantumFeatureMap import QuantumFeatureMap


class QKSVR(SVR):
    """
    Extended svm.SVR Scikit-Learn class that supports quantum kernels.
    Can be used in GridSearchCV for optimizing Quantum Feature Map and SVR hyperparameters.

    Args:
        n_qubits (int=1): Number of qubits
        n_layers (int=1): Number of layers (=circuit repetitions)
        feature_map (list=['rx', 'cz']: Quantum feature map structure
        entanglement (str='linear'): Type of 2-qubit communication
        alpha (float=2.0): Data scaling prefactor
        backend (QuantumInstance=None): Qiskit backend instance
        ... : Other parameters from svm.SVR (e.g., C, etc.)

    Returns:
        Scikit-Learn svm.SVR object with the quantum kernel
    """

    def __init__(
        self,
        n_qubits=1,
        n_layers=1,
        feature_map=["RX", "CZ"],
        entanglement="linear",
        alpha=2.0,
        backend=None,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        random_state=None,
    ):

        SVR.__init__(
            self,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.alpha = alpha
        self.entanglement = entanglement
        self.backend = backend
        self.feature_map = feature_map
        self.random_state = random_state

        if self.backend is None:
            algorithm_globals.random_seed = self.random_state
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                seed_simulator=self.random_state,
                seed_transpiler=self.random_state,
                backend_options={
                    "method": "automatic",
                    "max_parallel_threads": 0,
                    "max_parallel_experiments": 0,
                    "max_parallel_shots": 0,
                },
            )

    def fit(self, X, y):

        if isinstance(self.feature_map, list):
            self.fm = QuantumFeatureMap(
                num_features=len(X[0]),
                num_qubits=self.n_qubits,
                num_layers=self.n_layers,
                gates=[s.upper() for s in self.feature_map],
                entanglement=self.entanglement,
                alpha=self.alpha,
                repeat=True,
                scale=False,
            )
        elif isinstance(self.feature_map, QuantumCircuit):
            self.fm = copy.deepcopy(self.feature_map)
            self.fm.assign_parameters({self.fm.alpha: self.alpha}, inplace=True)
        # print(self.fm.draw(plot_barriers=False, fold=120))

        self.kernel = QuantumKernel(self.fm, quantum_instance=self.backend).evaluate
        SVR.fit(self, X, y)
        return self

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
