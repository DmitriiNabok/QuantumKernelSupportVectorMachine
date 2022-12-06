# External libraries
import numpy as np
import copy

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.kernels import QuantumKernel

# from qiskit_machine_learning.kernels import FidelityQuantumKernel # should be used in newer Qiskit releases
from qiskit.utils import algorithm_globals

# SciKit-Learn imports
from sklearn.svm import SVC

# Feature Map builder
from qksvm.QuantumFeatureMap import QuantumFeatureMap


class QKSVC(SVC):
    """
    Extended svm.SVC Scikit-Learn class that supports quantum kernels.
    Can be applied in GridSearchCV for optimizing Quantum Feature Map and SVC hyperparameters.

    Args:
        n_qubits (int=1): Number of qubits
        n_layers (int=1): Number of layers (=circuit repetitions)
        feature_map (list=['rx', 'cz']: Quantum feature map structure
        entanglement (str='linear'): Type of 2-qubit communication
        alpha (float=2.0): Data scaling prefactor
        backend (QuantumInstance=None): Qiskit backend instance
        ... : Other parameters from svm.SVC (e.g., C, class_weight, etc.)

    Returns:
        Scikit-Learn svm.SVC object with the quantum kernel
    """

    def __init__(
        self,
        n_qubits=1,
        n_layers=1,
        feature_map=["RX", "CZ"],
        entanglement="linear",
        alpha=2.0,
        backend=None,
        C=1.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):

        SVC.__init__(
            self,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.alpha = alpha
        self.entanglement = entanglement
        self.backend = backend
        self.feature_map = feature_map

        if self.backend is None:
            np.random.seed(self.random_state)
            algorithm_globals.random_seed = self.random_state
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                shots=1024,
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
        SVC.fit(self, X, y)
        return self

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
