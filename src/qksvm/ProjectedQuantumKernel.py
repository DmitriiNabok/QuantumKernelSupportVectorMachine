import numpy as np
from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit.quantum_info import Operator, Pauli
from qiskit.opflow import AerPauliExpectation, StateFn, PauliSumOp, ListOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers.aer import AerSimulator


class ProjectedQuantumKernel:
    """Projected Quantum Kernel from Huang et al., "Power of Data in Quantum Machine Learning", 
    https://doi.org/10.1038/s41467-021-22539-9

    Args:
        fm (QuantumCircuit): Quantum feature map circuit
        gamma (float=1.0): Projected kernel parameter
        projection (str="xyz_sum"): Projection type ("x", "y", "z", "xyz_sum", and "xyz")
        method (str="opflow"): Methods to compute the projections: "opflow" - using Qiskit `OpflowQNN` tools;
            "statevector" - statevector based implementation (very fast but memory intensive).
        backend (QuantumInstance): Qiskit quantum backend
        random_state (int=None): Random generator seed
        
    Returns:
        ProjectedQuantumKernel: self.evaluate() is used for the kernel matrix setup
    """
    def __init__(
        self,
        fm,
        gamma=1.0,
        projection="xyz_sum",
        method="opflow",
        backend=None,
        random_state=None,
    ):
        self.fm = fm
        self.num_qubits = self.fm.num_qubits
        self.gamma = gamma
        self.projection = projection
        self.seed = random_state
        self.method = method
        # quantum backend
        self.backend = backend
        if self.backend is None:
            backend_options = {
                "max_parallel_threads": 0,
                "max_parallel_experiments": 1,
                "max_memory_mb": 0,
                "statevector_sample_measure_opt": 10,
            }
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                shots=1024,
                backend_options=backend_options,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            )
        # generate projection operators
        if isinstance(self.projection, str):
            if self.projection == "x":
                self.proj_ops = self._measurement_operator(["X"])
            elif self.projection == "y":
                self.proj_ops = self._measurement_operator(["Y"])
            elif self.projection == "z":
                self.proj_ops = self._measurement_operator(["Z"])
            elif self.projection == "xyz":
                self.proj_ops = self._measurement_operator(["X", "Y", "Z"])
            elif self.projection == "xyz_sum":
                self.proj_ops = []
                for x, y, z in zip(
                    self._measurement_operator(["X"]),
                    self._measurement_operator(["Y"]),
                    self._measurement_operator(["Z"]),
                ):
                    self.proj_ops.append([x[0], y[0], z[0]])
        else:
            self.proj_ops = self.projection

    def _measurement_operator(self, gates):
        s = "I" * self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in gates:
                ss = list(s)
                ss[-i - 1] = gate
                op.append([("".join(ss), 1.0)])
        return op

    def projected_feature_map(self, x):
        """ """
        if self.method == "statevector":
            qc = self.fm.assign_parameters(x)
            result = self.backend.execute(qc)
            sv = result.get_statevector()
            ev = []
            for op in self.proj_ops:
                o = Operator(Pauli(op[0][0]))
                if len(op) > 1:
                    for i in range(1, len(op)):
                        o = o + Operator(Pauli(op[i][0]))
                ev.append(np.real(sv.expectation_value(o)))
        elif self.method == "opflow":
            fm_sfn = StateFn(self.fm)
            list_ops = []
            for op in self.proj_ops:
                op = ~StateFn(PauliSumOp.from_list(op)) @ fm_sfn
                list_ops.append(op)
            list_ops = ListOp(list_ops)
            expval = AerPauliExpectation()
            qnn = OpflowQNN(
                list_ops,
                input_params=self.fm.parameters,
                weight_params=[],
                exp_val=expval,
                gradient=None,
                quantum_instance=self.backend,
            )
            ev = qnn.forward(x, []).flatten()
        else:
            assert False, "Unknown method!"
        return ev

    def evaluate(self, X_1, X_2=None):
        """ """
        if len(X_1.shape) == 1:
            X_1_proj = np.array([self.projected_feature_map(X_1)])
        else:
            X_1_proj = np.array([self.projected_feature_map(x) for x in X_1])
        if X_2 is None:
            X_2_proj = X_1_proj
        else:
            if len(X_2.shape) == 1:
                X_2_proj = np.array([self.projected_feature_map(X_2)])
            else:
                X_2_proj = np.array([self.projected_feature_map(x) for x in X_2])

        kernel = np.zeros(shape=(X_1_proj.shape[0], X_2_proj.shape[0]))
        for i in range(X_1_proj.shape[0]):
            for j in range(X_2_proj.shape[0]):
                value = np.exp(-self.gamma * ((X_1_proj[i] - X_2_proj[j]) ** 2).sum())
                kernel[i, j] = value
        return kernel
