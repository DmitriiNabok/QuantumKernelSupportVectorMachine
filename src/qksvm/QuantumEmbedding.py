import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import XGate, YGate, ZGate


def entangle(fm, gate, connectivity="linear"):
    """2-qubit connectivity generator.
    
    Supports 3 basic connectivity types:
      - "linear": only connecting nearest lying qubits
      - "ring": same as linear + connectivity between the first and last qubits
      - "full": all-to-all connectivity
      
    For testing, "linear_" and "ring_" schemes generate "compact" connectivities, i.e.,
    the 2-qubit gates are applied to the independent qubit pairs simultaneously.
    "full_" generates the all-to-all connectivity by treating each qubit pair only once.
    """
    n = fm.num_qubits
    gate = getattr(fm, gate)
    if connectivity == "full_":
        for i in range(n):
            for ii in range(i + 1, n):
                gate(i, ii)
    elif connectivity == "full":
        for i in range(n):
            for ii in range(n):
                if i != ii:
                    gate(i, ii)
    else:
        if "linear" in connectivity:
            m = n - 1
        elif "ring" in connectivity:
            m = n
        if connectivity[-1] == "_":
            for i in range(0, m, 2):
                gate(i, (i + 1) % n)
            for i in range(1, m, 2):
                gate(i, (i + 1) % n)
        else:
            for i in range(m):
                gate(i, (i + 1) % n)
    fm.barrier()
    return


def ZZ_FeatureMap(n_features, n_qubits, n_layers, data_map_func=None):
    """IQP type H-Z-ZZ feature map.
    
    Same as Qiskit's ZZFeatureMap but extended to be used with any number of qubits version 
    (in the original version n_qubits=n_features).
    """
    fm = QuantumCircuit(n_qubits, name="ZZ_feature_map")

    x = ParameterVector("x", length=n_features)
    alpha = Parameter("α")
    fm.alpha = alpha

    data_map_func = data_map_func or self_product

    j = 0
    for r in range(n_layers):
        for i in range(n_qubits):
            fm.h(i)
            fm.rz(alpha * x[j % n_features], i)
            j += 1
        j -= n_qubits
        for i in range(n_qubits - 1):
            v = data_map_func(
                alpha * x[j % n_features], alpha * x[(j + 1) % n_features]
            )
            j += 1
            fm.rzz(v, i, (i + 1) % n_qubits)
        j += 1
        fm.barrier()

    return fm


def self_product(x, y):
    return x * y


def SingleQubitWall(n_qubits, rotations=None, random_state=None):
    """Prepare a single qubit X,Y,Z rotation wall"""

    qc = QuantumCircuit(n_qubits, name="1q_wall")

    if rotations is None:
        np.random.seed(random_state)
        rotations = np.random.uniform(-2, 2, size=3 * n_qubits)

    j = 0
    for i in range(n_qubits):
        for gate in [XGate(), YGate(), ZGate()]:
            qc.append(gate.power(rotations[j]), [i])
            j += 1

    return qc


def HamiltonianEvolution(n_features, n_qubits, n_trotter=1):
    """Prepare V(x/n_trotter)^n_trotter . U1qb circuit"""

    fm = QuantumCircuit(n_qubits, name="HamiltonianEvolution")

    x = ParameterVector("x", length=n_features)
    alpha = Parameter("α")
    fm.alpha = alpha

    # Add Hamiltonian evolution circuit
    p = (n_qubits / 3) / n_trotter

    for t in range(n_trotter):
        for j in range(max(n_features, n_qubits - 1)):
            fm.rxx(alpha * p * x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
            fm.ryy(alpha * p * x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
            fm.rzz(alpha * p * x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
        fm.barrier()

    return fm

def HamiltonianEvolutionSW(n_features, n_trotter=1, random_state=None):
    """Prepare V(x/n_trotter)^n_trotter . U1qb circuit"""

    n_qubits = n_features+1
    fm = SingleQubitWall(n_qubits, random_state=random_state)

    x = ParameterVector("x", length=n_features)
    alpha = Parameter("α")
    fm.alpha = alpha

    # Add Hamiltonian evolution circuit
    p = (n_qubits / 3) / n_trotter

    for t in range(n_trotter):
        for j in range(max(n_features, n_qubits-1)):
            fm.rxx(alpha*p*x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
            fm.ryy(alpha*p*x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
            fm.rzz(alpha*p*x[j % n_features], j % n_qubits, (j + 1) % n_qubits)
        fm.barrier()

    return fm