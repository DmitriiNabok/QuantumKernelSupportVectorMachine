import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.opflow import CircuitSampler, AerPauliExpectation, StateFn, PauliSumOp, ListOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer

class ProjectedQuantumKernel():
    
    def __init__(self, fm, gamma=1.0, projection='xyz_sum', backend=None, random_state=None):
        """ """
        self.fm = fm
        self.num_qubits = self.fm.num_qubits
        self.gamma = gamma
        self.projection = projection
        self.backend = backend
        self.seed = random_state
        self.proj_ops = []
        
        if self.backend is None:
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

    def _o_xyz_sum(self):
        s = 'I'*self.num_qubits
        op = []
        for i in range(self.num_qubits):
            op_s = []
            for gate in ['X', 'Y', 'Z']:
                ss = list(s)
                ss[-i-1] = gate
                op_s.append((''.join(ss), 1.0))
            op.append(op_s)
        return op
    
    def _o_xyz(self):
        s = 'I'*self.num_qubits
        op = []
        for i in range(self.num_qubits):
            for gate in ['X', 'Y', 'Z']:
                ss = list(s)
                ss[-i-1] = gate
                op.append([(''.join(ss), 1.0)])
        return op
    
    def add_measurement(self, circuit):
        """ """
        if isinstance(self.projection, str):
            if self.projection=='xyz':
                self.proj_ops = self._o_xyz()
            elif self.projection=='xyz_sum':
                self.proj_ops = self._o_xyz_sum()
        else:
            self.proj_ops = self.projection
        ops = []
        for proj_op in self.proj_ops:
            ops.append( ~StateFn(PauliSumOp.from_list(proj_op)) @ StateFn(circuit) )
        return ListOp(ops)
    
    def projected_feature_map(self, x):
        """ """
        qc = QuantumCircuit(self.fm.num_qubits)

        x_dict = dict(zip(self.fm.parameters, x))
        psi_x = self.fm.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)
        # print(qc.decompose().draw())
        
        op = self.add_measurement(qc)
        expectation = AerPauliExpectation().convert(op.reduce())
        sampler = CircuitSampler(self.backend).convert(expectation)
        return sampler.eval()

    def evaluate(self, X_1, X_2=None):
        """ """
        X_1_proj = np.array([ self.projected_feature_map(x) for x in X_1 ])
        if X_2 is None:
            X_2_proj = X_1_proj
        else:
            X_2_proj = np.array([ self.projected_feature_map(x) for x in X_2 ])

        kernel = np.zeros(shape=(X_1_proj.shape[0], X_2_proj.shape[0]))
        for i in range(X_1_proj.shape[0]):
            for j in range(X_2_proj.shape[0]):
                value = np.exp(-self.gamma * ((X_1_proj[i] - X_2_proj[j]) ** 2).sum())
                kernel[i,j] = value
        return kernel