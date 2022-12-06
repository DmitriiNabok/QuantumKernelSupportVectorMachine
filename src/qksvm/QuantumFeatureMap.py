from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import Sequence, Union

# -------------------------------------------------------------------
class QuantumFeatureMap(QuantumCircuit):
    """Feature Map Circuit builder class."""

    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        num_layers: int,
        gates: Sequence[str],
        scale: bool = False,
        alpha: float = None,
        repeat: bool = True,
        entanglement: Union[str, list] = "linear",
        name: str = "QuantumFeatureMap",
    ) -> None:
        """QKT Feature Map circuit constructor.

        Args:
            num_features: Number of features (input data dimensionality)
            num_qubits: Number of qubits (feature map width)
            num_layers: Number of repeating layers (feature map depth)
            gates: List of gates used to encode (CAPITAL) and train (small) the data
            scale: Include data scaling prefactor as an additional variational parameter (default=False)
            alpha: Value of the fixed data scaling prefactor when scale=False (default=None)
            repeat: Repeating encoding scheme, which works in the case when num_features < num_qubits (default=True)
            entanglement: Entanglement structure of the circuit ('linear', 'ring', 'full') (default='linear')
            name: Name of QuantumCircuit object

        Usage:
            fm = FeatureMap(
                num_features=2,
                num_qubits=2,
                num_layers=1,
                gates=['rx', 'rz', 'cz', 'RY'],
            )
            print(fm.draw(plot_barriers=False, fold=120))

        """
        if num_features < 1:
            raise ValueError("Wrong number of features (empty data)!")
        self.num_features = num_features

        if num_layers < 1:
            raise ValueError("Feature map depth must be larger than 0!")
        self.num_layers = num_layers

        # use the repeating encoding scheme
        self.repeat = repeat

        # Call the QuantumCircuit initialization
        super().__init__(
            num_qubits,
            name=name,
        )

        # data scaling prefactor as a model parameter
        self.scale = scale
        self.alpha = Parameter("α")

        # connection map for 2 parameter gates
        if type(entanglement) is str:
            # generate the entanglement map
            self.generate_map(entanglement)
        else:
            self.entanglement = entanglement

        # setup parameters for encoding and training
        self.setup_parameters(gates)

        # build the circuit
        i_encod = 0
        if self.scale:
            i_train = 1
        else:
            i_train = 0

        for _ in range(self.num_layers):
            for s in gates:
                if s.isupper():
                    # encoding sublayer
                    i_encod = self.build_circuit(s.lower(), self.encod_params, i_encod)
                    if self.repeat and i_encod == self.num_features:
                        i_encod = 0
                if s.islower():
                    # training sublayer
                    i_train = self.build_circuit(s.lower(), self.train_params, i_train)
            if not self.repeat:
                if i_encod == self.num_features:
                    i_encod = 0

        # if i_encod%self.num_features != 0:
        #     print('\nWarning:')
        #     print('\tNot all features seem to be equally encoded. Check your input and either increase the number of layers or the number of qubits.\n')

        # apply constant data scaling prefactor
        if not self.scale:
            if alpha is not None:
                try:
                    self.assign_parameters({self.alpha: alpha}, inplace=True)
                except:
                    pass
                self.alpha = alpha

        return

    def generate_map(self, entanglement: str) -> None:
        """Generate the entanglement map

        Args:
            entanglement: Entanglement type to generate the 2-qubits entanglement map
                          Currently supported: 'linear', 'ring', and 'full'
        """
        self.entanglement = []
        if entanglement == "linear":
            for i in range(self.num_qubits - 1):
                self.entanglement.append([i, i + 1])
        elif entanglement == "linear_":
            for i in range(0, self.num_qubits - 1, 2):
                self.entanglement.append([i, i + 1])
            for i in range(1, self.num_qubits - 1, 2):
                self.entanglement.append([i, i + 1])
        elif entanglement == "ring":
            for i in range(self.num_qubits):
                self.entanglement.append([i, (i + 1) % self.num_qubits])
        elif entanglement == "ring_":
            for i in range(0, self.num_qubits, 2):
                self.entanglement.append([i, (i + 1) % self.num_qubits])
            for i in range(1, self.num_qubits, 2):
                self.entanglement.append([i, (i + 1) % self.num_qubits])
        elif entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i != j:
                        self.entanglement.append([i, j])
        elif entanglement == "full_":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.entanglement.append([i, j])
        else:
            raise ValueError("Unknown entanglement type!")
        return

    def setup_parameters(self, gates: Sequence[str]) -> None:
        """Analyze the input gate structure to determine number of variational
        parameters and setup encoding and training parameter vectors.

        Args:
            gates: List of gates
        """
        # Parametrized gates
        r_gates = [
            "rx",
            "ry",
            "rz",
        ]
        cr_gates = [
            "crx",
            "cry",
            "crz",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
        ]

        # --------------------------------
        # Set training parameters vector
        # --------------------------------
        num_params = 0
        for gate in gates:
            if gate.islower():
                if gate in r_gates:
                    num_params += 1 * self.num_qubits
                elif gate in cr_gates:
                    num_params += 1 * len(self.entanglement)

        num_params = num_params * self.num_layers
        if self.scale:
            num_params += 1

        self.train_params = ParameterVector("θ", num_params)

        if self.scale:
            self.alpha = self.train_params[0]  # scaling parameter

        # --------------------------------
        # Set encoding parameters vector
        # --------------------------------
        t_list = [s.isupper() and (s.lower() in r_gates + cr_gates) for s in gates]
        if not any(t_list):
            print("\nWarning:")
            print(
                "\tEncoding circuit is not specified. Is it what you are aiming at?\n"
            )

        params = ParameterVector("x", self.num_features)
        self.encod_params = [self.alpha * p for p in params]
        return

    def build_circuit(self, gate: Sequence[str], params, j0: int = 0) -> int:
        """Generic circuit builder.

        Args:
            gates: List of gates
            params: List of gate parameters
            j0: Starting parameter index value

        Returns:
            j: starting parameter index value for next function call
        """

        # -------------------------------------------------------------
        # Auxiliary exception for the repeating encoding scheme
        # -------------------------------------------------------------
        class AllDone(Exception):
            pass

        def check(j):
            if j >= num_params:
                raise AllDone

        num_params = len(params)
        j = j0

        try:
            _gate = getattr(self, gate)
            if gate in ["crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx", "rzx"]:
                for pair in self.entanglement:
                    if not self.repeat:
                        check(j)
                    _gate(params[j % num_params], pair[0], pair[1])
                    j += 1
            elif gate in ["rx", "ry", "rz"]:
                for i in range(self.num_qubits):
                    if not self.repeat:
                        check(j)
                    _gate(params[j % num_params], self.qubits[i])
                    j += 1
            elif gate in ["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg"]:
                for i in range(self.num_qubits):
                    _gate(self.qubits[i])
            elif gate in ["cx", "cy", "cz", "ch", "swap", "iswap"]:
                for pair in self.entanglement:
                    _gate(pair[0], pair[1])
        except AllDone:
            pass

        self.barrier()
        return j
