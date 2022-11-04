import numpy as np
from functools import partial
from typing import Union, Optional, Sequence, List, Tuple
import copy
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit.utils import algorithm_globals
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA, Optimizer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit.algorithms.variational_algorithm import VariationalResult

# Modified versions
from qksvm.LossFunctions import SVCLoss, KTALoss


def QKTKernel(
    fm,
    X_train, y_train, 
    init_params, 
    C=1.0, class_weight=None,
    maxiter=100,
    plot=False, seed=None,
):
    """ 
    User template to control all the paramaters of the QKT algorithm and 
    visualize the training info.
    
    Return: 
        qkt_results object that contains the optimized kernel: qkt_results.quantum_kernel.evaluate
    """

    #----------------------
    # Select loss function
    #----------------------
    from qiskit_machine_learning.utils.loss_functions import SVCLoss
    loss = SVCLoss(C=C, class_weight=class_weight)

    #----------------------
    # Setup optimizer
    #----------------------
    from qksvm.QuantumKernelTraining import QKTCallback
    callback = QKTCallback()

    from qiskit.algorithms.optimizers import SPSA
    from qksvm.QuantumKernelTraining import TerminationChecker
    optimizer = SPSA(
        maxiter=maxiter,
        learning_rate=None,
        perturbation=None,
        callback=callback.callback,
        termination_checker=TerminationChecker(0.01, N=7),
    )

    #------------------------
    # Choose quantum backend
    #------------------------
    from qiskit.utils import QuantumInstance
    from qiskit.providers.aer import AerSimulator
    backend = QuantumInstance(
        AerSimulator(
            method='statevector',
            max_parallel_threads=8,
        ),
        seed_simulator=seed, seed_transpiler=seed,
    )

    #----------------------------
    # Run quantum kernel trainer
    #----------------------------
    from qksvm.QuantumKernelTraining import QuantumKernelOptimizer
    qkt_results = QuantumKernelOptimizer(
        fm, 
        X_train, y_train,
        init_params=init_params,
        backend=backend, 
        optimizer=optimizer,
        loss=loss,
        seed=seed,
    )

    print('SVCLoss optimal value: ', qkt_results.optimal_value)
    print('Optimal parameters:', qkt_results.optimal_point)
    if plot:
        # Visualize the optimization workflow
        plot_data = callback.get_callback_data()
        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot([i+1 for i in range(len(plot_data[0]))],
                np.array(plot_data[2]),
                c='k',
                marker='o'
        )
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.title.set_text('Optimization')
        plt.show()
        
    return qkt_results


def QuantumKernelOptimizer(
        fm, 
        X, y,
        init_params=None,
        backend=None, 
        optimizer=None,
        loss=None,
        seed=None, 
    ):
    """
    Quantum Kernel Training algorithm.
    
    Args:
        fm (QuantumCircuit): Quantum Kernel (feature map) circuit
        X, y (nd.array): Training data
        init_params (nd.array): Starting values for the optimization parameters
        backend (QuantumInstance): Qiskit Backend instance
        optimizer (Optimizer): An instance of ``Optimizer`` to be used in training
        loss (SVCLoss): User provided loss function
        seed (int): Random generator seed

    Returns:
        qkt_results (QuantumKernelTrainerResult): Results of the QKT algorithm
    """
    
    np.random.seed(seed)
    algorithm_globals.random_seed = seed
    
    if init_params is None:
        # Set initial parameters
        init_params = np.random.uniform(0, 2*np.pi, len(fm.train_params))
        if fm.scale:
            init_params[0] = 2.0
    
    if backend is None:
        # Default backend
        backend = QuantumInstance(
            AerSimulator(
                method='statevector',
                max_parallel_threads=0,
            ),
            seed_simulator=seed, seed_transpiler=seed,
        )

    if optimizer is None:
        # Set up the default optimizer SPSA
        optimizer = SPSA(
            maxiter=100,
            learning_rate=None,
            perturbation=None,
            # second_order=True,
        )
        
    # Instantiate Quantum Kernel
    quant_kernel = QuantumKernel(
        fm,
        user_parameters=fm.train_params,
        quantum_instance=backend
    )
    
    # Instantiate a quantum kernel trainer
    if loss is None:
        loss = SVCLoss(C=1.0, class_weight=None)
        
    QKT = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss=loss,
        optimizer=optimizer,
        initial_point=init_params,
    )

    # Train the kernel
    return QKT.fit(X, y)


class QuantumKernelTrainerResult(VariationalResult):
    """Quantum Kernel Trainer Result."""

    def __init__(self) -> None:
        super().__init__()
        self._quantum_kernel: QuantumKernel = None

    @property
    def quantum_kernel(self) -> Optional[QuantumKernel]:
        """Return the optimized quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        self._quantum_kernel = quantum_kernel


class QuantumKernelTrainer:

    def __init__(
        self,
        quantum_kernel: QuantumKernel,
        loss: Optional[Union[str, KernelLoss]] = None,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            quantum_kernel: QuantumKernel to be trained
            loss (str or KernelLoss): Loss functions available via string:
                       {'svc_loss': SVCLoss()}.
                       If a string is passed as the loss function, then the
                       underlying KernelLoss object will exhibit default
                       behavior.
            optimizer: An instance of ``Optimizer`` to be used in training. Since no
                       analytical gradient is defined for kernel loss functions, gradient-based
                       optimizers are not recommended for training kernels.
            initial_point: Initial point from which the optimizer will begin.

        Raises:
            ValueError: unknown loss function
        """
        # Class fields
        self._quantum_kernel = quantum_kernel
        self._initial_point = initial_point
        self._optimizer = optimizer or SPSA()

        # Loss setter
        self._set_loss(loss)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Return the quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        """Set the quantum kernel."""
        self._quantum_kernel = quantum_kernel

    @property
    def loss(self) -> KernelLoss:
        """Return the loss object."""
        return self._loss

    @loss.setter
    def loss(self, loss: Optional[Union[str, KernelLoss]]) -> None:
        """
        Set the loss.

        Args:
            loss: a loss function to set

        Raises:
            ValueError: Unknown loss function
        """
        self._set_loss(loss)

    @property
    def optimizer(self) -> Optimizer:
        """Return an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[Sequence[float]]:
        """Return initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Sequence[float]]) -> None:
        """Set the initial point"""
        self._initial_point = initial_point

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> QuantumKernelTrainerResult:
        """
        Train the QuantumKernel by minimizing loss over the kernel parameters. The input
        quantum kernel will not be altered, and an optimized quantum kernel will be returned.

        Args:
            data (numpy.ndarray): ``(N, D)`` array of training data, where ``N`` is the
                              number of samples and ``D`` is the feature dimension
            labels (numpy.ndarray): ``(N, 1)`` array of target values for the training samples

        Returns:
            QuantumKernelTrainerResult: the results of kernel training

        Raises:
            ValueError: No trainable user parameters specified in quantum kernel
        """
        # Number of parameters to tune
        num_params = len(self._quantum_kernel.user_parameters)
        if num_params == 0:
            msg = "Quantum kernel cannot be fit because there are no user parameters specified."
            raise ValueError(msg)

        # Bind inputs to objective function
        output_kernel = copy.deepcopy(self._quantum_kernel)

        # Randomly initialize the initial point if one was not passed
        if self._initial_point is None:
            self._initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        loss_function = partial(
            self._loss.evaluate, quantum_kernel=self.quantum_kernel, data=data, labels=labels
        )
        
        # Check if the optimizer support bounds
        if self._optimizer.is_bounds_required:
            bounds = np.array(self._optimizer._options['bounds'])
        else:
            bounds = None
            
        # Run the optimizer
        opt_results = self._optimizer.minimize(
            fun=loss_function,
            x0=self._initial_point,
            bounds=bounds,
        )

        # Return kernel training results
        result = QuantumKernelTrainerResult()
        result.optimizer_evals = opt_results.nfev
        result.optimal_value = opt_results.fun
        result.optimal_point = opt_results.x
        result.optimal_parameters = dict(zip(output_kernel.user_parameters, opt_results.x))

        # Return the QuantumKernel in optimized state
        output_kernel.assign_user_parameters(result.optimal_parameters)
        result.quantum_kernel = output_kernel

        return result

    def _set_loss(self, loss: Optional[Union[str, KernelLoss]]) -> None:
        """Internal setter."""
        if loss is None:
            loss = SVCLoss()
        elif isinstance(loss, str):
            loss = self._str_to_loss(loss)

        self._loss = loss

    def _str_to_loss(self, loss_str: str) -> KernelLoss:
        """Function which maps strings to default KernelLoss objects."""
        if loss_str == "svc_loss":
            loss_obj = SVCLoss()
        elif loss_str == "kta_loss":
            loss_obj = KTALoss()
        else:
            raise ValueError(f"Unknown loss {loss_str}!")

        return loss_obj
    
    
class QKTCallback:
    """Callback wrapper class for retrieving the data from optimization run."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]
        
    def plot(self):
        """Visualize the optimization results"""
        plot_data = self.get_callback_data()
        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot([i+1 for i in range(len(plot_data[0]))],
                np.array(plot_data[2]),
                c='k',
                marker='o'
        )
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.title.set_text('Optimization')
        plt.show()


class TerminationChecker:
    def __init__(self, tol: float, N: int=5):
        self.tol = tol
        self.N = N
        self.values = []
    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        self.values.append(value)
        if len(self.values) > self.N:
            last_values = self.values[-self.N:]
            std = np.std(last_values)
            if std < self.tol:
                return True
        return False