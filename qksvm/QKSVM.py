# External libraries
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from functools import partial
from typing import Sequence

# Qiskit imports
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit.utils import algorithm_globals

# SciKit-Learn imports
from sklearn.svm import SVC
# from src import scores


#-------------------------------------------------------------------    
def QSVM_QKE(fm,
             seed=None,
             backend=None,
             **kwargs,
    ):
    """Quantum Kernel Embedding algorithm.
    
    Args:
        fm (QuantumCircuit): Quantum Feature Map
        X_train, y_train (nd.array): If provided, the classifier is fit to the training data
        seed (int): Random generator seed
        backend (QuantumInstance): Qiskit backend instance
        **kwargs: Other Scikit-Learn SVC parameters (e.g., C, class_weight, etc.)

    Returns:
        qsvc: SciKit-Learn SVC classifier
    """
    
    if hasattr(fm, 'train_params'):
        if len(fm.train_params)>0:
            raise ValueError("Circuit contains variational parameters! QKE algorithm is not applicable!")
        
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    if backend is None:
        # default backend
        backend = QuantumInstance(
            AerSimulator(method="statevector"),
            shots=1024, seed_simulator=seed, seed_transpiler=seed,
            backend_options = {'method': 'automatic', 
                               'max_parallel_threads': 0,
                               'max_parallel_experiments': 0,
                               'max_parallel_shots': 0},
        )

    # Instantiate quantum kernel
    quant_kernel = QuantumKernel(fm, quantum_instance=backend)

    # Use SVC for classification
    qsvc = SVC(kernel=quant_kernel.evaluate, random_state=seed, **kwargs)
    return qsvc


#-------------------------------------------------------------------    
class QSVC(SVC):
    """
    Extended svm.SVC Scikit-Learn class that supports quantum kernels.
    Can be applied in GridSearchCV for optimizing SVM hyperparameters.
    
    Args:
        fm (QuantumCircuit): Qiskit quantum feature map circuit
        alpha (float=2.0): Input data scaling prefactor
        seed (int=None): Random generator seed
        backend (QuantumInstance=None): Qiskit backend instance
        ... : Other parameters from svm.SVC (e.g., C, class_weight, etc.)
    
    Returns:
        Scikit-Learn svm.SVC object with the quantum kernel
    """
    
    def __init__(
        self,
        fm,
        alpha=None,
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

        if hasattr(fm, 'train_params'):
            if len(fm.train_params)>0:
                raise ValueError("Circuit contains variational parameters! QKE algorithm is not applicable!")
                
        self.fm = fm
        self.backend = backend
        
        if isinstance(self.fm.alpha, float) and alpha is not None:
            print('\nWarning (QSVC.fit()):')
            print('\tFeature map circuit has already a fixed value for the scaling parameter alpha predefined earlier in QuantumFeatureMap.')
            print(f"\tSpecified alpha={alpha} will have no effect!")
        if alpha is None:
            self.alpha = 2.0
        else:
            self.alpha = alpha
        
        if self.backend is None:
            np.random.seed(self.random_state)
            algorithm_globals.random_seed = self.random_state
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                shots=1024, seed_simulator=self.random_state, seed_transpiler=self.random_state,
                backend_options = {'method': 'automatic', 
                                   'max_parallel_threads': 0,
                                   'max_parallel_experiments': 0,
                                   'max_parallel_shots': 0},
            )        
        
    def fit(self, X, y):
        if isinstance(self.fm.alpha, float):
            _fm = self.fm
        else:
            _fm = self.fm.assign_parameters({self.fm.alpha: self.alpha})
        self.kernel = QuantumKernel(_fm, quantum_instance=self.backend).evaluate
        SVC.fit(self, X, y)
        return self
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    
#-------------------------------------------------------------------    
def QuantumKernelOptimizer(
        fm, 
        X, y,
        init_params=None,
        seed=None, 
        backend=None, 
        optimizer=None,
        plot=False,
    ):
    """
    Quantum Kernel Training algorithm.
    
    Args:
        fm (QuantumCircuit): Quantum Kernel (feature map) circuit
        X, y (nd.array): Training data
        init_params (nd.array): Starting values for the optimization parameters
        seed (int): Random generator seed
        backend (QuantumInstance): Qiskit Backend instance

    Returns:
        qkt_results (QuantumKernelTrainerResult): Results of the QKT algorithm
    """
    
    if init_params is None:
        # Set initial parameters
        np.random.seed(seed)
        init_params = np.random.uniform(0, 2*np.pi, len(fm.train_params))
        if fm.scale:
            init_params[0] = 2.0
    
    if backend is None:
        # Default backend
        algorithm_globals.random_seed = seed
        backend = QuantumInstance(
            AerSimulator(method="statevector"),
            shots=1024, seed_simulator=seed, seed_transpiler=seed,
            backend_options = {'method': 'automatic', 
                               'max_parallel_threads': 0,
                               'max_parallel_experiments': 0,
                               'max_parallel_shots': 0},
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
    QKT = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss="svc_loss",
        optimizer=optimizer,
        initial_point=init_params,
    )

    # Train the kernel
    return QKT.fit(X, y)


#-------------------------------------------------------------------    
def QSVM_QKT(fm, 
             X_train, y_train, 
             scale0=2.0, maxiter=100, seed=None,
             init_params=None,
             backend=None,
             verbose=False,
             plot=False,
             **kwargs,
    ):
    """
    Wrapper around the QKT and SVM routines.

    Args:
        fm (QuantumCircuit): Quantum Feature Map circuit
        X_train, y_train (nd.array): Training data
        scale0 (float=2.0): Initial value of the data scaling prefactor
        init_params (nd.array): Starting values for the optimization parameters
        maxiter (int=100): Maximum number of the SPSA optimizer iterations
        seed (int): Random generator seed
        backend (QuantumInstance): Qiskit Backend instance
        verbose (bool): Increased output verbosity
        plot (bool): Visualize the loss function and the optimized kernel matrix
        **kwargs (dict): Other Scikit-Learn SVC parameters

    Returns:
        qsvc: SciKit-Learn SVC classifier
    """

    if init_params is None:
        # Set initial parameters
        np.random.seed(seed)
        init_params = np.random.uniform(0, 2*np.pi, len(fm.train_params))
        if fm.scale:
            init_params[0] = scale0

    qkt_results = quantum_kernel_trainer(
        fm, 
        X_train, y_train, 
        init_params, 
        maxiter=maxiter,
        backend=backend,
        seed=seed, 
        plot=plot,
    )
    if verbose: print(qkt_results)
    
    # Use SVC for classification
    qsvc = SVC(kernel=qkt_results.quantum_kernel.evaluate, random_state=seed, **kwargs)
    qsvc.fit(X_train, y_train)
    return qsvc


#-------------------------------------------------------------------            
def quantum_kernel_trainer(
        fm, 
        X_train, y_train, 
        initial_point, 
        maxiter=100, 
        backend=None, 
        seed=None, 
        plot=True,
    ):
    """
    Quantum Kernel Training algorithm driver routine. 
    Adopted from https://qiskit.org/documentation/machine-learning/tutorials/08_quantum_kernel_trainer.html
    
    Args:
        fm (QuantumCircuit): Quantum Feature Map circuit
        X_train, y_train (nd.array): Training data
        initial_point (nd.array): Initial values for variational circuit parameters
        maxiter (int=100): Maximum number of optimization iterations
        backend (QuantumInstance): Qiskit Backend instance
        seed (int=None): Random number generator seed
        plot (bool=True): Visualize the loss function and the optimized kernel matrix    
    """
    
    np.random.seed(seed)
    algorithm_globals.random_seed = seed
    
    if backend is None:
        # default backend
        backend = QuantumInstance(
            AerSimulator(method="statevector"),
            shots=1024, seed_simulator=seed, seed_transpiler=seed,
            backend_options = {'method': 'automatic', 
                               'max_parallel_threads': 0,
                               'max_parallel_experiments': 0,
                               'max_parallel_shots': 0},
        )

    # Instantiate Quantum Kernel
    quant_kernel = QuantumKernel(fm,
                                 user_parameters=fm.train_params,
                                 quantum_instance=backend)

    # Set up the optimizer
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=maxiter,
                    callback=cb_qkt.callback,
                    learning_rate=None,
                    perturbation=None,
                    # second_order=True,
               )

    # Instantiate a quantum kernel trainer
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss="svc_loss",
        optimizer=spsa_opt,
        initial_point=initial_point,
    )

    # Train the kernel using QKT directly
    qkt_results = qkt.fit(X_train, y_train)
    
    if plot:    
        plot_data = cb_qkt.get_callback_data() # callback data
        K = qkt_results.quantum_kernel.evaluate(X_train) # kernel matrix evaluated on the training samples

        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(1, 2, figsize=(14,5))
        ax[0].plot([i+1 for i in range(len(plot_data[0]))],
                   np.array(plot_data[2]),
                   c='k',
                   marker='o'
        )
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Loss')
        ax[1].imshow(K, cmap=cm.get_cmap('bwr', 20))
        ax[1].axis('off')
        fig.tight_layout()
        plt.show()
        
    return qkt_results


#-------------------------------------------------------------------    
class CustomKernelLoss(KernelLoss):
    """
    User defined Kernel Loss class that can be used to modify the loss function and 
    output it for plotting.
    Adopted from https://github.com/qiskit-community/prototype-quantum-kernel-training/blob/main/docs/how_tos/create_custom_kernel_loss_function.ipynb
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def evaluate(self,
                 parameter_values: Sequence[float],
                 quantum_kernel: QuantumKernel,
                 data: np.ndarray,
                 labels: np.ndarray):
        """
        Evaluate the Loss of a trainable quantum kernel
        at a particular setting of param_values on
        a particular dataset.
        """
        # Bind the user parameter values
        quantum_kernel.assign_user_parameters(parameter_values)
        kmatrix = quantum_kernel.evaluate(data)
        
        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kmatrix, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kmatrix = kmatrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (0.5 * (dual_coefs.T @ kmatrix @ dual_coefs))
        
        return loss

    def get_variational_callable(self,
                                 quantum_kernel: QuantumKernel,
                                 data: np.ndarray,
                                 labels: np.ndarray):
        """
        Wrap our evaluate method so to produce a callable
        which maps user_param_values to loss scores.
        """
        return partial(self.evaluate, quantum_kernel=quantum_kernel, data=data, labels=labels)
    

#-------------------------------------------------------------------    
def plot_kernel_loss(fm, X_train, y_train, params, backend=None, grid=[1, 10, 100], seed=None):
    """
    Specialized routine to compute and visualize the QKT-SVM loss function by 
    only varying the first optimization parameter (the scaling data prefactor).
    
    Args:
        fm (QuantimCircuit): Quantum Feature Map circuit
        X_train, y_train (nd.array): Training data
        params (nd.array): Starting values for the variational circuit parameters
        backend (QuantunInstance=None): Qiskit Backend instance
        seed (int=None): Random number generator seed
        grid (list): [start, end, N_points] - range of values (x-axis) where 
                     the loss function is computed and visualized
    """
    
    algorithm_globals.random_seed = seed
    
    if backend is None:
        # default backend
        backend = QuantumInstance(
            AerSimulator(method="statevector"),
            shots=1024, seed_simulator=seed, seed_transpiler=seed,
            backend_options = {'method': 'automatic', 
                               'max_parallel_threads': 0,
                               'max_parallel_experiments': 0,
                               'max_parallel_shots': 0},
        )

    quant_kernel = QuantumKernel(fm,
                                 user_parameters=fm.train_params,
                                 quantum_instance=backend)

    kernel_loss = CustomKernelLoss().get_variational_callable(quant_kernel, X_train, y_train)

    theta = np.linspace(grid[0], grid[1], int(grid[2]))
    loss_values = np.zeros(len(theta))
    
    for i, val in enumerate(theta):
        params[0] = val
        loss_values[i] = kernel_loss(params)

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(8, 4))
    plt.plot(theta, loss_values)
    plt.xlabel('θ[0]')
    plt.ylabel('Kernel Loss')
    plt.show()


#-------------------------------------------------------------------    
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
