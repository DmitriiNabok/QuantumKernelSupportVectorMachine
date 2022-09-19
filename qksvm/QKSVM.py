# External libraries
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from functools import partial
from typing import Union, Optional, Sequence, List, Tuple

# Qiskit imports
from qiskit.utils import algorithm_globals
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA, Optimizer
from qiskit_machine_learning.kernels import QuantumKernel

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
                AerSimulator(
                    method='statevector',
                    max_parallel_threads=8,
                ),
                seed_simulator=self.random_state, seed_transpiler=self.random_state,
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
            AerSimulator(
                method='statevector',
                max_parallel_threads=8,
            ),
            seed_simulator=seed, seed_transpiler=seed,
        )

    # Instantiate Quantum Kernel
    quant_kernel = QuantumKernel(fm,
                                 user_parameters=fm.train_params,
                                 quantum_instance=backend)

    # Set up the optimizer
    from qksvm.QuantumKernelTraining import QKTCallback
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=maxiter,
                    callback=cb_qkt.callback,
                    learning_rate=None,
                    perturbation=None,
                    # second_order=True,
               )

    # Instantiate a quantum kernel trainer
    from qksvm.QuantumKernelTraining import QuantumKernelTrainer
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss="svc_loss",
        optimizer=spsa_opt,
        initial_point=initial_point,
    )

    # Train the kernel using QKT directly
    qkt_results = qkt.fit(X_train, y_train)
    
    if plot:
        cb_qkt.plot()
        
    return qkt_results



