# Loss functions for the quantum kernel optimzation.
# Added visualization method: the loss function as a function of the first
# variational parameter.

from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Union, Optional, Sequence, List, Tuple


class LossPlot:
    def plot(
        self,
        quantum_kernel: QuantumKernel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: np.ndarray,
        grid: list = [0.1, 8, 50],
        show: bool = True,
    ) -> np.ndarray:
        """
        Specialized routine to compute and visualize the loss function by
        only varying the first optimization parameter (the scaling data prefactor).

        Args:
            quantum_kernel (QuantumKernel): Quantum kernel instance
            X_train, y_train (nd.array): Training data
            params (nd.array): Variational circuit parameters
            grid (list): [start, end, N_points] - range of values (x-axis) where
                         the loss function is computed
            show (bool): Loss function visualization

        Returns:
            loss_values (np.ndarray): Loss function values computed on the (single) parameter grid 
        """
        kernel_loss = partial(
            self.evaluate,
            quantum_kernel=quantum_kernel,
            data=X_train,
            labels=y_train,
        )

        theta = np.linspace(grid[0], grid[1], int(grid[2]))
        loss_values = np.zeros(len(theta))

        for i, val in enumerate(theta):
            params[0] = val
            loss_values[i] = kernel_loss(params)

        if show:
            plt.rcParams["font.size"] = 15
            plt.figure(figsize=(8, 4))
            plt.plot(theta, loss_values)
            plt.xlabel("θ[0]")
            plt.ylabel("Kernel Loss")
            plt.show()
        else:
            return loss_values


class SVCLoss(KernelLoss, LossPlot):
    """
    User defined Kernel Loss class that can be used to modify the loss function and
    output it for plotting.
    Adopted from https://github.com/qiskit-community/prototype-quantum-kernel-training/blob/main/docs/how_tos/create_custom_kernel_loss_function.ipynb
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Evaluate the SVC loss of a trainable quantum kernel.
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
        loss = np.sum(np.abs(dual_coefs)) - (
            0.5 * (dual_coefs.T @ kmatrix @ dual_coefs)
        )

        return loss


class KTALoss(KernelLoss, LossPlot):
    """Kernel-target alignment (KTA) between the computed kernel and training labels.
    Adopted from https://pennylane.ai/qml/demos/tutorial_kernels_module.html
    T. Hubregtsen, et al., “Training Quantum Embedding Kernels on Near-Term Quantum Computers”. arXiv preprint arXiv:2105.02276.
    """

    def __init__(self, rescale=True):
        self.rescale = rescale

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Evaluate the KTA loss of a trainable quantum kernel.
        """
        # Bind the user parameter values
        quantum_kernel.assign_user_parameters(parameter_values)
        K = quantum_kernel.evaluate(data)

        if self.rescale:
            nplus = np.count_nonzero(np.array(labels) == 1)
            nminus = len(labels) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in labels])
        else:
            _Y = np.array(labels)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        inner_product = inner_product / norm

        return -inner_product
