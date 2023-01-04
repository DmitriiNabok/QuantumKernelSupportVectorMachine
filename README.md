# Quantum Kernel Support Vector Machine

Collection of wrappers and tools for the quantum kernel based machine learning tasks.

Specifically, we are dealing with the hybrid quantum/classical algorithm when the kernel matrix is estimated on a quantum computer and imported into the support vector machine (SVM) (the QSVM approach).
The implementation is built around the [Qiskit](https://qiskit.org/) quantum computer simulation toolkit and includes different variants of the quantum kernel setup.
[Scikit-Learn](https://scikit-learn.org/stable/) is used as the classical backend.


## List of features

1. Generalized and flexible quantum circuit generator that simplifies
   construction of the data encoding and the variational ansaetze with Qiskit
   (`QuantumFeatureMap`)..
2. Quantum kernel support vector classifier (`QKSVC`).
3. Quantum kernel hyperparameter search with Scikit-Learn tools (`GridSearchCV`).
4. Custom version of the Qiskit quantum kernel training algorithm
   (`QuantumKernelTraining`).
5. Implementation of the projected quantum kernel technique (by Huang et al.
   (2021))
6. Quantum kernel version of the support vector regression (`QKSVR`) (with the
   support of the quantum circuit hyperparameter search).


## Theory

Quantum kernel machine learning is using the idea to apply a quantum feature map $`\phi(\vec{x})`$ to express a classical data point $`\vec{x}`$ in a quantum Hilbert space $`|\phi(\vec{x})\rangle\langle\phi(\vec{x})|`$.
In this way, the kernel matrix can be estimated with a quantum computer as

```math
K_{ij} = \left| \langle \phi(\vec{x}_j) | \phi(\vec{x}_i) \rangle\right|^2.
```

To setup the quantum feature map $`\phi(\vec{x})`$ one needs to provide a quantum circuit that embeds a data vector $`\vec{x}`$ into a quantum state. There are multiple ways how to build such a circuit. In this implementation, we are following the approach from [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/pdf/1804.11326.pdf) to encode the classical data with a help of quantum parametric gates that describe rotation of a qubit in the Hilbert space.
For understanding the basics of this approach, also known as the quantum kernel estimation (QKE) algorithm, we refer to the Qiskit tutorial [Quantum Kernel Machine Learning](https://qiskit.org/documentation/machine-learning/tutorials/03_quantum_kernel.html).

In our implementation, we are using the core subroutines provided in Qiskit to setup and compute the quantum kernel matrix ([`QuantumKernel` instance](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.kernels.QuantumKernel.html)).

Qiskit contains a few specialized feature map generators (`PauliFeatureMap`, `ZFeatureMap`, `ZZFeatureMap`).
Instead, we implement our own feature map circuit generator that combines all common and widely used strategies applied in literature to classify various datasets.
The implemented class `QuantumFeatureMap` provides high flexibility in setting parameters for the encoding circuits as well as for the more general encoding+variational parametric circuits setup.

As an important QKE algorithm extension for improving the classification performance of QSVM, the quantum kernel training (QKT) schemes have been recently proposed.
The scheme uses Quantum Kernel Alignment (QKA) for a binary classification task.
QKA is a technique that iteratively adapts a parametrized quantum kernel to a dataset while converging to the maximum SVM margin.
Details regarding the Qiskit's implementation are given in ["Covariant quantum kernels for data with group structure"](https://arxiv.org/abs/2105.03406).
We provide a simplified wrapper for QKT functions implemented in [Qiskit QKT](https://qiskit.org/documentation/machine-learning/tutorials/08_quantum_kernel_trainer.html).


## Installation (with `conda`)

### 0. Clone the repository

```bash
git clone https://jugit.fz-juelich.de/qai2/qksvm
cd qksvm
```

### 1. Create a virtual environment and activate it

```bash
conda create --name qksvm python=3
conda activate qksvm
```

### 2. Install packages (including all requirements)

```bash
pip install -e . 
```

### 3. Add the environment to Jupyter Notebooks

```bash
conda install -c ipykernel
python -m ipykernel install --user --name=qksvm
```


## Usage

For a quick start, applications of the implemented quantum kernel machine
learning tools are examplified in `tutorials`.
The recommended order to get familiar with the methods and implementation would be

1. Classical_Kernel.ipynb
2. Quantum_Kernel_Estimate.ipynb
3. Quantum_Kernel_Training.ipynb
4. (optional) Projected_Quantum_Kernel.ipynb
5. (in preparation) Quantum_Kernel_Regression.ipynb


