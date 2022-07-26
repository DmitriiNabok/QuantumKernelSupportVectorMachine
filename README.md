# Quantum Kernel Support Vector Machine

The repository provides implementation of the quantum kernel based support vector machine algorithm (QSVM) for binary classification.

Quantum kernel machine learning is using the idea to apply a quantum feature map $`\phi(\vec{x})`$ to express a classical data point $`\vec{x}`$ in a quantum Hilbert space $`|\phi(\vec{x})\rangle\langle\phi(\vec{x})|`$.
In this way, the kernel matrix can be estimated with a quantum computer as
```math
K_{ij} = \left| \langle \phi(\vec{x}_j) | \phi(\vec{x}_i) \rangle\right|^2.
```
To setup of the quantum feature map $`\phi(\vec{x})`$ one needs to provide a quantum circuit that embeds a data vector $`\vec{x}`$ into a quantum state. There are multiple ways how to build such a circuit. In this implementation, we are following the approach from [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/pdf/1804.11326.pdf) to encode the classical data with a help of quantum parametric gates that describe rotation of a qubit in the Hilbert space.
For understanding the basics of this approach, received a name quantum kernel embedding (QKE), we refer to the Qiskit tutorial [Quantum Kernel Machine Learning](https://qiskit.org/documentation/machine-learning/tutorials/03_quantum_kernel.html).

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
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=qksvm
```

## Usage

The notebooks exemplify the usage of the implemented functions and algorithms.

* [SciKit-Learn moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset is classified using RBF and QKE SVM in `moons-qke.ipynb`.
* Quantum Kernel Training algorithm (QKT-SVM) is applied to the [SciKit-Learn moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) moons dataset in `moons-qkt.ipynb`.
* In `circs-qkt.ipynb`, QKT QSVM algorithm is applied to a more challenging modified version of [SciKit-Learn circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) dataset consisted of 4 diffused folded circles.
* Quantum Feature Map circuit generator can be tested in `test_QFM.ipynb`.
