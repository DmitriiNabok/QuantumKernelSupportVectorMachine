from setuptools import setup, find_packages

setup(
    name='qksvm',
    version='0.1.0',
    packages=find_packages(include=['qksvm', 'qksvm.*']),
    install_requires=[
        'qiskit',
        'qiskit-machine-learning',
        'scikit-learn',
        'pandas',
        'matplotlib',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

