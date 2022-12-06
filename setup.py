from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='qksvm',
    version='0.3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
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

