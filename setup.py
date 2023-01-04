from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='qksvm',
    version='0.3',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=[
        'qiskit<=0.35.*',
        'qiskit-machine-learning<=0.4.*',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'pytest',
    ],
)

