# TT_pro


## Description

Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization of the multidimensional arrays and  discretized multivariable functions based on the tensor train (TT) format.


## Installation

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name tt_pro python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate tt_pro
    ```

4. Install dependencies:
    ```bash
    pip install numpy teneva==0.12.8 ttopt==0.5.0 jax optax equinox qubogen dwave-neal gekko nevergrad
    ```

5. Clean temporary dir after runs:
    ```bash
    find /tmp -type d   -maxdepth 1  -iname "*model*"  -exec rm -fr {} \;
    ```

6. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tt_pro --all -y
    ```


## Usage

1. Run `python calc.py control` to solve the optimal control problem. The results will be presented in the text file `result/logs/control.txt`.

2. Run `python calc.py func` to perform computations for various analytical multivariable functions. The results will be presented in the text file `result/logs/func.txt`.

3. Run `python calc.py qubo` to solve the QUBO problem. The results will be presented in the text file `result/logs/qubo.txt`.


## Authors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Ivan Oseledets](https://github.com/oseledets)
- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
