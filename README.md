# TT_pro


## Description

Method TT-PRO (Tensor Train PRobability Optmizer) for optimization of the multidimensional arrays and  discretized multivariable functions based on the tensor train (TT) format.


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
    pip install numpy teneva==0.12.8 jax optax equinox
    ```

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name NAME --all -y
    ```


## Usage

1. Run `python calc.py test`
    > The results will be presented in the text file `result/logs/calc_test.txt`


## Authors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)
