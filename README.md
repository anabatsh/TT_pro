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
    pip install numpy teneva==0.12.8 ttopt==0.5.0 jax optax equinox qubogen gekko nevergrad torch
    ```

5. Clean temporary dir after runs:
    ```bash
    find /tmp -type d -maxdepth 1 -iname "*model*" -exec rm -fr {} \;
    ```

6. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tt_pro --all -y
    ```


## Usage

Please, see our [colab notebook](https://colab.research.google.com/drive/1W36LHd9Rm1R4xi-wGFSXtzSH2iF4h_4y?usp=sharing) with various examples.


## Authors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Ivan Oseledets](https://github.com/oseledets)
- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{batsheva2023protes,
    author    = {Batsheva, Anastasia and Chertkov, Andrei  and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {PROTES: Probabilistic Optimization with Tensor Sampling},
    journal   = {arXiv preprint arXiv:2301.12162},
    url       = {https://arxiv.org/pdf/2301.12162.pdf}
}
```
