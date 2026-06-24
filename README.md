# Radial Basis Function Networks as Projection Heads in Self-Supervised Learning

This repository contains the main Python code for reproducing our experiments conducted as part of our paper

> [Radial Basis Function Networks as Projection Heads in Self-Supervised Learning](https://arxiv.org/abs/2606.21590)

by Andreas Schliebitz, Heiko Tapken and Martin Atzmueller.

As seen in [pyproject.toml](pyproject.toml), the code in this repository depends on the following external repositories, which are also maintained and open-sourced by the authors of this paper:

* SimSiam: <https://github.com/andreas-schliebitz/simsiam>
* MoCo (v2): <https://github.com/andreas-schliebitz/moco>
* BYOL: <https://github.com/andreas-schliebitz/byol>
* SimCLR: <https://github.com/andreas-schliebitz/simclr>
* SwAV: <https://github.com/andreas-schliebitz/swav>
* RBFN Projection Head: <https://github.com/andreas-schliebitz/rbfn-head>
* Logistic Regression: <https://github.com/andreas-schliebitz/logreg>
* Vision Datasets: <https://github.com/andreas-schliebitz/vision-datasets>

## Install

1. Create a Python virtual environment:

    ```bash
    python3 -m venv venv
    ```

2. Run the `install.sh` script:

    ```bash
    ./install.sh
    ```

## Run Experiments

1. Change into the experiments directory:

    ```bash
    cd experiments
    ```

2. Generate run scripts from YAML [configs](./experiments/configs):

    ```bash
    ./generate.sh
    ```

3. Execute generated run scripts:

    ```bash
    ./run-experiments.sh
    ```

Train and evaluation results as well as model artifacts will always be logged into a directory called `logs/csv` in [rbfn_ssl_projector](./rbfn_ssl_projector). In our experiments, we also configure optional but recommended [MLflow](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html) logging.

## Examine Results

The results published in our paper are based on MLflow CSV exports which can be found within the [results](./results) directory.
