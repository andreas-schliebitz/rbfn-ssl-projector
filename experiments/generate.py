#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import shutil
import argparse
import itertools

from pathlib import Path


def dict_to_exports(d: dict) -> list[str]:
    exports = []
    for name, value in d.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        exports.append(f"export {name}={value}")
    return exports


def dict_to_args(d: dict) -> list[str]:
    args = []
    for option, value in d.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{option}")
        elif isinstance(value, list):
            if value:
                args.append(f"--{option} {' '.join(map(str, value))}")
        else:
            args.append(f"--{option} {value}")
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate run scripts for a given experiment."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_const",
        const=True,
        help="Generate runs with single train and testing epoch for debugging.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python3.12",
        help="Name of python executable to use.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs",
        help="Path to output run scripts for experiment.",
    )
    parser.add_argument(
        "--slurm",
        default=False,
        action="store_const",
        const=True,
        help="Generate SBATCH headers to use with Slurm Workload Manager.",
    )
    parser.add_argument(
        "--slurm-gres",
        type=str,
        default="gpu:ampere80:1",
        help="Slurm GPU resource (gres) definition",
    )
    parser.add_argument(
        "--slurm-account",
        type=str,
        default="f_agrifood-tef",
        help="Slurm account name",
    )
    parser.add_argument(
        "--slurm-ntasks",
        type=int,
        default=1,
        help="Slurm number of tasks",
    )
    parser.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=8,
        help="Slurm CPUs per task",
    )

    ARGS = parser.parse_args()

    with open(ARGS.config, mode="r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    rootdir = config["rootdir"]
    exports = dict_to_exports(config["env"])

    constant_args = []
    for constants in config["constants"].values():
        if ARGS.debug:
            for epoch_key in ("epochs", "eval-epochs"):
                if epoch_key in constants:
                    constants[epoch_key] = 1
        constant_args += dict_to_args(constants)

    variables = config["variables"]

    ranges = config["ranges"]
    range_options = list(ranges.keys())
    combinations = list(itertools.product(*ranges.values()))

    output = Path(ARGS.output).expanduser().resolve()
    if output.exists() and output.is_dir():
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    RUN_ID = 1
    EXPERIMENT_NAME = config["constants"]["logging"]["experiment-name"]
    for i, combination in enumerate(combinations):
        architecture = combination[0]

        if architecture in {"simsiam", "byol", "swav"}:
            num_layers = combination[1]
            if num_layers > 2:
                print(
                    f"Skipping run with projection head num_layers={num_layers} for architecture '{architecture}'."
                )
                continue

        run_id_str = str(RUN_ID).zfill(5)
        JOB_NAME = f"{EXPERIMENT_NAME}_{run_id_str}"

        projection_head = variables[architecture]["projection-head"]
        train = variables[architecture]["train"]

        args = (
            constant_args
            + dict_to_args({"run-name": run_id_str})
            + dict_to_args(
                {range_options[j]: value for j, value in enumerate(combination)}
            )
            + dict_to_args(projection_head)
            + dict_to_args(train)
        )

        RUN_SCRIPT = ["#!/usr/bin/env bash"]

        if ARGS.slurm:
            RUN_SCRIPT += [
                f"#SBATCH --job-name={JOB_NAME}",
                f"#SBATCH --output={JOB_NAME}.stdout",
                f"#SBATCH --error={JOB_NAME}.stderr",
                "#SBATCH --nodes=1",
                f"#SBATCH --ntasks-per-node={ARGS.slurm_ntasks}",
                f"#SBATCH --cpus-per-task={ARGS.slurm_cpus_per_task}",
                f"#SBATCH --gres={ARGS.slurm_gres}",
                f"#SBATCH --account={ARGS.slurm_account}",
            ]

        RUN_SCRIPT += exports
        RUN_SCRIPT += [f"cd {rootdir} || exit 1"]
        RUN_SCRIPT += ["source venv/bin/activate"]
        RUN_SCRIPT += ["cd rbfn_ssl_projector || exit 2"]
        RUN_SCRIPT += [f"time {ARGS.python} main.py \\\n\t" + " \\\n\t".join(args)]
        RUN_SCRIPT = "\n".join(RUN_SCRIPT)

        run_script = output.joinpath(f"{JOB_NAME}.sh")
        with open(run_script, mode="w", encoding="utf-8") as fh:
            fh.write(RUN_SCRIPT)
        run_script.chmod(0o755)

        RUN_ID += 1
