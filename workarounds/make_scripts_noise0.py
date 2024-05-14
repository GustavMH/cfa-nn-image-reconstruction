#!/usr/bin/env python3
from pathlib   import Path

home       = Path("/home/zvq211")
ds_path    = home / "ds_160_log"
model_path = home / "models"
prog_path  = home / "cfa-raw" / "notebooks" / "train.py"


def write_script(form, ext, path, n):
    name     = f"noise0_{path}_{n}"
    filename = f"{name}.sh"
    commands = [
        ["#!/usr/bin/env bash"],
        ["#SBATCH --ntasks=1 --gres=gpu:titanrtx --cpus-per-task=4 --mem=12000M --time=0:40:00"],
        [f"source {home / 'cfa-raw'}/.venv/bin/activate"],
        [
            f"python3.11 {prog_path}",
            f"--clean {ds_path}/png/none/0pct/",
            f"--noise {ds_path}/{path}",
            f"--output {model_path}",
            f"--model {model_path}/{name}-model.pkl",
            f"--name {name}",
            f"--type .{ext}",
            f"--epochs 300"
        ]
    ]

    with open(filename, "w") as f:
        f.write("\n".join([" ".join(args) for args in commands]))

for path in ["m", "mw", "w", "d"]:
    for i in range(4):
        for n in range(5):
            write_script(form, ext, f"{path}{i}", n)
