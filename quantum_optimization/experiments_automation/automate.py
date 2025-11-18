import argparse
import copy
import itertools
import json
import os
from collections import OrderedDict
from datetime import datetime

import yaml


def load_file(path):
    with open(path, 'r') as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Formato file non supportato (usa .yaml o .json)")

def set_nested_value(d, key_path, value):
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def deep_ordered_copy(d):
    if isinstance(d, dict):
        return OrderedDict((k, deep_ordered_copy(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [deep_ordered_copy(i) for i in d]
    else:
        return d

def ordered_dict_to_dict(d):
    if isinstance(d, OrderedDict):
        return {k: ordered_dict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: ordered_dict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [ordered_dict_to_dict(i) for i in d]
    else:
        return d

def build_slurm_args(slurm_dict):
    return " ".join(f"--{k}={v}" for k, v in slurm_dict.items())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="Path al template YAML")
    parser.add_argument("--variations", required=True, help="Path al file YAML delle variazioni")
    parser.add_argument("--python_exec", required=True, help="Eseguibile Python (es: python3)")
    parser.add_argument("--script", required=True, help="Script Python da lanciare (es: run.py)")
    parser.add_argument("--slurm_options", required=False, help="File YAML con opzioni SLURM (facoltativo)")
    args = parser.parse_args()

    template = load_file(args.template)
    variations_raw = load_file(args.variations)
    template = deep_ordered_copy(template)

    # Slurm default fallback
    default_slurm_opts = {
        "mem": "2G",
        "cpus-per-task": "1",
        "time": "01:00:00"
    }

    if args.slurm_options:
        slurm_opts = load_file(args.slurm_options)
        # Merge con default (override con opzioni da file)
        slurm_opts = {**default_slurm_opts, **slurm_opts}
    else:
        slurm_opts = default_slurm_opts

    slurm_args_str = build_slurm_args(slurm_opts)

    param_keys = list(variations_raw.keys())

    value_lists = []
    labels = []

    for k in param_keys:
        entry = variations_raw[k]
        if not isinstance(entry, dict) or "values" not in entry or "label" not in entry:
            raise ValueError(f"Parametro '{k}' deve avere chiavi 'values' e 'label'.")
        value_lists.append(entry["values"])
        labels.append((k, entry["label"]))

    combinations = list(itertools.product(*value_lists))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = f"exp_{timestamp}"
    os.makedirs(exp_root, exist_ok=True)

    bash_lines = ["#!/bin/bash\n", "set -e\n\n"]
    slurm_lines = ["#!/bin/bash\n", "set -e\n\n", "mkdir -p logs\n\n"]

    for combo in combinations:
        config = copy.deepcopy(template)
        label_parts = []

        for (key, label_prefix), value in zip(labels, combo):
            set_nested_value(config, key, value)
            label_parts.append(f"{label_prefix}{value}")

        label = "_".join(label_parts)
        run_dir = os.path.join(exp_root, label)
        os.makedirs(run_dir, exist_ok=True)

        output_folder_name = label + "_out"
        output_path = os.path.join(exp_root, output_folder_name)
        rel_output_path = os.path.relpath(output_path, exp_root).replace(os.sep, "/")

        config["output_folder"] = rel_output_path
        config["cache_filename"] = f"{rel_output_path}/cache.yaml"
        config_path = os.path.join(run_dir, f"{label}.yaml")
        rel_config_path = os.path.relpath(config_path, exp_root).replace(os.sep, "/")

        with open(config_path, "w") as out_file:
            yaml.dump(ordered_dict_to_dict(config), out_file, sort_keys=False)

        print(f"Creato: {config_path}")

        bash_lines.append(
            f"mkdir -p {rel_output_path} && {args.python_exec} {args.script} {rel_config_path} > {rel_output_path}/output.log\n"
        )

        slurm_lines.append(
            f'sbatch --job-name={label} {slurm_args_str} '
            f'--output=logs/{label}_%j.out '
            f'--wrap="mkdir -p {rel_output_path} && {args.python_exec} {args.script} {rel_config_path} > {rel_output_path}/output.log"\n'
        )

    bash_path = os.path.join(exp_root, "run_all.sh")
    with open(bash_path, "w") as f:
        f.writelines(bash_lines)
    os.chmod(bash_path, 0o755)

    slurm_path = os.path.join(exp_root, "run_all_slurm.sh")
    with open(slurm_path, "w") as f:
        f.writelines(slurm_lines)
    os.chmod(slurm_path, 0o755)

    print(f"Script bash generato: {bash_path}")
    print(f"Script SLURM generato: {slurm_path}")
    print(f"\nPer eseguire:\n  cd {exp_root} && ./run_all.sh       # sequenziale\n                    ./run_all_slurm.sh  # parallelo via SLURM")

if __name__ == "__main__":
    main()
