import argparse
import itertools
import json
import os
from datetime import datetime
import pickle
import sys
import yaml

from qiskit_algorithms.utils import algorithm_globals

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.datasets import create_dataset


def generate_combinations_method_dets(methods_params_dict):
# def generate_combinations_method_dets(methods_params_dict: Dict[str, List[Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate all combinations of the values in the dictionary and return them as
    (formatted_string, param_dict) pairs.

    Args:
        methods_params_dict (dict): Dictionary with parameter names as keys and lists of possible values.

    Returns:
        list: List of tuples (formatted_string, param_dict).
    """
    keys = list(methods_params_dict.keys())
    values_lists = list(methods_params_dict.values())

    combinations = itertools.product(*values_lists)

    result = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        formatted_string = "/".join(f"{val}_{key}" for key, val in param_dict.items())
        result.append((formatted_string, param_dict))

    return result

def load_file(file_path: str) -> dict:

    if not os.path.isfile(file_path):
        print(f"Error: file '{file_path}' does not exist.")
        sys.exit(1)

    SUPPORTED_FORMATS = ('.yaml', '.yml', '.json')

    if not file_path.endswith(SUPPORTED_FORMATS):
        print(f"Unsupported file format. Use one of {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    try:
          
        with open(file_path, 'r') as file:
            if file_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            else:
                return json.load(file)

    except Exception as e:

        print(f"Error reading parameters file '{file_path}': {e}")
        sys.exit(1)


def create_parameters_file(variation_dict):
    
    method = variation_dict.get('quantum_methods')

    if method not in ['QSVC', 'VQC']:
        raise ValueError("Unsupported method. Use 'QSVC' or 'VQC'.")
    
    experimental_parameter_dict = {
        "seed": variation_dict.get('seed', 42),
        "output_path": os.path.realpath(variation_dict.get('output_folder_path', './output')),
        "backend": variation_dict.get('backend', 'aer'),
        "method_type": method,
        "feature_map": variation_dict.get('feature_map', 'ZZFeatureMap'),
        "feature_map_params": {
            "reps": variation_dict['methods_params'].get('reps_map', 2),
            "entanglement": variation_dict['methods_params'].get('entangl_map', 'full')
        },
    }
    
    if method == 'VQC':
        ansatz_dict = {
            "ansatz": variation_dict.get('ansatz', 'RealAmplitudes'),
            "ansatz_params": {"reps": variation_dict['methods_params'].get('reps_ansatz', 2),
                              "entanglement": variation_dict['methods_params'].get('entangl_ansatz', 'full')
                             }
                    }

        optimizer_dict = {
            "opt_method": variation_dict['methods_params'].get('opt_method', 'SLSQP'),
            "opt_method_params": {"maxiter": variation_dict['methods_params'].get("maxiter", 1000)
                            },
            "plot": variation_dict.get('plot', True)
                    }
        
        experimental_parameter_dict = {**experimental_parameter_dict, **ansatz_dict, **optimizer_dict}
    
    return experimental_parameter_dict

def save_parameters_to_yaml(parameter_dict: dict, experiment_output_path: str) -> str:
    
    parameter_path = os.path.join(experiment_output_path, "experimental_parameter.yaml")

    try:
        with open(parameter_path, "w") as f:
            yaml.safe_dump(parameter_dict, f, sort_keys=False)
        print(f"Parameters saved to {parameter_path}")
    except Exception as e:
        print(f"Failed to save parameters to YAML: {e}")
        raise  # oppure: sys.exit(1)

    return parameter_path


def create_dataset_combinations(dataset_type, dataset_params, file_path=None):
    """
    Create dataset combinations using the unified dataset interface.
    
    Args:
        dataset_type (str): Type of dataset
        dataset_params (dict): Parameters for dataset creation
        file_path (str, optional): Path to CSV file for real datasets
    
    Returns:
        list: List of dataset dictionaries with metadata
    """
    combinations = itertools.product(*list(dataset_params.values()))
    keys = list(dataset_params.keys())

    datasets_feats_list = []
    
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        
        # Create dataset using unified interface
        splitted_dataset = create_dataset(
            dataset_type=dataset_type,
            dataset_params=combo_dict,
            file_path=file_path,
            seed=algorithm_globals.random_seed
        )
        
        # Generate dataset name based on type
        if dataset_type in ["synthetic", "synthetic_red"]:
            name_parts = [f"synt"] + [f"{val}" for val in combo_dict.values()]
        else:
            name_parts = [f"{val}" for val in combo_dict.values()]
        
        combo_dict["name"] = "_".join(str(part) for part in name_parts)
        combo_dict["splitted_dataset"] = splitted_dataset
        datasets_feats_list.append(combo_dict)

    return datasets_feats_list


def build_slurm_args(slurm_dict):
    return " ".join(f"--{k}={v}" for k, v in slurm_dict.items())

def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", required=True, help="Path al template YAML")
    parser.add_argument("--configuration_runs", required=True, help="Path al file json sulle configurazioni delle variazioni da esguire")
    parser.add_argument("--python_exec", required=True, help="Eseguibile Python (es: python3)")
    parser.add_argument("--script", required=True, help="Script Python da lanciare (es: run.py)")
    parser.add_argument("--slurm_options", required=False, help="File YAML con opzioni SLURM (facoltativo)")
    args = parser.parse_args()

    # File di configurazione con i parametri
    configuration_runs = load_file(args.configuration_runs)

    ##############################################################################################
    algorithm_globals.random_seed = configuration_runs["seed"]
    ##############################################################################################

    output_folder_path = configuration_runs["output_folder_path"]
    method = configuration_runs["quantum_methods"]

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


    # Creazione della cartella principale per gli esperimenti
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # exp_root = f"exp_{timestamp}"
    exp_root = os.path.join(output_folder_path, f"ML_exp_{timestamp}_{method}")
    os.makedirs(exp_root, exist_ok=True)

    bash_lines = ["#!/bin/bash\n", "set -e\n\n"]
    # slurm_lines = ["#!/bin/bash\n", "set -e\n\n", "mkdir -p logs\n\n"]
    slurm_lines = ["#!/bin/bash\n", "set -e\n\n"]

    
    ##################################
    #### Datasets list
    ##################################
    datasets_feats_list = create_dataset_combinations(
                                        dataset_type=configuration_runs["dataset_type"], 
                                        dataset_params=configuration_runs["datasets_params"], 
                                        file_path=configuration_runs.get('file_path')
                                    )
    
    ##################################
    #### Create a list of method parameter combinations
    ##################################

    directory_list = generate_combinations_method_dets(configuration_runs["methods_params"])

    ##########################################################
    #### Loop through Datasets
    ##########################################################
    for dataset_dict in datasets_feats_list:
        dataset_name = dataset_dict["name"]
        print(f"Dataset: {dataset_name}")

        # Create a directory for the dataset
        dataset_dict["dataset_path"] = os.path.join(exp_root, dataset_name)
        os.makedirs(dataset_dict["dataset_path"], exist_ok=True)

        # Save the dataset dictionary in pkl file
        dataset_dict_path = os.path.join(dataset_dict["dataset_path"], f"{dataset_name}_dataset.pkl")
        with open(dataset_dict_path, "wb") as f:
            pickle.dump(dataset_dict["splitted_dataset"], f)

        print(f"Dataset saved to: {dataset_dict_path}")

        # Creating folders concatenated with configuration variations
        for dir in directory_list:
            # Create the parameter combination folder
            experiment_output_path = os.path.join(dataset_dict["dataset_path"], dir[0])
            os.makedirs(experiment_output_path, exist_ok=True) 
            
            print(f"Output path is set to {experiment_output_path}")
            conf_dict =  configuration_runs.copy()
            conf_dict["output_folder_path"] = experiment_output_path
            conf_dict["methods_params"] = dir[1]
            parameter_dict = create_parameters_file(conf_dict)

            # Save the yaml file and memorize the path
            parameter_path = save_parameters_to_yaml(parameter_dict, experiment_output_path)

            # command for the single execution
            command = f"{args.python_exec} {os.path.relpath(args.script, exp_root)} {os.path.relpath(parameter_path, exp_root)} {os.path.relpath(dataset_dict_path, exp_root)} > {os.path.relpath(experiment_output_path, exp_root)}/output.log\n"

            bash_lines.append(command)

            slurm_lines.append(
                f'sbatch --job-name={os.path.relpath(experiment_output_path, exp_root)} {slurm_args_str} '
                f'--output={os.path.join(os.path.relpath(experiment_output_path, exp_root), "log_%j.out")} '
                f'--wrap="{command}"\n'
            )

    bash_path = os.path.join(exp_root, "run_all.sh")
    with open(bash_path, "w") as f:
        f.writelines(bash_lines)
    os.chmod(bash_path, 0o755)

    slurm_path = os.path.join(exp_root, "run_all_slurm.sh")
    with open(slurm_path, "w") as f:
        f.writelines(slurm_lines)
    os.chmod(slurm_path, 0o755)

    print(f"Script bash generated: {bash_path}")
    print(f"Script SLURM generated: {slurm_path}")
    print(f"\nTo execute:\n  cd {exp_root} && ./run_all.sh       # sequential\n                    ./run_all_slurm.sh  # parallel via SLURM")

if __name__ == "__main__":
    main()
