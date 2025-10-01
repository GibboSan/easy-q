import json
import logging
import os
import sys
import pickle
import yaml
import numpy as np

import pandas as pd

from pipeline.main import single_run
from pipeline.plots import plot_obj_func

logger = logging.getLogger("pipeline_logger")


def configure_logger():

    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)


def read_parameters(parameter_path: str) -> dict:

    if not os.path.isfile(parameter_path):
        logger.error(f"Error: file '{parameter_path}' does not exist.")
        sys.exit(1)

    SUPPORTED_FORMATS = ('.yaml', '.yml', '.json')

    if not parameter_path.endswith(SUPPORTED_FORMATS):
        logger.error(f"Unsupported file format. Use one of {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    try:
          
        with open(parameter_path, 'r') as file:
            if parameter_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            else:
                return json.load(file)

    except Exception as e:
        logger.error(f"Error reading parameters file '{parameter_path}': {e}")
        sys.exit(1)

        


def read_dataset(dataset_path: str) -> dict:

    if not os.path.isfile(dataset_path):
        logger.error(f"Error: file '{dataset_path}' does not exist.")
        sys.exit(1)

    SUPPORTED_FORMATS = ('.pkl')

    if not dataset_path.endswith(SUPPORTED_FORMATS):
        logger.error(f"Unsupported file format. Use one of {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    try:

        with open(dataset_path, 'rb') as file:
            return pickle.load(file)

    except Exception as e:

        logger.error(f"Error reading parameters file '{dataset_path}': {e}")
        sys.exit(1)


def main():

    configure_logger()

    if len(sys.argv) != 3:
        logger.error("Usage: python run.py <yaml_parameters_path> <pkl_dataset_path>")
        sys.exit(1)

    parameter_path = sys.argv[1]

    logger.info(f"reading parameters file: '{parameter_path}'")
    parameter_dict = read_parameters(parameter_path)

    output_folder = parameter_dict['output_path']

    dataset_path = sys.argv[2]

    logger.info(f"reading dataset: '{dataset_path}'")
    dataset_dict = read_dataset(dataset_path)

    # Run the fit
    output_dict, trained_model = single_run(parameter_dict, dataset_dict)

    logger.info(f"writing output file: {output_folder}/output.json")
    with open(f"{output_folder}/output.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    # Save fit model
    try:
        logger.info(f"Saving fit model: {output_folder}/classifier.model")
        trained_model["model"].save(f"{output_folder}/classifier.model")
        
    except Exception as e:
        logger.error(f"Unable to save the model to '{output_folder}/classifier.model': {e}")

    if "kernel" in trained_model:

        k_train = trained_model["kernel"]["kernel_train"]
        k_test = trained_model["kernel"]["kernel_test"]

        k_train_path = f"{output_folder}/kernel_train.npy"
        k_test_path = f"{output_folder}/kernel_test.npy"

        logger.info(f"Save Kernel Train: {k_train_path}")
        np.save(k_train_path, k_train)

        logger.info(f"Save Kernel Test: {k_test_path}")
        np.save(k_test_path, k_test)

    if "objective_func" in trained_model:

        obj_func = trained_model["objective_func"]
        obj_func_csv_path = f"{output_folder}/objective_func.csv"
        logger.info(f"Save Objective Function: {obj_func_csv_path}")
        obj_func_df = pd.DataFrame({
            "iteration": range(len(obj_func)),  # 0,1,2,...
            "value": obj_func
        })
        obj_func_df.to_csv(obj_func_csv_path, index=False)

        obj_func_plot_path = f"{output_folder}/object_function.png"
        logger.info(f"Save Objective Function Plot: {obj_func_plot_path}")
        plot_obj_func(obj_func, obj_func_plot_path)


if __name__ == "__main__":
    main()
