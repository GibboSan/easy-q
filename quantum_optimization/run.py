import json
import logging
import os
import sys

import yaml

from pipeline.main import single_run

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

    try:

        with open(parameter_path, 'r') as file:
            return yaml.safe_load(file)

    except Exception as e:

        logger.error(f"Error reading parameters file: {e}")
        sys.exit(1)


def main():

    configure_logger()

    if len(sys.argv) != 2:
        logger.error("Usage: python run.py <yaml_parameters_path>")
        sys.exit(1)

    parameter_path = sys.argv[1]

    logger.info(f"reading parameters file: '{parameter_path}'")
    parameter_dict = read_parameters(parameter_path)

    output_folder = parameter_dict['output_folder']

    output_dict = single_run(parameter_dict)

    logger.info(f"writing output file: {output_folder}/output.json")
    with open(f"{output_folder}/output.json", "w") as f:
        json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
