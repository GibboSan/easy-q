import os
import json
import pandas as pd
import sys
import re



def get_folder(path, root_dir):
    """
    Extract the dataset folder name from the full path.
    
    Args:
        path (str): Full path to the experiment directory
        root_dir (str): Root directory of experiments
        
    Returns:
        str: Folder name (e.g., 'synt_1000_2_0.6')
    """
    rel_path = os.path.relpath(path, root_dir)
    return rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path


def get_record_from_output(out_data, method_type=None):
    """
    Extract relevant metrics from output JSON data.
    
    Args:
        out_data (dict): Parsed JSON output from experiment
        method_type (str, optional): Type of quantum method used
        
    Returns:
        dict: Dictionary containing extracted metrics
    """

    return {
        'seed': out_data.get('seed'),
        'method_type': out_data.get('method_type', method_type).upper() if isinstance(out_data.get('method_type', method_type), str) else None,
        'feature_map': out_data.get('feature_map'),
        'feature_map_reps': out_data.get('feature_map_params', {}).get('reps'),
        'feature_map_entanglement': out_data.get('feature_map_params', {}).get('entanglement'),

        'ansatz': out_data.get('ansatz'),
        'ansatz_reps': out_data.get('ansatz_params', {}).get('reps'),
        'ansatz_entanglement': out_data.get('ansatz_params', {}).get('entanglement'),
        'opt_method': out_data.get('opt_method'),
        'opt_method_tol': out_data.get('opt_method_params', {}).get('tol'),
        'opt_method_maxiter': out_data.get('opt_method_params', {}).get('maxiter'),

        'n_qubit_feat_map': out_data.get('n_qubit_feat_map'),
        'circuit_depth_feat_map': out_data.get('circuit_depth_feat_map'),
        'circuit_n_gate_feat_map': out_data.get('circuit_n_gate_feat_map'),
        'circuit_n_cnot_feat_map': out_data.get('circuit_n_cnot_feat_map'),
        'n_qubit_ansatz': out_data.get('n_qubit_ansatz'),
        'circuit_depth_ansatz': out_data.get('circuit_depth_ansatz'),
        'circuit_n_gate_ansatz': out_data.get('circuit_n_gate_ansatz'),
        'circuit_n_cnot_ansatz': out_data.get('circuit_n_cnot_ansatz'),

        'train_score_q': out_data.get('quantum_results', out_data.get(method_type, {})).get('train_score'),
        'test_score_q': out_data.get('quantum_results', out_data.get(method_type, {})).get('test_score'),
        'walltime_kernel_train_q': out_data.get('quantum_results', {}).get('walltime_kernel_train'),
        'walltime_kernel_test_q': out_data.get('quantum_results', {}).get('walltime_kernel_test'),
        'walltime_fit_q': out_data.get('quantum_results', out_data.get(method_type, {})).get('walltime'),

        'train_score_svc': out_data.get('classic_SVC', {}).get('train_score'),
        'test_score_svc': out_data.get('classic_SVC', {}).get('test_score'),
        'walltime_svc': out_data.get('classic_SVC', {}).get('walltime')
    }


def def_output_columns():
    """
    Define the standard column order for output DataFrames.
    
    Returns:
        list: Ordered list of column names
    """
    return [
        'folder',
        'hardware',
        'dataset', 
        'samples', 'features', 'separability', 'dim_red_method',
        'method_type', 
        'feature_map', 'feature_map_reps', 'feature_map_entanglement',
        'ansatz', 'ansatz_reps', 'ansatz_entanglement', 
        'opt_method', 'opt_method_tol', 'opt_method_maxiter',
        'train_score_q', 'test_score_q', 
        'walltime_kernel_train_q', 'walltime_kernel_test_q', 'walltime_fit_q',
        'walltime_q', 'walltime_q_h', 
        'train_score_svc', 'test_score_svc', 
        'walltime_svc', 'walltime_svc_h'
    ]



def extract_all_output_json_data(root_dir):
    """
    Walk through experiment directory tree and extract all output.json data.
    
    Args:
        root_dir (str): Root directory containing experiment outputs
        
    Returns:
        pd.DataFrame: Aggregated results from all experiments
    """
    records = []
    columns_list = def_output_columns()

    # Determine hardware type from parent directory
    dir_hardware = os.path.basename(os.path.dirname(root_dir))
    if dir_hardware == 'output_local':
        hardware = 'Local'
    elif dir_hardware == 'output_HPC':
        hardware = 'HPC'
    else:
        hardware = None

    for subdir, _, files in os.walk(root_dir):       
        if 'output.json' in files:
            json_path = os.path.join(subdir, 'output.json')
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Handle legacy format from local simulations
                if "qsvc" in data:
                    method_type = "qsvc"
                elif "vqc" in data:
                    method_type = "vqc"
                else:
                    method_type= None

                record = get_record_from_output(data, method_type)
                record["folder"] = os.path.basename(root_dir)
                record["dataset"] = get_folder(subdir, root_dir)

                # Calculate total quantum walltime
                record['walltime_q'] = (
                    float(record['walltime_kernel_train_q'] or 0) + 
                    float(record['walltime_kernel_test_q'] or 0) + 
                    float(record['walltime_fit_q'] or 0)
                )

                # Parse dataset parameters from folder name
                match_synt = re.match(r'synt_(\d+)_(\d+)_(\d+\.?\d*)', record['dataset'])
                match_unsw_nb15 = re.match(r'unsw_nb15_(\d+)_([a-zA-Z]+)_(\d+)(?:_balanced)?', record['dataset'])
                match_breast_cancer = re.match(r'Breast_cancer_([a-zA-Z]+)_(\d+)', record['dataset'])


                if match_synt:
                    record['samples'] = int(match_synt.group(1))
                    record['features'] = int(match_synt.group(2))
                    record['separability'] = float(match_synt.group(3))
                    record['dim_red_method'] = '-'

                elif match_unsw_nb15:
                    record['samples'] = int(match_unsw_nb15.group(1))
                    record['features'] = int(match_unsw_nb15.group(3))
                    record['separability'] = 0
                    record['dim_red_method'] = match_unsw_nb15.group(2)

                elif record['dataset'] == 'Iris_dataset':
                    record['samples'] = 150
                    record['features'] = 4
                    record['separability'] = 0
                    record['dim_red_method'] = '-'

                elif record['dataset'] == 'Wine_dataset':
                    record['samples'] = 178
                    record['features'] = 13
                    record['separability'] = 0
                    record['dim_red_method'] = '-'

                elif match_breast_cancer:
                    record['samples'] = 568
                    record['features'] = int(match_breast_cancer.group(2))
                    record['separability'] = 0
                    record['dim_red_method'] = match_breast_cancer.group(1)
                
                elif record['dataset'] == 'Breast_cancer_dataset':
                    record['samples'] = 568
                    record['features'] = 30
                    record['separability'] = 0
                    record['dim_red_method'] = '-'

                else:
                    record['samples'] = None
                    record['features'] = None
                    record['separability'] = None
                    record['dim_red_method'] = '-'

                records.append(record)

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    df = pd.DataFrame(records)

    # Convert walltime to hours
    df["walltime_q_h"] = df["walltime_q"] / 3600
    df["walltime_svc_h"] = df["walltime_svc"] / 3600

    # Add hardware column
    df['hardware'] = hardware

    return df[columns_list]


def translate_classical_results(df):
    """
    Aggregate classical SVC results and append them as separate rows.
    
    Args:
        df (pd.DataFrame): DataFrame with quantum and classical results
        
    Returns:
        pd.DataFrame: DataFrame with appended aggregated classical results
    """
    # Define grouping columns
    group_cols = ['dataset', 'samples', 'features', 'separability', 'dim_red_method']

    # Define aggregation for classical columns
    agg_cols = {
        'train_score_svc': 'mean',
        'test_score_svc': 'mean',
        'walltime_svc': 'max',
        'walltime_svc_h': 'max'
    }
    
    # Column renaming for consistency
    rename_cols_dict = {
        'train_score_svc': 'train_score',
        'test_score_svc': 'test_score',
        'walltime_svc': 'walltime',
        'walltime_svc_h': 'walltime_h',
        'train_score_q': 'train_score',
        'test_score_q': 'test_score',
        'walltime_q': 'walltime',
        'walltime_q_h': 'walltime_h'
    }

    # Aggregate classical results by dataset
    agg_df = df.groupby(group_cols).agg(agg_cols).reset_index()
    agg_df = agg_df.rename(columns=rename_cols_dict)

    # Remove classical columns from main df and rename quantum columns
    df = df.drop(columns=agg_cols.keys()).rename(columns=rename_cols_dict)

    # Fill missing columns in aggregated classical results
    for col in df.columns:
        if col not in agg_df.columns:
            if col == 'method_type':
                agg_df[col] = 'SVC'
            elif col == 'folder':
                agg_df[col] = df[col].unique()[0]
            elif col == 'hardware':
                agg_df[col] = df[col].unique()[0]
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_df[col] = 0
            elif pd.api.types.is_string_dtype(df[col]):
                agg_df[col] = '-'
            else:
                agg_df[col] = None  
    
    # Concatenate quantum and classical results
    df_final = pd.concat([df, agg_df], ignore_index=True)

    return df_final


def main():
    """Main execution function for command-line usage."""
    if len(sys.argv) != 3:
        print("Usage: python aggregate_results.py <experimental_output_folder_path> <csv_experimental_report_path>")
        sys.exit(1)

    experimental_folder_path = sys.argv[1]
    output_ds_folder = sys.argv[2]

    csv_file_name = os.path.basename(experimental_folder_path)
    print(f"Processing: {csv_file_name}")
    dataset_path = f"{output_ds_folder}/{csv_file_name}.csv"

    os.makedirs(output_ds_folder, exist_ok=True)

    output_df = extract_all_output_json_data(experimental_folder_path)
    output_df = translate_classical_results(output_df)

    print(f"Writing output file: {dataset_path}")
    output_df.to_csv(dataset_path, index=False)


if __name__ == "__main__":
    main()