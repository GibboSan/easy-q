from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap

# =============================================================================
# DATASET SOURCES
# =============================================================================

TOY_DATASETS = {"Iris": load_iris(),
                "Wine": load_wine(),
                "Breast_cancer": load_breast_cancer()}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_scaling(X, scaler_type="minmax"):
    """
    Apply scaling to features.
    
    Args:
        X (array-like): Feature matrix
        scaler_type (str): Type of scaler ('minmax' or 'standard')
    
    Returns:
        np.ndarray: Scaled features
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()         # utilizzato per i toys dataset
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    return scaler.fit_transform(X)


def apply_dimensionality_reduction(X, reduction_type, n_components, seed=42):
    """
    Apply dimensionality reduction to features.
    
    Args:
        X (array-like): Feature matrix
        reduction_type (str): Type of reduction ('pca' or 'umap')
        n_components (int): Number of components
        seed (int): Random seed
    
    Returns:
        np.ndarray: Reduced features
    """
    if reduction_type == "pca":
        reducer = PCA(n_components=n_components, random_state=seed)
    elif reduction_type == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        raise ValueError("reduction_type must be 'pca' or 'umap'")
    
    X_reduced = reducer.fit_transform(X)
    print(f"Shape of {reduction_type.upper()} transformed data: {X_reduced.shape}")
    return X_reduced


def perform_train_test_split(X, y, train_size=0.8, seed=42):
    """
    Perform train-test split with stratification.
    
    Args:
        X (array-like): Features
        y (array-like): Labels
        train_size (float): Proportion of training data
        seed (int): Random seed
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return {
        "train_features": X_train,
        "test_features": X_test,
        "train_labels": y_train,
        "test_labels": y_test
    }


def encode_categorical_features(X):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        X (pd.DataFrame): Feature DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with encoded categorical features
    """
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=["object"]).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
    
    return X_encoded


# =============================================================================
# TOY DATASETS
# =============================================================================

def get_toy_dataset(dataset_name, seed=42, train_size=0.8):
    """
    Load and prepare toy datasets (Iris, Wine, Breast Cancer).
    
    Args:
        dataset_name (str): Name of the dataset
        seed (int): Random seed
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    if dataset_name not in TOY_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(TOY_DATASETS.keys())}")
    
    dataset = TOY_DATASETS[dataset_name]
    X = dataset.data
    y = dataset.target
    
    # Scale features
    X_scaled = apply_scaling(X, scaler_type="minmax")
    
    # Split data
    return perform_train_test_split(X_scaled, y, train_size, seed)


def get_toy_dataset_with_reduction(dataset_name, reduction_type, n_components, seed=42, train_size=0.8):
    """
    Load toy dataset and apply dimensionality reduction.
    
    Args:
        dataset_name (str): Name of the toy dataset
        reduction_type (str): Type of reduction ('pca' or 'umap')
        n_components (int): Number of components for reduction
        seed (int): Random seed
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    if dataset_name not in TOY_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(TOY_DATASETS.keys())}")
    
    dataset = TOY_DATASETS[dataset_name]
    X = dataset.data
    y = dataset.target
    
    # Scale first, then reduce
    X_scaled = apply_scaling(X, scaler_type="minmax")
    X_reduced = apply_dimensionality_reduction(X_scaled, reduction_type, n_components, seed)
    
    # Split data
    return perform_train_test_split(X_reduced, y, train_size, seed)


# =============================================================================
# SYNTHETIC DATASETS
# =============================================================================

def get_synthetic_dataset(sample_size, n_features, seed=42, class_sep=1.0, 
                         n_redundant=0, train_size=0.8):
    """
    Generate synthetic classification dataset.
    
    Args:
        sample_size (int): Number of samples
        n_features (int): Number of features
        seed (int): Random seed
        class_sep (float): Class separation difficulty
        n_redundant (int): Number of redundant features
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    X, y = make_classification(
        n_samples=sample_size, 
        n_features=n_features,
        class_sep=class_sep,
        n_redundant=n_redundant,
        random_state=seed
    )
    
    # Scale features
    X_scaled = apply_scaling(X, scaler_type="minmax")
    
    # Split data
    return perform_train_test_split(X_scaled, y, train_size, seed)

def get_synthetic_dataset_with_reduction(sample_size, n_features, reduction_type, 
                                       n_components, seed=42, class_sep=1.0, 
                                       n_redundant=0, train_size=0.8):
    """
    Generate synthetic dataset with dimensionality reduction.
    
    Args:
        sample_size (int): Number of samples
        n_features (int): Original number of features
        reduction_type (str): Type of reduction ('pca' or 'umap')
        n_components (int): Number of components after reduction
        seed (int): Random seed
        class_sep (float): Class separation difficulty
        n_redundant (int): Number of redundant features
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    X, y = make_classification(
        n_samples=sample_size, 
        n_features=n_features,
        class_sep=class_sep,
        n_redundant=n_redundant,
        random_state=seed
    )
    
    # Scale first, then reduce
    X_scaled = apply_scaling(X, scaler_type="minmax")
    X_reduced = apply_dimensionality_reduction(X_scaled, reduction_type, n_components, seed)
    
    # Split data
    return perform_train_test_split(X_reduced, y, train_size, seed)


# =============================================================================
# REAL DATASETS (CSV)
# =============================================================================

def map_classes_by_frequency(df, class_column="attack_cat"):
    """
    Map class categories to integer values based on descending frequency.
    
    Args:
        df (pd.DataFrame): DataFrame with class column
        class_column (str): Column name for class categories
    
    Returns:
        pd.DataFrame: DataFrame with additional numeric class column
    """
    if class_column not in df.columns:
        raise KeyError(f"Column '{class_column}' not found in DataFrame.")
    
    # Sort categories by frequency and create mapping
    sorted_categories = df[class_column].value_counts().index.tolist()
    class_dict = {category: idx for idx, category in enumerate(sorted_categories)}
    
    # Map to numeric values
    df[class_column + "_num"] = df[class_column].map(class_dict)
    return df

def extract_features_and_labels(df, label_column="attack_cat_num", columns_to_remove=None):
    """
    Extract features and labels from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        label_column (str): Name of label column
        columns_to_remove (list): Additional columns to remove from features
    
    Returns:
        tuple: (X, y) features and labels
    """
    if columns_to_remove is None:
        columns_to_remove = []
    
    columns_to_drop = [label_column] + columns_to_remove
    X = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' for missing columns
    y = df[label_column]
    return X, y


def balanced_sampling(df, target_col, n_samples, random_state=42):
    """
    Extract balanced subset from DataFrame.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        n_samples (int): Total number of samples to extract
        seed (int): Random seed
    
    Returns:
        pd.DataFrame: Balanced subset of the data
    """
    unique_classes = df[target_col].unique()
    n_classes = len(unique_classes)
    
    # Compute per-class sample sizes
    samples_per_class = n_samples // n_classes
    remaining = n_samples % n_classes
    
    class_counts = {
        cls: samples_per_class + (1 if i < remaining else 0)
        for i, cls in enumerate(unique_classes)
    }
    
    # Perform sampling
    df_sampled = (
        df.groupby(target_col, group_keys=False)
          .apply(lambda g: g.sample(
              n=class_counts[g.name],
              random_state=random_state
          ))
    )
    
    return df_sampled

def preprocess_real_dataset(file_path, class_column="attack_cat", seed=42, 
                          n_samples=None, class_threshold=None, sampling=None):
    """
    Load and preprocess real dataset from CSV file.
    
    Args:
        file_path (str): Path to CSV file
        class_column (str): Column containing class labels
        seed (int): Random seed
        n_samples (int, optional): Number of samples to extract
        class_threshold (int, optional): Keep only classes with rank < threshold
        sampling (str, optional): Sampling strategy ('balanced' or None)
    
    Returns:
        tuple: (X_scaled, y) preprocessed features and labels
    """
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Map classes to numeric values
    df = map_classes_by_frequency(df, class_column=class_column)
    numeric_label = class_column + "_num"
    
    # Apply class threshold filtering
    if class_threshold is not None:
        if class_threshold < 2:
            raise ValueError("class_threshold must be >= 2")
        df = df[df[numeric_label] < class_threshold]
        print(f"After class filtering: {df.shape}")
    
    # Apply sampling
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        
        if sampling == "balanced":
            df = balanced_sampling(df, class_column, n_samples, random_state=seed)
        else:
            df = df.sample(n=n_samples, random_state=seed)
        
        print(f"After sampling: {df.shape}")
    
    # Extract features and labels
    columns_to_remove = ["label", class_column, numeric_label]  # 'label' for UNSW_NB15
    X, y = extract_features_and_labels(df, numeric_label, columns_to_remove)
    
    # Encode categorical features
    X_encoded = encode_categorical_features(X)
    
    # Scale features
    X_scaled = apply_scaling(X_encoded, scaler_type="standard")
    
    print(f"Processed feature matrix shape: {X_scaled.shape}")
    return X_scaled, y

def get_real_dataset(file_path, class_column="attack_cat", seed=42, 
                    n_samples=None, class_threshold=None, sampling=None, 
                    train_size=0.8):
    """
    Load and prepare real dataset with train/test split.
    
    Args:
        file_path (str): Path to CSV file
        class_column (str): Column containing class labels
        seed (int): Random seed
        n_samples (int, optional): Number of samples to extract
        class_threshold (int, optional): Keep only classes with rank < threshold
        sampling (str, optional): Sampling strategy ('balanced' or None)
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    X, y = preprocess_real_dataset(
        file_path, class_column, seed, n_samples, class_threshold, sampling
    )
    
    return perform_train_test_split(X, y, train_size, seed)


def get_real_dataset_with_reduction(file_path, dataset_name, reduction_type, 
                                  n_components, class_column="attack_cat", 
                                  seed=42, sample_size=None, class_threshold=None, 
                                  sampling=None, train_size=0.8):
    """
    Load real dataset with dimensionality reduction.
    
    Args:
        file_path (str): Path to CSV file
        dataset_name (str): Name identifier for the dataset
        reduction_type (str): Type of reduction ('pca' or 'umap')
        n_components (int): Number of components after reduction
        class_column (str): Column containing class labels
        seed (int): Random seed
        sample_size (int, optional): Number of samples to extract
        class_threshold (int, optional): Keep only classes with rank < threshold
        sampling (str, optional): Sampling strategy ('balanced' or None)
        train_size (float): Proportion of training data
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    print(f"Processing dataset: {dataset_name}")
    print(f"File path: {file_path}")
    
    # Preprocess dataset
    X, y = preprocess_real_dataset(
        file_path, class_column, seed, sample_size, class_threshold, sampling
    )
    
    # Apply dimensionality reduction
    X_reduced = apply_dimensionality_reduction(X, reduction_type, n_components, seed)
    
    # Split data
    return perform_train_test_split(X_reduced, y, train_size, seed)

# =============================================================================
# UNIFIED INTERFACE FUNCTIONS
# =============================================================================

def create_dataset(dataset_type, dataset_params, file_path=None, seed=42):
    """
    Unified function to create datasets of different types.
    
    Args:
        dataset_type (str): Type of dataset ('toys', 'toys_red', 'synthetic', 
                           'synthetic_red', 'real')
        dataset_params (dict): Parameters specific to dataset type
        file_path (str, optional): Path to CSV file for real datasets
        seed (int): Random seed
    
    Returns:
        dict: Dictionary with train/test features and labels
    """
    if dataset_type == "toys":
        return get_toy_dataset(
            dataset_name=dataset_params['dataset_name'],
            seed=seed,
            train_size=dataset_params.get('train_size', 0.8)
        )
    
    elif dataset_type == "toys_red":
        return get_toy_dataset_with_reduction(
            dataset_name=dataset_params['dataset_name'],
            reduction_type=dataset_params['reduction'],
            n_components=dataset_params['features'],
            seed=seed,
            train_size=dataset_params.get('train_size', 0.8)
        )
    
    elif dataset_type == "synthetic":
        return get_synthetic_dataset(
            sample_size=dataset_params['sample_size'],
            n_features=dataset_params['features'],
            seed=seed,
            class_sep=dataset_params.get('class_sep', 1.0),
            n_redundant=dataset_params.get('redundant', 0),
            train_size=dataset_params.get('train_size', 0.8)
        )
    
    elif dataset_type == "synthetic_red":
        return get_synthetic_dataset_with_reduction(
            sample_size=dataset_params['sample_size'],
            n_features=dataset_params['features'],
            reduction_type=dataset_params['reduction'],
            n_components=dataset_params['features_red'],
            seed=seed,
            class_sep=dataset_params.get('class_sep', 1.0),
            n_redundant=dataset_params.get('redundant', 0),
            train_size=dataset_params.get('train_size', 0.8)
        )
    
    elif dataset_type == "real":
        return get_real_dataset_with_reduction(
            file_path=file_path,
            dataset_name=dataset_params['dataset_name'],
            reduction_type=dataset_params['reduction'],
            n_components=dataset_params['features'],
            class_column=dataset_params.get('class_column', 'attack_cat'),
            seed=seed,
            sample_size=dataset_params.get('sample_size'),
            class_threshold=dataset_params.get('class_threshold'),
            sampling=dataset_params.get('sampling'),
            train_size=dataset_params.get('train_size', 0.8)
        )
    
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

# Maintain backward compatibility with existing code
def prepare_dataset_with_reduction(dataset_name="Breast_cancer", reduction="pca", 
                                 features=2, seed=42):
    """Legacy function for backward compatibility."""
    return get_toy_dataset_with_reduction(dataset_name, reduction, features, seed)

def prepare_dataset_split_with_reduction(file_path, dataset_name="unsw_nb15", 
                                       class_column="attack_cat", reduction="pca", 
                                       features=2, seed=42, sample_size=None, 
                                       class_threshold=None, sampling=None):
    """Legacy function for backward compatibility."""
    return get_real_dataset_with_reduction(
        file_path, dataset_name, reduction, features, class_column, 
        seed, sample_size, class_threshold, sampling
    )