import os
import sys
from pathlib import Path
from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import json

import numpy as np
import pandas as pd
import openml

    
cc18_dataset_ids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 
                    31, 32, 37, 38, 44, 46, 50, 54, 151, 182, 188, 
                    300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 
                    1067, 1068, 1461, 1462, 1464, 1468, 1475, 1478, 
                    1480, 1485, 1486, 1487, 1489, 1494, 1497, 1501, 
                    1510, 1590, 4134, 4534, 4538, 6332, 23381, 23517, 
                    40499, 40668, 40670, 40701, 40923, 40927, 40966, 
                    40975, 40978, 40979, 40982, 40983, 40984, 40994, 
                    40996, 41027]

ctr23_dataset_ids = [41021, 44956, 44957, 44958, 44959, 44960, 44962, 
                     44963, 44964, 44965, 44966, 44967, 44969, 44970, 
                     44971, 44972, 44973, 44974, 44975, 44976, 44977, 
                     44978, 44979, 44980, 44981, 44983, 44984, 44987, 
                     44989, 44990, 44992, 44993, 44994, 45012, 45402]


##########################  |
### metadata functions ###  |
##########################  V


def save_df_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to a CSV file. If the path directory does not exist, it will be created."""
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True)
    print(f"DataFrame saved to {filepath}")



def load_df_from_csv(filepath: Path) -> pd.DataFrame:
    """Load a DataFrame from a CSV file."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"DataFrame loaded from {filepath}")
    return df



def count_missing_values(df: pd.DataFrame) -> dict:
    """Count missing values in a DataFrame."""
    n_missing = df.isnull().sum().sum()
    n_rows_with_missing = df.isnull().any(axis=1).sum()
    n_cols_with_missing = df.isnull().any(axis=0).sum()
    return n_missing, n_rows_with_missing, n_cols_with_missing



def count_onehot_features(df: pd.DataFrame, categorical_indicator: list) -> int:
    """Count one-hot encoded features in a DataFrame."""
    n_features = df.shape[1]
    n_features_onehot = n_features
    for name, is_cat in zip(df.columns, categorical_indicator):
        if is_cat:
            n_unique = df[name].nunique(dropna=False)
            if n_unique > 2:
                n_features_onehot += n_unique - 1  # -1 because original column is replaced
    return n_features_onehot



def _get_openml_metadata(data_openml_dir: Path,
                        metadata_filename: str,
                        collection_id: int,
                        classification_or_regression: str,
                        verbose: bool,
                        overwrite_csv: bool)-> pd.DataFrame:
    #load cache if exists
    openml.config.set_root_cache_directory(data_openml_dir)
    metadata_path = data_openml_dir / metadata_filename
    if metadata_path.exists() and not overwrite_csv:
        return load_df_from_csv(metadata_path)
    
    # Fetch the OpenML collection with ID
    collection = openml.study.get_suite(collection_id)
    dataset_ids = collection.data

    metadata_list = []
    for i, dataset_id in enumerate(dataset_ids):
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        n_missing, n_rows_with_missing, n_cols_with_missing = count_missing_values(X)
        n_features = X.shape[1]
        n_features_onehot = count_onehot_features(X, categorical_indicator)

        metadata = {
            'id': dataset.id,
            'name': dataset.name,
            'n_obs': int(dataset.qualities['NumberOfInstances']),
            'n_features': n_features,
            'n_features_onehot': n_features_onehot,
            'n_missing_values': n_missing,
            'n_rows_with_missing': n_rows_with_missing,
            'n_cols_with_missing': n_cols_with_missing,
        }
        if classification_or_regression == 'classification':
            metadata['n_classes'] = len(np.unique(y))
        elif classification_or_regression == 'regression':
            metadata['%_unique_y'] = len(np.unique(y))/len(y)
        else:
            raise ValueError("classification_or_regression must be 'classification' or 'regression'")
        
        metadata_list.append(metadata)
        if verbose:
            print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset.id}: {dataset.name}")

    df_metadata = pd.DataFrame(metadata_list).set_index("id").sort_index()
    save_df_to_csv(df_metadata, metadata_path)
    return df_metadata



def get_openml_cc18_metadata(data_openml_dir: Path,
                            metadata_filename: str = "openml_cc18_metadata.csv",
                            verbose: bool = False,
                            overwrite_csv: bool = False)-> pd.DataFrame:
    """
    Fetch metadata for the OpenML Curated Classification benchmarking suite 2018.
    Returns a DataFrame with dataset metadata.
    Saves the metadata to a CSV file in the results directory.
    """
    return _get_openml_metadata(
        data_openml_dir=data_openml_dir,
        metadata_filename=metadata_filename,
        collection_id=99, # https://www.openml.org/search?type=study&study_type=task&id=99
        classification_or_regression='classification',
        verbose=verbose,
        overwrite_csv=overwrite_csv
    )



def get_openml_ctr23_metadata(data_openml_dir: Path,
                            metadata_filename: str = "openml_ctr23_metadata.csv",
                            verbose: bool = False,
                            overwrite_csv: bool = False)-> pd.DataFrame:
    """
    Fetch metadata for the OpenML Curated Tabular Regression benchmarking suite 2018.
    Returns a DataFrame with dataset metadata.
    Saves the metadata to a CSV file in the results directory.
    """
    return _get_openml_metadata(
        data_openml_dir=data_openml_dir,
        metadata_filename=metadata_filename,
        collection_id=353, # https://www.openml.org/search?type=study&study_type=task&id=353
        classification_or_regression='regression',
        verbose=verbose,
        overwrite_csv=overwrite_csv
    )
    
    

#################################  |
### load openml numpy dataset ###  |
#################################  V

def np_load_openml_dataset(
        dataset_id: int, 
        regression_or_classification: str = "regression",
        one_hot_features: bool = True,
        normalize_features: bool = True,
        data_openml_dir:Optional[Path] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads and optionally preprocesses an openML dataset.
    For regression, preprocessing includes normalization of features and clipping.
    For classification it additionally supports one-hot encoding of labels.
    
    Returns X (shape N,D), and y (shape N,d) for regression or 
    one-hot (N) for classification.
    """
    
    # set default data directory if provided
    if data_openml_dir is not None:
        openml.config.cache_directory = data_openml_dir
    
    # Fetch dataset from OpenML by its ID
    dataset_id = int(dataset_id)
    dataset = openml.datasets.get_dataset(dataset_id)
    df, _, categorical_indicator, attribute_names = dataset.get_data()
    
    # for classification, convert target to integer codes
    if regression_or_classification == "classification":
        tar = dataset.default_target_attribute
        target_index = attribute_names.index(tar)
        categorical_indicator[target_index] = False
        df[tar] = df[tar].astype('category').cat.codes
    elif regression_or_classification == "regression":
        #normalize target
        tar = dataset.default_target_attribute
        target_index = attribute_names.index(tar)
        categorical_indicator[target_index] = False
        df[tar] = (df[tar] - df[tar].mean()) / (df[tar].std() + 1e-5)
        #df[tar] = np.clip(df[tar], -5, 5)
    else:
        raise ValueError("regression_or_classification must be 'regression' or 'classification'")
    
    # Fix missspecified features: Dataset 44962 has numeric month and day
    if dataset_id == 44962:
        categorical_indicator[2:4] = [True, True]
        df['month'] = df['month'].astype('category')
        df['day'] = df['day'].astype('category')
    
    # Remove columns with more than 50% missing values
    missing_frac = df.isnull().mean()
    cols_to_remove = missing_frac[missing_frac > 0.5].index.tolist()
    remove_indices = [attribute_names.index(col) for col in cols_to_remove if col in attribute_names]
    df = df.drop(columns=cols_to_remove, errors='ignore')
    for idx in sorted(remove_indices, reverse=True):
        del attribute_names[idx]
        del categorical_indicator[idx]
        
    # Replace missing values with the median (numeric) or mode (categorical), using categorical_indicator
    cat_cols = [col for col, is_cat in zip(df.columns, categorical_indicator) if is_cat]
    num_cols = [col for col, is_cat in zip(df.columns, categorical_indicator) if not is_cat and col!= dataset.default_target_attribute]
    if regression_or_classification == "classification" and dataset.default_target_attribute not in cat_cols:
        cat_cols.append(dataset.default_target_attribute)
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # One-hot encode categorical variables
    if one_hot_features:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype("category").cat.codes)
    
    # Normalize features
    if normalize_features:
        df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-5)
        df[num_cols] = np.clip(df[num_cols], -5, 5)
    
    # Separate target variable
    y = np.array(df.pop(dataset.default_target_attribute))
    X = np.array(df).astype(np.float32)
    return X, y, (categorical_indicator if not one_hot_features else None)