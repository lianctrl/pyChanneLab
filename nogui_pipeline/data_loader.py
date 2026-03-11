"""
Data loading utilities for experimental datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import DATA_FOLDER


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single dataset from CSV file
    
    Parameters
    ----------
    filename : str
        Name of the CSV file to load
        
    Returns
    -------
    x_data : np.ndarray
        X-axis data (voltage or time)
    y_data : np.ndarray
        Y-axis data (normalized current/conductance)
    y_err : np.ndarray
        Error bars
    """
    filepath = DATA_FOLDER + filename
    dataset = pd.read_csv(filepath, sep=',', skiprows=1)
    data = dataset.to_numpy()
    
    return data[:, 0], data[:, 1], data[:, 2]


def load_all_experimental_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all experimental datasets
    
    Returns
    -------
    data_dict : dict
        Dictionary containing all experimental data with keys:
        'activation', 'inactivation', 'cs_inactivation', 'recovery'
        Each value is a tuple of (x_data, y_data, y_err)
    """
    datasets = {
        'activation': 'Activation_WT.csv',
        'inactivation': 'Inactivation_WT.csv',
        'cs_inactivation': 'CS-Inactivation_WT.csv',
        'recovery': 'Recovery_WT.csv'
    }
    
    data_dict = {}
    for key, filename in datasets.items():
        data_dict[key] = load_dataset(filename)
    
    return data_dict


# Load data when module is imported
EXPERIMENTAL_DATA = load_all_experimental_data()
