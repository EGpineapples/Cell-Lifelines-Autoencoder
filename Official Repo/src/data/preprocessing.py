import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(folder_path, expected_timepoints=376, downsample_factor=10):
    """
    Load and preprocess lifeline data from a specified folder.
    
    Parameters:
    - folder_path: str, path to the folder containing lifeline data files.
    - expected_timepoints: int, the number of timepoints to keep after preprocessing.
    - downsample_factor: int, factor to downsample the timepoints for faster processing.
    
    Returns:
    - normalized_data: numpy array, the preprocessed and normalized data.
    """
    file_names = sorted(os.listdir(folder_path))
    data_list = []
    filtered_reactions = None

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        lifeline_df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Downsample and ensure a specific number of timepoints
        reduced_df = lifeline_df.iloc[::downsample_factor][:expected_timepoints]
        
        if filtered_reactions is None:
            non_constant_fluxes_df = lifeline_df.loc[:, (lifeline_df != lifeline_df.iloc[0]).any()]
            unique_reactions_fluxes_df = non_constant_fluxes_df.loc[:, ~non_constant_fluxes_df.columns.duplicated()]
            filtered_reactions = unique_reactions_fluxes_df.columns

        filtered_lifeline_df = reduced_df[filtered_reactions]
        data_list.append(filtered_lifeline_df.to_numpy())

    data_3d = np.stack(data_list, axis=0)  # Stack along new axis to create 3D array

    # Normalize the data
    n_samples, n_timesteps, n_features = data_3d.shape
    scalers = [[StandardScaler() for _ in range(n_features)] for _ in range(n_samples)]
    normalized_data = np.empty_like(data_3d, dtype=float)
    
    for sample_index in range(n_samples):
        for feature_index in range(n_features):
            feature_data = data_3d[sample_index, :, feature_index].reshape(-1, 1)
            normalized_feature_data = scalers[sample_index][feature_index].fit_transform(feature_data)
            normalized_data[sample_index, :, feature_index] = normalized_feature_data.flatten()

    print(f"Shape of the preprocessed data: {normalized_data.shape}")
    return normalized_data

