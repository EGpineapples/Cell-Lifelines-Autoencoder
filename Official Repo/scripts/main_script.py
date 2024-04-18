import os
import sys
import yaml

src_directory = os.path.abspath(r"D:\Desktop\Work\Official Repo\src")
if src_directory not in sys.path:
    sys.path.append(src_directory)

from data.preprocessing import load_and_preprocess_data
from models.autoencoder import train_and_optimize_autoencoder, save_latent_representations, plot_lifeline_reconstruction

def load_config(config_path):
    """
    Load YAML configuration file from the specified path.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = load_config("D:/Desktop/Work/Official Repo/configs/config.yaml")

    folder_path = r"D:\Desktop\Work\Official Repo\src\data\100_dFBA_Lifelines"
    downsample_factor = config["data"]["downsample_factor"]
    expected_timepoints = config["data"]["expected_timepoints"]

    # Load and preprocess data
    print("Loading and preprocessing data...")
    normalized_data = load_and_preprocess_data(folder_path, expected_timepoints, downsample_factor)
    
    # Train and optimize autoencoder for the standard model
    print("Training and optimizing standard autoencoder...")
    best_model_standard = train_and_optimize_autoencoder(normalized_data, config, model_type='autoencoder')

    latent_file_path_standard = r"D:\Desktop\Work\Official Repo\src\models\latent_representations_standard.xlsx"
    save_latent_representations(best_model_standard, normalized_data, latent_file_path_standard)

    lifeline_index = config["plotting"]["lifeline_index"]
    reaction_index = config["plotting"]["reaction_index"]
    plot_save_path_standard = "D:/Desktop/Work/Official Repo/plots/lifeline_reconstruction_standard.png"  # Adjust this path as needed
    plot_lifeline_reconstruction(best_model_standard, normalized_data, lifeline_index, reaction_index, plot_save_path_standard)

    # Train and optimize autoencoder for the timestep model
    print("Training and optimizing timestep autoencoder...")
    best_model_timesteps = train_and_optimize_autoencoder(normalized_data, config, model_type='autoencoder_timesteps')

    # Save latent representations and plot reconstruction for the timestep model
    latent_file_path_timesteps = r"D:\Desktop\Work\Official Repo\src\models\latent_representations_timesteps.csv"
    save_latent_representations(best_model_timesteps, normalized_data, latent_file_path_timesteps)

    plot_save_path_timesteps = "D:/Desktop/Work/Official Repo/plots/lifeline_reconstruction_timesteps.png"  # Adjust this path as needed
    plot_lifeline_reconstruction(best_model_timesteps, normalized_data, lifeline_index, reaction_index, plot_save_path_timesteps)