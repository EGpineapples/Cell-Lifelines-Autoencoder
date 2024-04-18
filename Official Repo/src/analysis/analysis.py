import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

def load_autoencoder_model(model_path):
    return load_model(model_path)

def build_decoder(autoencoder, model_type):
    if model_type == 'autoencoder':
        latent_layer_name = 'dense_2'  
    else:  
        latent_layer_name = 'dense_6' 
    
    latent_layer = autoencoder.get_layer(latent_layer_name)
    
    latent_layer_index = autoencoder.layers.index(latent_layer)
    
    decoder_input = Input(shape=latent_layer.output_shape[1:])
    x = decoder_input
    for layer in autoencoder.layers[latent_layer_index + 1:]:
        x = layer(x)
    
    decoder = Model(inputs=decoder_input, outputs=x)
    return decoder

def analyze_lifeline_effect(autoencoder, decoder, normalized_data, lifeline_index, feature_index, model_type):
    if model_type == 'autoencoder':
        latent_layer_name = 'dense_2'  
    else:
        latent_layer_name = 'dense_6'  

    latent_space_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(latent_layer_name).output)
    latent_space = latent_space_model.predict(normalized_data)

    original_latent_sample = latent_space[lifeline_index]
    original_reconstruction = decoder.predict(np.expand_dims(original_latent_sample, axis=0))

    modified_latent_sample = np.copy(original_latent_sample)
    modified_latent_sample[:, 0] = 0  
    modified_reconstruction = decoder.predict(np.expand_dims(modified_latent_sample, axis=0))

    plot_reconstructions(normalized_data[lifeline_index, :, feature_index], original_reconstruction[0, :, feature_index], modified_reconstruction[0, :, feature_index], lifeline_index, feature_index)

def plot_reconstructions(original_data, original_reconstruction, modified_reconstruction, lifeline_index, feature_index):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(original_reconstruction, label='Original Reconstruction')
    plt.plot(modified_reconstruction, label='Modified Reconstruction')
    plt.title(f'Lifeline {lifeline_index+1}, Feature {feature_index+1}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_latent_perturbation_analysis(autoencoder, normalized_data, save_dir, model_type):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    decoder = build_decoder(autoencoder, model_type)
    num_lifelines, num_timesteps, num_features = normalized_data.shape
    
    for lifeline_index in range(num_lifelines):
        for feature_index in range(num_features):
            analyze_lifeline_effect(autoencoder, decoder, normalized_data, lifeline_index, feature_index, model_type)
