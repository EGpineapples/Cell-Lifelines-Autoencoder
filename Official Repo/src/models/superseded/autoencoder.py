import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_autoencoder(input_shape, lstm_units, lstm_units_layer2, dense_units, learning_rate):
    """
    Create autoencoder model based on the provided hyperparameters.
    """
    inputs = Input(shape=input_shape)
    encoded = LSTM(lstm_units, activation='tanh', return_sequences=True)(inputs)
    encoded = LSTM(lstm_units_layer2, activation='tanh', return_sequences=False)(encoded)
    latent_space = Dense(dense_units, activation='tanh')(encoded)
    decoded = RepeatVector(input_shape[0])(latent_space)
    decoded = LSTM(lstm_units_layer2, activation='tanh', return_sequences=True)(decoded)
    decoded = LSTM(lstm_units, activation='tanh', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    model = Model(inputs, outputs)
    return model

def objective(trial, train_data, val_data, config):
    """
    Objective function for Optuna study.
    """
    lstm_units = trial.suggest_categorical('lstm_units', config['model']['lstm_units'])
    lstm_units_layer2 = trial.suggest_categorical('lstm_units_layer2', config['model']['lstm_units_layer2'])
    dense_units = trial.suggest_categorical('dense_units', config['model']['dense_units'])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', config['model']['learning_rate']['min'], config['model']['learning_rate']['max'], log=True)
    batch_size = trial.suggest_categorical('batch_size', config['model']['batch_size'])

    model = create_autoencoder(train_data.shape[1:], lstm_units, lstm_units_layer2, dense_units, learning_rate)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    model.fit(train_data, train_data, epochs=100, batch_size=32, validation_data=(val_data, val_data), callbacks=[early_stop, reduce_lr], verbose=0)
    val_loss = model.evaluate(val_data, val_data, verbose=0)
    return val_loss

def train_and_optimize_autoencoder(normalized_data, config):
    """
    Train and optimize the autoencoder model using the provided normalized data and configuration.
    """
    train_data, val_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, val_data, config), n_trials=config['model']['n_trials'])

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Use the updated dictionary to create and fit the best model
    best_model = create_autoencoder(train_data.shape[1:], lstm_units=best_params['lstm_units'], lstm_units_layer2=best_params['lstm_units_layer2'], dense_units=best_params['dense_units'], learning_rate=best_params['learning_rate'])
    best_model.fit(train_data, train_data, epochs=config['model']['epochs'], batch_size=best_params['batch_size'], validation_data=(val_data, val_data), verbose=1)
    # Assuming 'best_model' is your trained model
    model_save_path = 'src/models/autoencoder.h5'  # The path where you want to save your model
    best_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return best_model


def save_latent_representations(model, train_data, file_path):
    """
    Save latent space representations to an Excel file.
    """
    latent_model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    latent_representations = latent_model.predict(train_data)
    latent_df = pd.DataFrame(latent_representations)
    latent_df.to_excel(file_path, index=False)
    print(f"Latent representations saved to {file_path}")

def plot_lifeline_reconstruction(autoencoder, normalized_data, lifeline_index, reaction_index, save_path):
    """
    Plot original vs reconstructed data for a specific lifeline and reaction.
    """
    reconstructed = autoencoder.predict(normalized_data)
    scaled_series = normalized_data[lifeline_index, :, reaction_index]
    reconstructed_series = reconstructed[lifeline_index, :, reaction_index]

    plt.figure(figsize=(10, 6))
    plt.plot(scaled_series, label='Original Data', color='blue')
    plt.plot(reconstructed_series, label='Reconstructed Data', color='red', linestyle='--')
    plt.title(f'Lifeline {lifeline_index + 1} for Reaction {reaction_index + 1}')
    plt.xlabel('Timepoint')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.show()