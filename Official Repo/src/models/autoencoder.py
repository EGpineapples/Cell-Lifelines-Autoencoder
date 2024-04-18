import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Dropout, RepeatVector, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow.keras.backend as K
import tensorflow as tf

class OneCycleLRScheduler(tf.keras.callbacks.Callback):
    """
    One Cycle Learning Rate Scheduler.
    """
    def __init__(self, max_lr, total_steps, div_factor=25, increase_fraction=0.3):
        super(OneCycleLRScheduler, self).__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.div_factor = div_factor
        self.increase_fraction = increase_fraction
        self.higher_lr = self.max_lr
        self.lower_lr = self.max_lr / self.div_factor
        self.step_size_up = int(self.total_steps * self.increase_fraction)
        self.step_size_down = self.total_steps - self.step_size_up

    def on_train_batch_begin(self, batch, logs=None):
        if batch < self.step_size_up:
            lr = self.lower_lr + (self.higher_lr - self.lower_lr) * (batch / self.step_size_up)
        else:
            lr = self.higher_lr - (self.higher_lr - self.lower_lr) * ((batch - self.step_size_up) / self.step_size_down)
        K.set_value(self.model.optimizer.lr, lr)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_autoencoder_model(input_shape, lstm_units, lstm_units_layer2, dense_units, dropout_rate, learning_rate, model_type='autoencoder'):
    """
    Create autoencoder model based on the provided hyperparameters and model type.
    
    Parameters:
    - input_shape: Tuple, shape of the input data.
    - lstm_units: int, number of units in the first LSTM layer.
    - lstm_units_layer2: int, number of units in the second LSTM layer.
    - dense_units: int, number of units in the dense layer for latent space representation.
    - dropout_rate: float, dropout rate for regularization.
    - learning_rate: float, learning rate for the optimizer.
    - model_type: str, type of model to create ('autoencoder' or 'autoencoder_timesteps').

    Returns:
    - model: keras Model, the autoencoder model.
    """
    inputs = Input(shape=input_shape)

    # First LSTM layer with dropout
    encoded = LSTM(lstm_units, activation='tanh', return_sequences=True)(inputs)
    encoded = Dropout(dropout_rate)(encoded)

    if model_type == 'autoencoder':
        encoded = LSTM(lstm_units_layer2, activation='tanh', return_sequences=False)(encoded)
        latent_space = Dense(dense_units, activation='tanh')(encoded)
    elif model_type == 'autoencoder_timesteps':
        encoded = LSTM(lstm_units_layer2, activation='tanh', return_sequences=True)(encoded)
        latent_space = Dense(dense_units, activation='tanh')(encoded)

    if model_type == 'autoencoder':
        decoded = RepeatVector(input_shape[0])(latent_space)
    else:
        decoded = latent_space
    decoded = LSTM(lstm_units_layer2, activation='tanh', return_sequences=True)(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    decoded = LSTM(lstm_units, activation='tanh', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

def objective(trial, train_data, val_data, config, model_type='autoencoder'):
    lstm_units = trial.suggest_categorical('lstm_units', config['model']['lstm_units'])
    lstm_units_layer2 = trial.suggest_categorical('lstm_units_layer2', config['model']['lstm_units_layer2'])
    dense_units = trial.suggest_categorical('dense_units', config['model']['dense_units'])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', config['model']['learning_rate']['min'], config['model']['learning_rate']['max'], log=True)
    batch_size = trial.suggest_categorical('batch_size', config['model']['batch_size'])

    model = create_autoencoder_model(train_data.shape[1:], lstm_units, lstm_units_layer2, dense_units, dropout_rate, learning_rate, model_type)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=config['model']['patience'], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config['model']['reduce_lr_patience'], min_lr=config['model']['min_lr'])

    model.fit(train_data, train_data, epochs=config['model']['epochs'], batch_size=batch_size, validation_data=(val_data, val_data), callbacks=[early_stop, reduce_lr], verbose=0)
    val_loss = model.evaluate(val_data, val_data, verbose=0)
    return val_loss

def train_and_optimize_autoencoder(normalized_data, config, model_type='autoencoder', use_one_cycle_lr=False):
    train_data, val_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, val_data, config, model_type), n_trials=config['model']['n_trials'])

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    best_dropout_rate = best_params.get('dropout_rate', study.best_trial.params['dropout_rate'])

    best_model = create_autoencoder_model(
        train_data.shape[1:], 
        lstm_units=best_params['lstm_units'], 
        lstm_units_layer2=best_params['lstm_units_layer2'], 
        dense_units=best_params['dense_units'], 
        dropout_rate=best_dropout_rate, 
        learning_rate=best_params['learning_rate'], 
        model_type=model_type
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=config['model']['patience'], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config['model']['reduce_lr_patience'], min_lr=config['model']['min_lr'])
    callbacks = [early_stop, reduce_lr]
    
    if use_one_cycle_lr:
        batch_size = best_params['batch_size']
        steps_per_epoch = np.ceil(len(train_data) / batch_size)
        total_steps = steps_per_epoch * config['model']['epochs']
        max_lr = best_params.get('learning_rate', 0.001)
        one_cycle_lr = OneCycleLRScheduler(max_lr=max_lr, total_steps=total_steps)
        callbacks.append(one_cycle_lr)
    
    best_model.fit(train_data, train_data, epochs=config['model']['epochs'], batch_size=best_params['batch_size'], validation_data=(val_data, val_data), callbacks=callbacks, verbose=1)
    model_save_path = f'src/models/{model_type}_best_model.h5'
    best_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return best_model


def save_latent_representations(model, train_data, file_path, layer_name=None):
    """
    Save latent space representations to an Excel file.
    If the output is 3D, flatten it into 2D.
    """
    if layer_name:
        latent_layer_output = model.get_layer(layer_name).output
    else:
        latent_layer_output = model.layers[-2].output # CHECK 

    latent_model = Model(inputs=model.input, outputs=latent_layer_output)
    latent_representations = latent_model.predict(train_data)

    if len(latent_representations.shape) == 3:
        num_samples, num_timesteps, num_features = latent_representations.shape
        latent_2d = latent_representations.reshape(num_samples * num_timesteps, num_features)
    else:
        latent_2d = latent_representations

    latent_df = pd.DataFrame(latent_2d)
    latent_df.to_csv(file_path, index=False)
    print(f"Latent representations saved to {file_path}.")


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

    plt.savefig(save_path)
    plt.show()