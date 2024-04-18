# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
#import cobra
#from cobra.io import load_model
import pandas as pd
from tqdm import tqdm

# Parameters (Feeding point)
x_feed, y_feed, z_feed = 1.15, 10.24, 0.15     #m

# Monod Kinetics (Modeling of Overflow Metabolism in Batch and Fed-Batch Cultures of Escherichia coli)
# (Lin et.al. 2000)
q_S_max = 1.5     # gS/gX/h
K_S = 0.03         # g/L

# Define seaborn theme
sns.set_theme()
sns.set_context('paper')
# Set the Seaborn color palette
sns.set_palette("colorblind")

# Define scientific formatter in matplotlib
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
formatter.format = '%.1e'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

import os
import pandas as pd
import numpy as np

folder_path = 'dFBA_Data_Frames_Lifelines - Small'
file_names = sorted(os.listdir(folder_path))

expected_timepoints = 376 # reduced timepoints by 10 for faster runs, just temporary 

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    lifeline_df = pd.read_csv(file_path, sep="\t", index_col=0)

    # Select every 10th row and ensure there are exactly 376 timepoints
    reduced_df = lifeline_df.iloc[::10][:expected_timepoints]
    reduced_df.to_csv(file_path, sep="\t")

data_list = []
filtered_reactions = None

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    lifeline_df = pd.read_csv(file_path, sep="\t", index_col=0)

    if filtered_reactions is None:
        non_constant_fluxes_df = lifeline_df.loc[:, (lifeline_df != lifeline_df.iloc[0]).any()]
        unique_reactions_fluxes_df = non_constant_fluxes_df.loc[:, ~non_constant_fluxes_df.columns.duplicated()]
        filtered_reactions = unique_reactions_fluxes_df.columns

    filtered_lifeline_df = lifeline_df[filtered_reactions]

    if filtered_lifeline_df.shape != (expected_timepoints, len(filtered_reactions)):
        print(f"Shape mismatch for file: {file_name}. Expected ({expected_timepoints}, {len(filtered_reactions)}) but got {filtered_lifeline_df.shape}")

    data_list.append(filtered_lifeline_df.to_numpy())

data_3d = np.stack(data_list, axis=1)
print("Shape of the Big Matrix:", data_3d.shape)

from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming data_3d is original data with shape (n_timesteps, n_samples, n_features)
# Example: n_timesteps = 376, n_samples = 14, n_features = 48

n_timesteps, n_samples, n_features = data_3d.shape

# Initialize a StandardScaler for each feature for each sample
scalers = [[StandardScaler() for _ in range(n_features)] for _ in range(n_samples)]

# Normalized data will have the same shape as data_3d initially
normalized_data = np.empty_like(data_3d, dtype=float)

# Apply the scalers to each feature for each sample across all timesteps
for sample_index in range(n_samples):
    for feature_index in range(n_features):
        # Extract the feature data for the current sample
        feature_data = data_3d[:, sample_index, feature_index].reshape(-1, 1)
        
        # Fit and transform the data using the corresponding scaler
        normalized_feature_data = scalers[sample_index][feature_index].fit_transform(feature_data)
        
        # Assign the normalized data back to its position in the normalized_data array
        normalized_data[:, sample_index, feature_index] = normalized_feature_data.flatten()

# Transpose normalized_data to have the shape (n_samples, n_timesteps, n_features)
normalized_data = np.transpose(normalized_data, (1, 0, 2))

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Assuming five_lifeline is a 3D array with the shape (1, 376, 48)
train_data = normalized_data
val_data = train_data  # Using the same data for validation as a placeholder

# Define the input layer with the shape (timesteps, features)
input_shape = (train_data.shape[1], train_data.shape[2])
inputs = Input(shape=input_shape)

# Encoder layers with glorot_uniform initializer
encoded = LSTM(300, activation='tanh', return_sequences=True,
               kernel_initializer=glorot_uniform(seed=0))(inputs)
encoded = LSTM(200, activation='tanh', return_sequences=True,
               kernel_initializer=glorot_uniform(seed=0))(encoded)
encoded = LSTM(150, activation='tanh', return_sequences=False,
               kernel_initializer=glorot_uniform(seed=0))(encoded)
# Latent space with he_normal initializer
latent_space = Dense(100, activation='tanh',
                     kernel_initializer=he_normal(seed=0))(encoded)

# Decoder layers, repeat the latent space at each timestep
decoded = RepeatVector(input_shape[0])(latent_space)
decoded = LSTM(150, activation='tanh', return_sequences=True,
               kernel_initializer=glorot_uniform(seed=0))(decoded)
decoded = LSTM(200, activation='tanh', return_sequences=True,
               kernel_initializer=glorot_uniform(seed=0))(decoded)
decoded = LSTM(300, activation='tanh', return_sequences=True,
               kernel_initializer=glorot_uniform(seed=0))(decoded)

# Output layer, reconstruct the original input
outputs = TimeDistributed(Dense(input_shape[1]))(decoded)

# Create the autoencoder model
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# Summary of the model
autoencoder.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Fit the model
history = autoencoder.fit(
    train_data,
    train_data,
    epochs = 800,
    batch_size=32,  # Adjust the batch size if necessary
    validation_data=(val_data, val_data),
    callbacks=[reduce_lr, early_stop]
)

# Predict on the validation set
val_predictions = autoencoder.predict(val_data)

# Compute mean squared error on the validation set
mean_squared_error = np.mean(np.square(val_data - val_predictions))

# Compute root mean squared error (RMSE)
reconstruction_loss = np.sqrt(mean_squared_error)
print(f"Validation RMSE: {reconstruction_loss}")

import matplotlib.pyplot as plt

# Assuming that one_lifeline is your data reshaped correctly for the model input
# Example: one_lifeline = np.random.rand(1, 376, 1)  # Use your actual reshaped data here

# Predict the reconstruction for the lifeline
reconstructed = autoencoder.predict(train_data)
lifeline = 4
feature = 6 

# Flatten the arrays to be able to plot them since they contain a single feature
scaled_series = train_data[lifeline,:,feature]
reconstructed_series = reconstructed[lifeline,:,feature]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(scaled_series, label='Original Data', color='blue')
plt.plot(reconstructed_series, label='Reconstructed Data', color='red', linestyle='--')
plt.title('Lifeline 1 for Single Feature')
plt.xlabel('Timepoint')
plt.ylabel('Value')
plt.legend()
plt.show()

from tensorflow.keras.models import Model

# Assuming 'encoded' is the output of your last encoder layer and 'inputs' is your input layer
latent_model = Model(inputs, latent_space)
latent_representations = latent_model.predict(train_data)

import pandas as pd

# Convert the latent representations to a DataFrame
latent_df = pd.DataFrame(latent_representations)

# Specify your desired Excel file path
excel_file_path = '/latent_representations_14_lifelines.xlsx'

# Save the DataFrame to an Excel file
latent_df.to_excel(excel_file_path, index=False)

print(f"Latent representations saved to {excel_file_path}")
