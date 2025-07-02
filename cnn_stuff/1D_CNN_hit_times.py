import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Shape: (num_samples, sequence_length, 1)
X = np.array(waveforms)  # shape: (N, 1928)
X = X[..., np.newaxis]   # Add channel dimension: (N, 1928, 1)

# For classification: binary label (pulse or no pulse)
y = np.array(labels)     # shape: (N,) or (N,1)

# For regression: peak time index (float or int)
# y = np.array(peak_times)  # shape: (N,)

model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(1928, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])
