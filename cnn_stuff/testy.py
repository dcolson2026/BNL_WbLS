import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def generate_lognormal_pulse(length=1000, peak_time=None, sigma=0.5, scale=1.0, noise_level=0.001):
    """
    Generate a synthetic lognormal pulse with specified peak time
    
    Args:
        length: Length of the signal
        peak_time: Time index of the peak (if None, random)
        sigma: Shape parameter of lognormal distribution
        scale: Scale parameter of lognormal distribution
        noise_level: Amount of Gaussian noise to add
    
    Returns:
        signal: 1D array containing the pulse
        peak_time: Actual peak time index
    """
    if peak_time is None:
        # peak_time = np.random.randint(length // 4, 3 * length // 4)
        peak_time = np.random.randint(0, length)
    
    # Create time array
    t = np.arange(length)
    
    # Shift time array so peak occurs at peak_time
    t_shifted = t - peak_time
    
    # Only compute lognormal for positive values, zero elsewhere
    signal = np.zeros(length)
    positive_mask = t_shifted > 0
    
    if np.any(positive_mask):
        # Generate lognormal distribution
        signal[positive_mask] = (1 / (t_shifted[positive_mask] * sigma * np.sqrt(2 * np.pi))) * \
                               np.exp(-0.5 * ((np.log(t_shifted[positive_mask]) - np.log(scale)) / sigma) ** 2)
    
    # Normalize to max amplitude of 1
    if np.max(signal) > 0:
        signal = signal / np.max(signal)
    
    # Add noise
    signal += np.random.normal(0, noise_level, length)
    
    return signal, peak_time

def generate_dataset(n_samples=10000, signal_length=1000):
    """
    Generate a dataset of synthetic pulses with their peak times
    
    Args:
        n_samples: Number of samples to generate
        signal_length: Length of each signal
    
    Returns:
        X: Array of shape (n_samples, signal_length, 1) containing signals
        y: Array of shape (n_samples,) containing normalized peak times
    """
    X = np.zeros((n_samples, signal_length, 1))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Vary parameters for diversity
        sigma = np.random.uniform(0.3, 0.8)
        scale = np.random.uniform(0.5, 2.0)
        noise_level = np.random.uniform(0.005, 0.02)
        
        signal, peak_time = generate_lognormal_pulse(
            length=signal_length,
            sigma=sigma,
            scale=scale,
            noise_level=noise_level
        )
        
        X[i, :, 0] = signal
        # Normalize peak time to [0, 1] range
        y[i] = peak_time / signal_length
    
    return X, y

def create_1d_cnn_model(input_length):
    """
    Create a 1D CNN model for pulse peak detection
    
    Args:
        input_length: Length of input signals
    
    Returns:
        model: Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv1D(filters=32, kernel_size=11, activation='relu', 
                      input_shape=(input_length, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Second convolutional block
        layers.Conv1D(filters=64, kernel_size=9, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Third convolutional block
        layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Fourth convolutional block
        layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Global average pooling to reduce overfitting
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer (single neuron for regression)
        layers.Dense(1, activation='sigmoid')  # sigmoid to output [0,1] range
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae']
    )
    
    return model

def train_model():
    """
    Main function to generate data, create model, and train
    """
    print("Generating synthetic dataset...")
    signal_length = 1000
    n_samples = 20000
    
    # Generate dataset
    X, y = generate_dataset(n_samples, signal_length)
    
    # Split into train/validation/test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    print("Creating model...")
    model = create_1d_cnn_model(signal_length)
    model.summary()
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert back to actual time indices
    y_test_actual = y_test * signal_length
    y_pred_actual = y_pred.flatten() * signal_length
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    
    print(f"\nTest Results:")
    print(f"Mean Absolute Error: {mae:.2f} time steps")
    print(f"Root Mean Square Error: {rmse:.2f} time steps")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE (normalized): {test_mae:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/media/disk_o/my_histograms/testy.pdf")
    
    # Visualize some predictions
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        
        # Plot the signal
        plt.plot(X_test[i, :, 0], 'b-', alpha=0.7, label='Signal')
        
        # Plot actual peak
        actual_peak = int(y_test_actual[i])
        #plt.axvline(x=actual_peak, color='g', linestyle='--', linewidth=2, label=f'Actual: {actual_peak}')
        
        # Plot predicted peak
        pred_peak = int(y_pred_actual[i])
        plt.axvline(x=pred_peak, color='r', linestyle='--', linewidth=2, label=f'Predicted: {pred_peak}')
        
        plt.title(f'Sample {i+1} (Error: {abs(actual_peak - pred_peak):.1f})')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/media/disk_o/my_histograms/testy2.pdf")
    
    return model, history, (X_test, y_test, y_pred)

# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    #np.random.seed(42)
    #tf.random.set_seed(42)
    
    # Train the model
    model, history, test_data = train_model()
    
    # Save the model
    model.save('pulse_peak_detector.h5')
    print("\nModel saved as 'pulse_peak_detector.h5'")
    
    # Example of using the trained model for prediction
    print("\nExample prediction on new data:")
    signal_length = 1000
    test_signal, true_peak = generate_lognormal_pulse(length=signal_length)
    test_signal_reshaped = test_signal.reshape(1, signal_length, 1)
    
    predicted_peak_norm = model.predict(test_signal_reshaped, verbose=0)[0, 0]
    predicted_peak = int(predicted_peak_norm * signal_length)
    
    print(f"True peak time: {true_peak}")
    print(f"Predicted peak time: {predicted_peak}")
    print(f"Error: {abs(true_peak - predicted_peak)} time steps")
