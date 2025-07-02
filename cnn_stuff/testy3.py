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
        # Allow peak anywhere in the signal with sufficient margin
        peak_time = np.random.randint(50, length - 50)
    
    # Create time array
    t = np.arange(length, dtype=np.float64)
    
    # Shift time array so peak occurs at peak_time
    t_shifted = t - peak_time
    
    # Only compute lognormal for positive values, zero elsewhere
    signal = np.zeros(length, dtype=np.float64)
    positive_mask = t_shifted > 0
    
    if np.any(positive_mask):
        # Generate lognormal distribution - more precise calculation
        t_pos = t_shifted[positive_mask]
        log_t = np.log(t_pos)
        log_scale = np.log(scale)
        
        # Compute lognormal PDF
        signal[positive_mask] = (1 / (t_pos * sigma * np.sqrt(2 * np.pi))) * \
                               np.exp(-0.5 * ((log_t - log_scale) / sigma) ** 2)
    
    # Normalize to max amplitude of 1
    if np.max(signal) > 0:
        signal = signal / np.max(signal)
    
    # Add noise
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, length)
    
    return signal, peak_time


def find_actual_peak(signal):
    """
    Find the actual peak in the signal (for validation)
    """
    return np.argmax(signal)


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
    X = np.zeros((n_samples, signal_length, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        # Vary parameters for diversity - keeping reasonable ranges
        sigma = np.random.uniform(0.3, 0.8)  
        scale = np.random.uniform(1.0, 4.0)  
        noise_level = np.random.uniform(0.002, 0.01)  # High SNR
        
        signal, peak_time = generate_lognormal_pulse(
            length=signal_length,
            sigma=sigma,
            scale=scale,
            noise_level=noise_level
        )
        
        # Verify the peak is where we expect it
        actual_peak = find_actual_peak(signal)
        
        # Use the actual measured peak for better accuracy
        X[i, :, 0] = signal
        y[i] = actual_peak / (signal_length - 1)  # Normalize to [0, 1]
    
    return X, y


def create_peak_detection_model(input_length):
    """
    Create an improved model specifically for peak detection
    Uses multi-scale analysis and attention-like mechanisms
    """
    input_layer = layers.Input(shape=(input_length, 1))
    
    # Multi-scale convolutional branches
    # Branch 1: Fine-scale features
    conv1_1 = layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    conv1_2 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv1_1)
    pool1 = layers.MaxPooling1D(2)(conv1_2)
    
    # Branch 2: Medium-scale features  
    conv2_1 = layers.Conv1D(64, 7, activation='relu', padding='same')(input_layer)
    conv2_2 = layers.Conv1D(64, 7, activation='relu', padding='same')(conv2_1)
    pool2 = layers.MaxPooling1D(2)(conv2_2)
    
    # Branch 3: Large-scale features
    conv3_1 = layers.Conv1D(64, 15, activation='relu', padding='same')(input_layer)
    conv3_2 = layers.Conv1D(64, 15, activation='relu', padding='same')(conv3_1)
    pool3 = layers.MaxPooling1D(2)(conv3_2)
    
    # Concatenate multi-scale features
    merged = layers.Concatenate()([pool1, pool2, pool3])
    
    # Additional processing layers
    conv4 = layers.Conv1D(128, 5, activation='relu', padding='same')(merged)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.1)(conv4)
    
    conv5 = layers.Conv1D(256, 5, activation='relu', padding='same')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.1)(conv5)
    
    # Attention-like mechanism for peak localization
    # This helps the model focus on the most relevant parts
    attention = layers.Conv1D(1, 1, activation='sigmoid', padding='same')(conv5)
    attended = layers.Multiply()([conv5, attention])
    
    # Global context
    gap = layers.GlobalAveragePooling1D()(attended)
    gmp = layers.GlobalMaxPooling1D()(attended)
    global_context = layers.Concatenate()([gap, gmp])
    
    # Dense layers for final prediction
    dense1 = layers.Dense(256, activation='relu')(global_context)
    dense1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(128, activation='relu')(dense1)
    dense2 = layers.Dropout(0.2)(dense2)
    
    dense3 = layers.Dense(64, activation='relu')(dense2)
    dense3 = layers.Dropout(0.1)(dense3)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    # Compile with custom loss that emphasizes accuracy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # MSE for precise regression
        metrics=['mae', 'mse']
    )
    
    return model


def create_alternative_model(input_length):
    """
    Alternative model using dilated convolutions for better temporal modeling
    """
    model = keras.Sequential([
        # Input layer
        layers.Conv1D(64, 7, activation='relu', padding='same', 
                      input_shape=(input_length, 1)),
        layers.BatchNormalization(),
        
        # Dilated convolution layers to capture different temporal scales
        layers.Conv1D(64, 5, activation='relu', padding='same', dilation_rate=1),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Conv1D(128, 5, activation='relu', padding='same', dilation_rate=2),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Conv1D(128, 5, activation='relu', padding='same', dilation_rate=4),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Conv1D(256, 5, activation='relu', padding='same', dilation_rate=8),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Conv1D(256, 3, activation='relu', padding='same', dilation_rate=16),
        layers.BatchNormalization(),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(use_alternative=False):
    """
    Main function to generate data, create model, and train
    """
    print("Generating synthetic dataset...")
    signal_length = 1000
    n_samples = 40000  # Larger dataset for better training
    
    # Generate dataset
    X, y = generate_dataset(n_samples, signal_length)
    
    # Analyze data distribution
    y_actual = y * (signal_length - 1)
    print(f"Peak time distribution:")
    print(f"  Min: {y_actual.min():.1f}, Max: {y_actual.max():.1f}")
    print(f"  Mean: {y_actual.mean():.1f}, Std: {y_actual.std():.1f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nDataset splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Create model
    print(f"\nCreating {'alternative' if use_alternative else 'multi-scale'} model...")
    if use_alternative:
        model = create_alternative_model(signal_length)
    else:
        model = create_peak_detection_model(signal_length)
    
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=1e-7
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_peak_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Load best weights and evaluate
    model.load_weights('best_peak_model.h5')
    
    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Convert to actual time indices
    y_test_actual = y_test * (signal_length - 1)
    y_pred_actual = y_pred.flatten() * (signal_length - 1)
    
    # Metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    
    print(f"\nTest Results:")
    print(f"  Mean Absolute Error: {mae:.2f} time steps")
    print(f"  Root Mean Square Error: {rmse:.2f} time steps")
    print(f"  Median Absolute Error: {np.median(np.abs(y_test_actual - y_pred_actual)):.2f} time steps")
    
    # Analyze error distribution
    errors = np.abs(y_test_actual - y_pred_actual)
    print(f"\nError Analysis:")
    print(f"  Errors ≤ 1 step: {np.mean(errors <= 1.0)*100:.1f}%")
    print(f"  Errors ≤ 2 steps: {np.mean(errors <= 2.0)*100:.1f}%")
    print(f"  Errors ≤ 5 steps: {np.mean(errors <= 5.0)*100:.1f}%")
    print(f"  Max error: {errors.max():.1f} steps")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Training history
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Val')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Prediction scatter
    plt.subplot(2, 3, 3)
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, s=10)
    plt.plot([0, signal_length-1], [0, signal_length-1], 'r--', lw=2)
    plt.xlabel('Actual Peak Time')
    plt.ylabel('Predicted Peak Time')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Error histogram
    plt.subplot(2, 3, 4)
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (time steps)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Error vs position
    plt.subplot(2, 3, 5)
    plt.scatter(y_test_actual, errors, alpha=0.5, s=10)
    plt.xlabel('Actual Peak Position')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Peak Position')
    plt.grid(True, alpha=0.3)
    
    # Example predictions
    plt.subplot(2, 3, 6)
    idx = np.random.choice(len(X_test), 1)[0]
    plt.plot(X_test[idx, :, 0], 'b-', alpha=0.7, label='Signal')
    actual_peak = int(y_test_actual[idx])
    pred_peak = int(y_pred_actual[idx])
    plt.axvline(actual_peak, color='g', linestyle='--', linewidth=2, 
                label=f'Actual: {actual_peak}')
    plt.axvline(pred_peak, color='r', linestyle='--', linewidth=2, 
                label=f'Pred: {pred_peak}')
    plt.title(f'Example (Error: {abs(actual_peak - pred_peak):.1f})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/media/disk_o/my_histograms/testy3.pdf")
    # plt.show()
    
    return model, history, (X_test, y_test, y_pred)


def test_model_on_new_data(model, signal_length=1000, n_tests=10):
    """
    Test the trained model on completely new data
    """
    print(f"\nTesting on {n_tests} new samples:")
    
    total_error = 0
    for i in range(n_tests):
        # Generate new test signal
        test_signal, true_peak = generate_lognormal_pulse(
            length=signal_length,
            sigma=np.random.uniform(0.3, 0.8),
            scale=np.random.uniform(1.0, 4.0),
            noise_level=np.random.uniform(0.002, 0.01)
        )
        
        # Verify actual peak
        actual_peak = find_actual_peak(test_signal)
        
        # Predict
        test_input = test_signal.reshape(1, signal_length, 1)
        pred_norm = model.predict(test_input, verbose=0)[0, 0]
        pred_peak = int(pred_norm * (signal_length - 1))
        
        error = abs(actual_peak - pred_peak)
        total_error += error
        
        print(f"  Sample {i+1}: True={actual_peak:3d}, Pred={pred_peak:3d}, Error={error:3.1f}")
    
    avg_error = total_error / n_tests
    print(f"\nAverage error on new data: {avg_error:.2f} time steps")
    
    return avg_error


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train both models for comparison
    print("="*60)
    print("Training Multi-Scale Model")
    print("="*60)
    model1, history1, test_data1 = train_model(use_alternative=False)
    test_model_on_new_data(model1)
    
    print("\n" + "="*60)
    print("Training Dilated Convolution Model")
    print("="*60)
    model2, history2, test_data2 = train_model(use_alternative=True)
    test_model_on_new_data(model2)
    
    # Save the better performing model
    model1.save('peak_detector_multiscale.h5')
    model2.save('peak_detector_dilated.h5')
    
    print("\nModels saved successfully!")