import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

def generate_precise_lognormal_pulse(length=1000, peak_time=None, sigma=0.5, scale=1.0, 
                                   noise_level=0.01, oversampling=50):
    """
    Generate ultra-high precision synthetic pulse using analytical approach
    """
    if peak_time is None:
        peak_time = np.random.uniform(length * 0.25, length * 0.75)
    
    # Create high-resolution analytical pulse
    t_fine = np.linspace(0, length, length * oversampling)
    dt_fine = t_fine[1] - t_fine[0]
    
    # Generate analytical lognormal pulse
    t_shifted = t_fine - peak_time
    pulse_fine = np.zeros(len(t_fine))
    
    # Only compute for positive times to avoid log(0)
    valid_mask = t_shifted > 0.01
    if np.any(valid_mask):
        t_valid = t_shifted[valid_mask]
        pulse_fine[valid_mask] = (1 / (t_valid * sigma * np.sqrt(2 * np.pi))) * \
                               np.exp(-0.5 * ((np.log(t_valid) - np.log(scale)) / sigma) ** 2)
    
    # Normalize to unit amplitude
    if np.max(pulse_fine) > 0:
        pulse_fine = pulse_fine / np.max(pulse_fine)
    
    # Proper anti-aliasing downsampling using integration
    pulse_sampled = np.zeros(length)
    samples_per_point = oversampling
    
    for i in range(length):
        start_idx = i * samples_per_point
        end_idx = (i + 1) * samples_per_point
        if end_idx <= len(pulse_fine):
            # Integrate over the sampling interval (proper anti-aliasing)
            pulse_sampled[i] = np.trapz(pulse_fine[start_idx:end_idx], dx=dt_fine) / (samples_per_point * dt_fine)
    
    # Add realistic noise
    if noise_level > 0:
        pulse_sampled += np.random.normal(0, noise_level, length)
    
    return pulse_sampled, peak_time

def find_analytical_peak(signal, initial_guess, search_range=10):
    """
    Find peak using analytical curve fitting around CNN prediction
    """
    center_idx = int(np.round(initial_guess))
    start_idx = max(0, center_idx - search_range)
    end_idx = min(len(signal), center_idx + search_range + 1)
    
    # Extract local region
    local_signal = signal[start_idx:end_idx]
    local_indices = np.arange(start_idx, end_idx)
    
    if len(local_signal) < 5:
        return initial_guess
    
    # Find the maximum in the local region first
    local_max_idx = np.argmax(local_signal)
    global_max_idx = start_idx + local_max_idx
    
    # Use quadratic interpolation around the maximum
    if global_max_idx > 0 and global_max_idx < len(signal) - 1:
        # Get three points around the maximum
        x_points = np.array([global_max_idx - 1, global_max_idx, global_max_idx + 1])
        y_points = np.array([signal[global_max_idx - 1], signal[global_max_idx], signal[global_max_idx + 1]])
        
        # Fit quadratic: y = ax² + bx + c
        A = np.vstack([x_points**2, x_points, np.ones(len(x_points))]).T
        try:
            coeffs = np.linalg.lstsq(A, y_points, rcond=None)[0]
            a, b, c = coeffs
            
            # Find peak of quadratic: x = -b/(2a)
            if a < 0:  # Ensure it's a maximum
                peak_x = -b / (2 * a)
                # Verify peak is reasonable
                if abs(peak_x - global_max_idx) < 2:
                    return peak_x
        except:
            pass
    
    return float(global_max_idx)

def create_optimized_cnn(input_length):
    """
    Optimized CNN architecture for sub-sample precision
    """
    # Input layer
    inputs = keras.Input(shape=(input_length, 1))
    
    # Multi-scale feature extraction
    # Scale 1: Large receptive field
    conv1 = layers.Conv1D(32, 31, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv1D(32, 31, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling1D(2)(conv1)
    
    # Scale 2: Medium receptive field
    conv2 = layers.Conv1D(64, 15, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv1D(64, 15, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling1D(2)(conv2)
    
    # Scale 3: Fine features
    conv3 = layers.Conv1D(128, 7, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv1D(128, 7, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling1D(2)(conv3)
    
    # Scale 4: Very fine features
    conv4 = layers.Conv1D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv1D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    
    # Global and local feature combination
    # global_pool = layers.GlobalAveragePooling1D()(conv4)
    # global_max = layers.GlobalMaxPooling1D()(conv4)
    
    # Combine global features
    # combined = layers.Concatenate()([global_pool, global_max])

    
    # Dense layers for regression
    dense1 = layers.Dense(512, activation='relu')
    dense1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    dense2 = layers.Dropout(0.2)(dense2)
    dense3 = layers.Dense(128, activation='relu')(dense2)
    dense3 = layers.Dropout(0.1)(dense3)
    
    # Output layer
    output = layers.Dense(1, activation='linear')(dense3)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Use Huber loss for robustness to outliers
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(delta=1.0),  # More robust than MSE
        metrics=['mae']
    )
    
    return model

def generate_optimized_dataset(n_samples=100000, signal_length=1000):
    """
    Generate high-quality dataset with verified peak positions
    """
    X = np.zeros((n_samples, signal_length, 1))
    y = np.zeros(n_samples)
    
    print("Generating optimized dataset...")
    for i in range(n_samples):
        if i % 10000 == 0:
            print(f"Generated {i}/{n_samples} samples")
        
        # More controlled parameter variation
        sigma = np.random.uniform(0.3, 0.8)
        scale = np.random.uniform(0.5, 2.0)
        noise_level = np.random.uniform(0.001, 0.01)  # Very low noise
        
        signal, true_peak = generate_precise_lognormal_pulse(
            length=signal_length,
            sigma=sigma,
            scale=scale,
            noise_level=noise_level,
            oversampling=100  # Higher oversampling
        )
        
        X[i, :, 0] = signal
        y[i] = true_peak
        
        # Verify the peak is reasonable
        if true_peak < 0 or true_peak >= signal_length:
            print(f"Warning: Peak {true_peak} out of bounds for sample {i}")
    
    return X, y

def train_optimized_model():
    """
    Train the optimized model with better hyperparameters
    """
    print("Training optimized sub-nanosecond model...")
    signal_length = 1000
    
    # Generate larger, higher-quality dataset
    X, y = generate_optimized_dataset(n_samples=100000, signal_length=signal_length)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    model = create_optimized_cnn(signal_length)
    
    # Optimized callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpointing
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train with optimized parameters
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Load best model
    model = keras.models.load_model('best_model_checkpoint.h5')
    
    # Comprehensive evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    # Calculate errors
    sample_errors = np.abs(y_test - y_pred)
    
    # Convert to time
    sampling_rate = 500e6
    time_per_sample = 1 / sampling_rate * 1e9  # ns per sample
    time_errors_ns = sample_errors * time_per_sample
    
    # Statistics
    print(f"\nOptimized Model Results:")
    print(f"Mean error: {np.mean(sample_errors):.4f} samples ({np.mean(time_errors_ns):.3f} ns)")
    print(f"Median error: {np.median(sample_errors):.4f} samples ({np.median(time_errors_ns):.3f} ns)")
    print(f"Std error: {np.std(sample_errors):.4f} samples ({np.std(time_errors_ns):.3f} ns)")
    print(f"95th percentile: {np.percentile(sample_errors, 95):.4f} samples ({np.percentile(time_errors_ns, 95):.3f} ns)")
    print(f"Sub-nanosecond accuracy: {np.sum(time_errors_ns < 1.0) / len(time_errors_ns) * 100:.1f}%")
    
    # Test with analytical refinement
    print("\nTesting with analytical peak finding...")
    n_test_cases = 100
    cnn_errors = []
    refined_errors = []
    
    for i in range(n_test_cases):
        # Generate test signal
        test_signal, true_peak = generate_precise_lognormal_pulse(
            length=1000, 
            noise_level=0.005,
            oversampling=100
        )
        
        # CNN prediction
        test_input = test_signal.reshape(1, 1000, 1)
        cnn_pred = model.predict(test_input, verbose=0)[0, 0]
        
        # Analytical refinement
        refined_pred = find_analytical_peak(test_signal, cnn_pred, search_range=5)
        
        # Calculate errors
        cnn_error = abs(true_peak - cnn_pred) * time_per_sample
        refined_error = abs(true_peak - refined_pred) * time_per_sample
        
        cnn_errors.append(cnn_error)
        refined_errors.append(refined_error)
    
    print(f"CNN mean error: {np.mean(cnn_errors):.3f} ns")
    print(f"Refined mean error: {np.mean(refined_errors):.3f} ns")
    print(f"Improvement: {(np.mean(cnn_errors) - np.mean(refined_errors)):.3f} ns")
    
    # Detailed example
    test_signal, true_peak = generate_precise_lognormal_pulse(length=1000, noise_level=0.005)
    test_input = test_signal.reshape(1, 1000, 1)
    cnn_pred = model.predict(test_input, verbose=0)[0, 0]
    refined_pred = find_analytical_peak(test_signal, cnn_pred)
    
    cnn_error_ns = abs(true_peak - cnn_pred) * time_per_sample
    refined_error_ns = abs(true_peak - refined_pred) * time_per_sample
    
    print(f"\nDetailed Example:")
    print(f"True peak: {true_peak:.6f} samples")
    print(f"CNN prediction: {cnn_pred:.6f} samples (error: {cnn_error_ns:.3f} ns)")
    print(f"Refined prediction: {refined_pred:.6f} samples (error: {refined_error_ns:.3f} ns)")
    
    # Plot the test case
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(test_signal)
    plt.axvline(x=true_peak, color='g', linestyle='-', linewidth=2, label=f'True: {true_peak:.3f}')
    plt.axvline(x=cnn_pred, color='r', linestyle='--', linewidth=2, label=f'CNN: {cnn_pred:.3f}')
    plt.axvline(x=refined_pred, color='b', linestyle=':', linewidth=2, label=f'Refined: {refined_pred:.3f}')
    plt.title('Peak Detection Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(time_errors_ns, bins=50, alpha=0.7)
    plt.axvline(x=1.0, color='r', linestyle='--', label='1 ns')
    plt.xlabel('Error (ns)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 2, 4)
    plt.scatter(cnn_errors[:50], refined_errors[:50], alpha=0.6)
    plt.plot([0, max(cnn_errors[:50])], [0, max(cnn_errors[:50])], 'r--', label='No improvement')
    plt.xlabel('CNN Error (ns)')
    plt.ylabel('Refined Error (ns)')
    plt.title('CNN vs Refined Errors')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/media/disk_o/my_histograms/hyper.pdf")
    # plt.show()
    
    return model, history

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train optimized model
    model, history = train_optimized_model()
    
    # Save the model
    model.save('ultra_precision_pulse_detector.h5')
    print("\nModel saved as 'ultra_precision_pulse_detector.h5'")
