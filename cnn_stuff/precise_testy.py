import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def generate_high_precision_pulse(length=1000, peak_time=None, sigma=0.5, scale=1.0, 
                                noise_level=0.01, sampling_rate=500e6):
    """
    Generate synthetic pulse with sub-sample precision peak timing
    
    Args:
        length: Length of the signal
        peak_time: Continuous peak time (can be fractional)
        sigma: Shape parameter of lognormal distribution
        scale: Scale parameter
        noise_level: Noise level
        sampling_rate: Sampling rate in Hz
    
    Returns:
        signal: Sampled signal
        peak_time: Actual continuous peak time
    """
    if peak_time is None:
        # Allow fractional peak times
        # peak_time = np.random.uniform(length // 4, 3 * length // 4)
        peak_time = np.random.uniform(20, length-20)
    
    # High-resolution time array for accurate pulse generation
    oversampling_factor = 10
    t_high_res = np.linspace(0, length, length * oversampling_factor)
    
    # Generate high-resolution pulse
    t_shifted = t_high_res - peak_time
    signal_high_res = np.zeros(len(t_high_res))
    positive_mask = t_shifted > 0.1  # Avoid division by zero
    
    if np.any(positive_mask):
        signal_high_res[positive_mask] = (1 / (t_shifted[positive_mask] * sigma * np.sqrt(2 * np.pi))) * \
                                       np.exp(-0.5 * ((np.log(t_shifted[positive_mask]) - np.log(scale)) / sigma) ** 2)
    
    # Normalize
    if np.max(signal_high_res) > 0:
        signal_high_res = signal_high_res / np.max(signal_high_res)
    
    # Downsample to original sampling rate (anti-aliasing)
    signal = np.zeros(length)
    for i in range(length):
        start_idx = i * oversampling_factor
        end_idx = (i + 1) * oversampling_factor
        signal[i] = np.mean(signal_high_res[start_idx:end_idx])
    
    # Add noise
    signal += np.random.normal(0, noise_level, length)
    
    return signal, peak_time

def create_sub_ns_cnn_model(input_length):
    """
    Enhanced CNN model for sub-sample precision
    """
    model = keras.Sequential([
        # First block - capture broad features
        layers.Conv1D(filters=64, kernel_size=15, activation='relu', 
                      input_shape=(input_length, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=64, kernel_size=15, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Second block - medium resolution features
        layers.Conv1D(filters=128, kernel_size=11, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=11, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Third block - fine features
        layers.Conv1D(filters=256, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=256, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Fourth block - very fine features
        layers.Conv1D(filters=512, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=512, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Fifth block - ultra-fine features
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer for continuous prediction
        layers.Dense(1, activation='linear')  # Linear for unrestricted continuous output
    ])
    
    # Use a more precise loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def generate_high_precision_dataset(n_samples=50000, signal_length=1000):
    """
    Generate dataset with continuous peak times for sub-sample precision
    """
    X = np.zeros((n_samples, signal_length, 1))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Vary parameters
        sigma = np.random.uniform(0.2, 1.0)
        scale = np.random.uniform(0.3, 3.0)
        noise_level = np.random.uniform(0.005, 0.02)  # Lower noise for precision
        
        signal, peak_time = generate_high_precision_pulse(
            length=signal_length,
            sigma=sigma,
            scale=scale,
            noise_level=noise_level
        )
        
        X[i, :, 0] = signal
        # Store actual continuous peak time (not normalized)
        y[i] = peak_time
    
    return X, y

def train_high_precision_model():
    """
    Train model for sub-nanosecond precision
    """
    print("Generating high-precision dataset...")
    signal_length = 1000
    n_samples = 50000  # More samples for better precision
    
    X, y = generate_high_precision_dataset(n_samples, signal_length)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create enhanced model
    model = create_sub_ns_cnn_model(signal_length)
    model.summary()
    
    # Enhanced callbacks for precision
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=7,
        min_lr=1e-8
    )
    
    # Train with smaller batch size for better precision
    history = model.fit(
        X_train, y_train,
        batch_size=16,  # Smaller batch size
        epochs=150,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Calculate errors in samples and time
    sample_error = np.abs(y_test - y_pred.flatten())
    
    # Convert to time units (500 MHz = 2 ns per sample)
    sampling_rate = 500e6  # Hz
    time_per_sample = 1 / sampling_rate  # 2 ns
    time_error_ns = sample_error * time_per_sample * 1e9  # Convert to ns
    
    # Statistics
    mean_error_samples = np.mean(sample_error)
    std_error_samples = np.std(sample_error)
    mean_error_ns = np.mean(time_error_ns)
    std_error_ns = np.std(time_error_ns)
    
    print(f"\nHigh-Precision Results:")
    print(f"Mean error: {mean_error_samples:.3f} ± {std_error_samples:.3f} samples")
    print(f"Mean error: {mean_error_ns:.3f} ± {std_error_ns:.3f} ns")
    print(f"Median error: {np.median(time_error_ns):.3f} ns")
    print(f"95th percentile error: {np.percentile(time_error_ns, 95):.3f} ns")
    print(f"Samples with sub-ns precision: {np.sum(time_error_ns < 1.0) / len(time_error_ns) * 100:.1f}%")
    
    # Plot error distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(time_error_ns, bins=50, alpha=0.7)
    plt.axvline(x=1.0, color='r', linestyle='--', label='1 ns threshold')
    plt.xlabel('Timing Error (ns)')
    plt.ylabel('Count')
    plt.title('Distribution of Timing Errors')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(sample_error, bins=50, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='0.5 sample threshold')
    plt.xlabel('Sample Error')
    plt.ylabel('Count')
    plt.title('Distribution of Sample Errors')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/media/disk_o/my_histograms/precise_cnn.pdf")
    # plt.show()
    
    return model, history

# Additional technique: Parabolic interpolation post-processing
def parabolic_interpolation_refinement(signal, cnn_prediction):
    """
    Refine CNN prediction using parabolic interpolation around the predicted peak
    """
    pred_idx = int(np.round(cnn_prediction))
    
    # Ensure we have points around the prediction
    if pred_idx > 0 and pred_idx < len(signal) - 1:
        # Get three points around prediction
        y1, y2, y3 = signal[pred_idx-1], signal[pred_idx], signal[pred_idx+1]
        
        # Parabolic interpolation formula
        a = ((y1 - y2) + (y3 - y2)) / 2
        b = (y3 - y1) / 2
        
        if a != 0:
            # Sub-sample correction
            correction = -b / (2 * a)
            refined_peak = pred_idx + correction
            return refined_peak
    
    return cnn_prediction

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model, history = train_high_precision_model()
    
    # Test parabolic refinement
    print("\nTesting with parabolic interpolation refinement...")
    test_signal, true_peak = generate_high_precision_pulse(length=1000)
    test_signal_reshaped = test_signal.reshape(1, 1000, 1)
    
    cnn_pred = model.predict(test_signal_reshaped, verbose=0)[0, 0]
    refined_pred = parabolic_interpolation_refinement(test_signal, cnn_pred)
    
    sampling_rate = 500e6
    time_per_sample = 1 / sampling_rate * 1e9  # ns per sample
    
    cnn_error_ns = abs(true_peak - cnn_pred) * time_per_sample
    refined_error_ns = abs(true_peak - refined_pred) * time_per_sample
    
    print(f"True peak: {true_peak:.3f} samples")
    print(f"CNN prediction: {cnn_pred:.3f} samples (error: {cnn_error_ns:.3f} ns)")
    print(f"Refined prediction: {refined_pred:.3f} samples (error: {refined_error_ns:.3f} ns)")
