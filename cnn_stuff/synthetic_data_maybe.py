"""Building 1D CNN to find the time of the pulse's maximum."""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def lognormal_pulse(t, t0=10, A=1.0, mu=1.0, sigma=0.2):
    """Generates a log-normal pulse centered around t0."""
    t_shifted = t - t0
    pulse = np.zeros_like(t)
    valid = t_shifted > 0
    pulse[valid] = A * np.exp(-(np.log(t_shifted[valid]) - mu)**2 / (2 * sigma**2))
    return pulse

def generate_pulse(sampling_rate=1e9, duration=100e-9, t0=30e-9, A=1.0, mu=1.0, sigma=0.2, noise_sigma=300e-6):
    """
    Generate a single digitized PMT pulse.
    
    sampling_rate: in Hz
    duration: total signal duration in seconds
    t0: true time of the pulse maximum (sec)
    A: amplitude of the pulse
    noise_sigma: standard deviation of the Gaussian noise (V)
    """
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)
    pulse = lognormal_pulse(t, t0=t0, A=A, mu=mu, sigma=sigma)
    noise = np.random.normal(0, noise_sigma, size=pulse.shape)
    signal = pulse + noise
    return t, signal, t0

def create_dataset(num_samples=10000, sampling_rate=1e9, duration=100e-9):
    X = []
    y = []
    for _ in range(num_samples):
        t0 = np.random.uniform(20e-9, 80e-9)  # randomly choose pulse time
        A = np.random.uniform(0.5, 2.0)       # randomly scale amplitude
        _, signal, true_t0 = generate_pulse(sampling_rate=sampling_rate, duration=duration, t0=t0, A=A)
        X.append(signal)
        y.append(true_t0)
    return np.array(X), np.array(y)

# Example usage
t, signal, true_t0 = generate_pulse(t0=45e-9, A=1.2)
plt.plot(t * 1e9, signal * 1e6)
plt.axvline(true_t0 * 1e9, color='r', linestyle='--', label='True max time')
plt.title("Simulated PMT Pulse with Noise")
plt.xlabel("Time (ns)")
plt.ylabel("Signal (µV)")
plt.legend()
plt.grid()

plt.savefig("/home/dcolson/my_histograms/1d_cnnn_test.pdf")
plt.close()




# # Generate data
# X_train, y_train = create_dataset(num_samples=10000)



# X_train = X_train[..., np.newaxis]  # shape becomes (10000, N, 1)
