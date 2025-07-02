import numpy as np
import matplotlib.pyplot as plt

def generate_lognormal_pulse(length=100, mu=2.0, sigma=0.4):
    x = np.linspace(0, 1, length)
    t_peak = np.random.uniform(0.2, 0.8)
    pulse = np.exp(-((np.log(x + 1e-4) - np.log(t_peak + 1e-4) - mu)**2) / (2 * sigma**2))
    pulse /= np.max(pulse)  # Normalize
    peak_index = np.argmax(pulse)
    return pulse, peak_index


# Example: Generate and plot a pulse
pulse, peak_index = generate_lognormal_pulse()
plt.plot(pulse)
plt.title(f"Peak at index {peak_index}")
plt.savefig("/media/disk_o/my_histograms/poop.pdf")
plt.close()


def create_dataset(n_samples=1000, length=100):
    X = []
    y = []
    for _ in range(n_samples):
        pulse, peak = generate_lognormal_pulse(length=length)
        X.append(pulse)
        y.append(peak)
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    return X, y

X_train, y_train = create_dataset(n_samples=5000)
X_val, y_val = create_dataset(n_samples=1000)
