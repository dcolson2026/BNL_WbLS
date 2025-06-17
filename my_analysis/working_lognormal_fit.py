import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timing_corrections import get_1t_info, is_pulse, waveform_daisy_correction

###
# Your data (AI help)
#y_data = np.array([7., 3., -2., -0., -3., 7., 221., 767., 454., 336., 283., 168., 60., 45., 28., 23., 58.])
#y_data = np.array([8,8,8,8, 7, 3, -2., -0., -3., 7., 221., 767., 454., 336., 283., 168., 60., 45., 28., 23., 58.])
#y_data = np.array([np.float64(4.0), np.float64(-1.0), np.float64(4.0), np.float64(3.0), np.float64(4.0), np.float64(13.0), np.float64(231.0), np.float64(259.0), np.float64(86.0), np.float64(74.0), np.float64(36.0), np.float64(20.0), np.float64(8.0), np.float64(5.0), np.float64(7.0), np.float64(3.0), np.float64(3.0)])
# y_data = np.array([np.float64(-1.0), np.float64(-2.0), np.float64(-2.0), np.float64(4.0), np.float64(2.0), np.float64(5.0), np.float64(148.0), np.float64(1086.0), np.float64(749.0), np.float64(407.0), np.float64(211.0), np.float64(170.0), np.float64(230.0), np.float64(201.0), np.float64(158.0), np.float64(56.0), np.float64(42.0)])
# ANALYZING CORRECTED FILE
###

###
# from corrected root files for 500MHz sampling
# test_num = 4
# corrected_file_name = f"/home/dcolson/my_corrected_roots/test{test_num}.root"
# (
#     file_traces,
#     event_ttt1,
#     event_ttt2,
#     event_ttt3,
#     event_ttt4,
#     event_ttt5,
#     file_event_ids,
#     file_event_sanity,
#     file_daqkeys,
#     file_run_info,
# ) = get_1t_info(corrected_file_name)

# # waveforms
# some_detected_pulse_waveforms = []
# no_detected_pulse_waveforms = []
# for i in range(9991):
#     temp_waveform = waveform_daisy_correction(file_traces["adc_b1_ch1"][i], 1)
#     pulse = is_pulse(temp_waveform)
#     if pulse:
#         some_detected_pulse_waveforms.append([i, temp_waveform])
#     else:
#         no_detected_pulse_waveforms.append([i, temp_waveform])

# # t: time array
# # y: measured waveform (clean or denoised)
# event_num = 10
# y = some_detected_pulse_waveforms[event_num][1]
###

###
# from highly sampled csv files
# Path to your CSV file
csv_file_path = '/media/disk_a/wfm/2in_2/2in_2__ch1_20240604145306005.csv'

# Number of lines to skip
lines_to_skip = 14

csv_data = []

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    lines = file.readlines()

# Skip the first few lines and strip newline characters
for line in lines[lines_to_skip:]:
    # Split CSV line into fields
    fields = line.strip().split(',')
    csv_data.append(fields)

# Example csv_data
#x = [float(row[0]) for row in csv_data]
y = [float(row[1]) for row in csv_data]
###

# Estimate baseline (use pre-pulse region)
baseline = np.median(y[:100])  # assuming first 50 samples are before the pulse

# Subtract baseline
y_data = (y - baseline)*-1
peak_idx = np.argmax(y_data)
# window_min = peak_idx-7
# window_max = peak_idx + 10
window_min = peak_idx - 500
window_max = peak_idx + 900
y_data = y_data[window_min:window_max]
print(y_data)



# Simple index-based approach first
x = np.arange(len(y_data))  # Just use indices 0, 1, 2, ... 16

print(f"Data points: {len(y_data)}")
print(f"Peak amplitude: {np.max(y_data)}")
print(f"Peak at index: {np.argmax(y_data)}")

def simple_lognormal(x, A, x0, sigma):
    """
    Very simple lognormal: A * exp(-0.5 * ((ln(x - x0 + 1) - mu) / sigma)^2)
    where mu is chosen to put the peak at the right location
    """
    # Shift x to avoid log(0)
    x_shifted = x - x0 + 1.0  # Add 1 to ensure positive argument
    
    result = np.zeros_like(x)
    valid = x_shifted > 0
    
    if np.any(valid):
        # For lognormal, mode occurs at exp(mu - sigma^2)
        # We want mode at peak location, so mu = ln(peak_offset) + sigma^2
        peak_offset = np.argmax(y_data) - x0 + 1.0
        if peak_offset > 0:
            mu = np.log(peak_offset) + sigma**2
        else:
            mu = 0
            
        log_vals = np.log(x_shifted[valid])
        exponent = -0.5 * ((log_vals - mu) / sigma)**2
        # Prevent overflow
        exponent = np.clip(exponent, -50, 50)
        result[valid] = A * np.exp(exponent)
    
    return result

# Much simpler initial guesses
A_init = np.max(y_data)
#x0_init = 6.0  # Start just before the peak (peak is at index 7)
x0_init = np.argmax(y_data) - 1
sigma_init = 0.5

p0 = [A_init, x0_init, sigma_init]
print(f"Initial parameters: A={A_init}, x0={x0_init}, sigma={sigma_init}")

# Add bounds to keep parameters reasonable
bounds = ([0, 0, 0.1], [2000, 1000, 2.0])

try:
    popt, pcov = curve_fit(simple_lognormal, x, y_data, p0=p0, bounds=bounds, maxfev=10000)
    
    print(f"\nFit results:")
    print(f"A = {popt[0]:.1f}")
    print(f"x0 = {popt[1]:.2f}")
    print(f"sigma = {popt[2]:.3f}")
    
    # Calculate fit and R-squared
    y_fit = simple_lognormal(x, *popt)
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared:.4f}")
    
    # Parameter uncertainties
    if pcov is not None and np.all(np.isfinite(pcov)):
        param_errors = np.sqrt(np.diag(pcov))
        print(f"\nParameter uncertainties:")
        print(f"A ± {param_errors[0]:.1f}")
        print(f"x0 ± {param_errors[1]:.3f}")
        print(f"sigma ± {param_errors[2]:.3f}")
    else:
        print("Covariance matrix issues - but fit parameters look reasonable")
    
    fit_success = True
    
except Exception as e:
    print(f"Fitting failed: {e}")
    popt = p0
    y_fit = simple_lognormal(x, *popt)
    fit_success = False

# Alternative: try scipy's lognorm distribution
from scipy import stats

print("\n" + "="*50)
print("Alternative approach using scipy.stats.lognorm:")

# Shift data to be positive and normalize
y_positive = y_data - np.min(y_data) + 1
peak_idx = np.argmax(y_positive)

# Fit using scipy's lognorm
try:
    # Use only the data from start to slightly past peak for better fit
    end_idx = min(peak_idx + 8, len(y_positive))
    x_fit = x[:end_idx]
    y_fit_data = y_positive[:end_idx]
    
    # Normalize weights by distance from peak for better fit
    weights = np.exp(-0.5 * ((x_fit - peak_idx) / 3)**2)
    
    # Fit lognormal parameters
    shape, loc, scale = stats.lognorm.fit(y_fit_data, floc=0)
    loc = 6 ###
    
    # Generate fitted curve
    y_scipy = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
    # Scale to match data amplitude
    y_scipy = y_scipy * np.max(y_positive) / np.max(y_scipy)
    # Shift back to original baseline
    y_scipy = y_scipy + np.min(y_data) - 1
    
    print(f"Scipy lognorm parameters: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")
    
    # Calculate R-squared for scipy fit
    ss_res_scipy = np.sum((y_data - y_scipy) ** 2)
    r_squared_scipy = 1 - (ss_res_scipy / ss_tot)
    print(f"Scipy R² = {r_squared_scipy:.4f}")
    
    scipy_success = True
    
except Exception as e:
    print(f"Scipy approach failed: {e}")
    y_scipy = np.zeros_like(y_data)
    scipy_success = False

# Plot results
plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(x, y_data, 'bo-', label='Data', markersize=6, linewidth=2)

if fit_success:
    x_smooth = np.linspace(0, len(y_data)-1, 100)
    y_smooth = simple_lognormal(x_smooth, *popt)
    plt.plot(x_smooth, y_smooth, 'r-', label=f'Custom lognormal fit (R²={r_squared:.3f})', linewidth=2)

if scipy_success:
    pass
    # plt.plot(x, y_scipy, 'g--', label=f'Scipy lognormal fit (R²={r_squared_scipy:.3f})', linewidth=2)

plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('PMT Pulse Lognormal Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 1, 2)
if fit_success:
    residuals = y_data - simple_lognormal(x, *popt)
    plt.plot(x, residuals, 'ro-', label='Custom fit residuals', markersize=4)

if scipy_success:
    pass
    # residuals_scipy = y_data - y_scipy
    # plt.plot(x, residuals_scipy, 'g^-', label='Scipy fit residuals', markersize=4)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Residuals')
plt.title('Fit Residuals')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/media/disk_o/my_histograms/pulse_fit_improved3.pdf")
plt.show()

print(f"\nConclusion:")
if fit_success and r_squared > 0.8:
    print("✓ Custom lognormal fit successful!")
elif scipy_success and r_squared_scipy > 0.8:
    print("✓ Scipy lognormal fit successful!")
else:
    print("⚠ Both fits struggled - PMT pulse might need different model")
    print("Consider: double exponential, gamma function, or asymmetric models")