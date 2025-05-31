#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for projection='3d'
import numpy as np

# Load the dataset
# Create a string buffer with the provided CSV data

csv_data =(r"C:\Users\ASUS\Downloads\data(1)\cleaned_dataset\metadata.csv")
df = pd.read_csv(csv_data)



# Filter for impedance data for a specific battery (e.g., B0047) at a specific temperature
battery_to_plot = 'B0047'
# Ambient temperature for impedance tests is usually 24 in this dataset
ambient_temp_filter = 24

df_impedance_filtered = df[
    (df['type'] == 'impedance') &
    (df['battery_id'] == battery_to_plot) &
    (df['ambient_temperature'] == ambient_temp_filter)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Convert relevant columns to numeric, coercing errors for non-numeric entries
# This will turn problematic entries (like complex numbers as strings, or empty strings) into NaN
df_impedance_filtered['Re_numeric'] = pd.to_numeric(df_impedance_filtered['Re'], errors='coerce')
df_impedance_filtered['Rct_numeric'] = pd.to_numeric(df_impedance_filtered['Rct'], errors='coerce')

# 'test_id' will serve as the cycle count / aging proxy
df_impedance_filtered['cycle_count'] = pd.to_numeric(df_impedance_filtered['test_id'], errors='coerce')

# Drop rows where conversion to numeric failed for key plotting columns
df_plot_data = df_impedance_filtered.dropna(subset=['Re_numeric', 'Rct_numeric', 'cycle_count'])

# Sort by cycle_count to ensure the line plot connects points in chronological order
df_plot_data = df_plot_data.sort_values(by='cycle_count')

print(f"Number of data points for {battery_to_plot} at {ambient_temp_filter}C: {len(df_plot_data)}")
if len(df_plot_data) == 0:
    print("No suitable data to plot. Please check filters or data integrity.")
else:
    # Extract data for plotting
    x_re_z = df_plot_data['Re_numeric']
    # Using Rct as a proxy for the magnitude of the imaginary part of impedance
    # In typical Nyquist plots, -Im(Z) is plotted on Y. Rct is a resistance,
    # usually positive, related to the semicircle diameter.
    y_im_z_proxy = df_plot_data['Rct_numeric']
    z_cycle = df_plot_data['cycle_count']

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data as a line to show progression with aging
    # A scatter plot with color mapping to cycle count can also be effective
    # ax.scatter(x_re_z, y_im_z_proxy, z_cycle, c=z_cycle, cmap='viridis', marker='o')
    ax.plot(x_re_z, y_im_z_proxy, z_cycle, marker='o', linestyle='-', label=f'Battery {battery_to_plot} EIS trend')

    # Set labels and title
    ax.set_xlabel('Re(Z) / Ohm (from Re column)')
    ax.set_ylabel('Proxy for |Im(Z)| / Ohm (from Rct column)')  # Rct is a resistance, usually positive
    ax.set_zlabel('Aging (Cycle Count / test_id)')
    ax.set_title(
        f'3D EIS Plot: Impedance vs. Aging for Battery {battery_to_plot}\n(Ambient Temp: {ambient_temp_filter}°C)')

    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nFirst 5 data points used for plotting:")
    print(df_plot_data[['cycle_count', 'Re_numeric', 'Rct_numeric']].head())


# In[ ]:





# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter
from mpl_toolkits.mplot3d import Axes3D


# --- Function to generate synthetic single-cycle data and ICA ---
def generate_synthetic_ica(q_points_base, v_points_base, cycle_num=0, total_cycles=100, fine_points=1000):
    """
    Generates a synthetic discharge curve and its ICA (dQ/dV vs V).
    Simulates aging by slightly modifying V and Q points based on cycle_num.
    """
    # Simulate aging:
    # 1. Capacity fade: Reduce total capacity slightly with each cycle
    capacity_fade_factor = 1 - (cycle_num / total_cycles) * 0.3  # Max 30% fade
    q_points = q_points_base * capacity_fade_factor

    # 2. Voltage droop: Lower plateau voltages slightly
    voltage_droop = (cycle_num / total_cycles) * 0.1  # Max 0.1V droop overall
    v_points = v_points_base - voltage_droop
    # Ensure end voltage doesn't go too low unrealistically
    v_points = np.maximum(v_points, v_points_base[-1] - 0.2)

    if len(q_points) < 2 or len(v_points) < 2 or not np.all(np.isfinite(q_points)) or not np.all(np.isfinite(v_points)):
        print(f"Warning: Invalid q_points or v_points for cycle {cycle_num}. Skipping.")
        return None, None, None

    # Ensure Q is monotonically increasing and V is monotonically decreasing
    # For PchipInterpolator, x must be strictly increasing.
    # Sort by Q first, then handle V if needed (though V should naturally follow for discharge)
    sort_indices = np.argsort(q_points)
    q_points_sorted = q_points[sort_indices]
    v_points_sorted = v_points[sort_indices]

    # Remove duplicate Q points if any, keeping the first occurrence for V
    unique_q_indices = np.unique(q_points_sorted, return_index=True)[1]
    if len(unique_q_indices) < 2:  # Need at least 2 points for interpolation
        print(f"Warning: Not enough unique Q points for cycle {cycle_num} after sorting. Skipping.")
        return None, None, None

    q_points_unique = q_points_sorted[unique_q_indices]
    v_points_unique = v_points_sorted[unique_q_indices]

    # Interpolate to get smooth, dense V-Q curve
    # We need to ensure Q is strictly increasing for PchipInterpolator
    if not np.all(np.diff(q_points_unique) > 0):
        # If Q is not strictly increasing after unique, it's problematic.
        # This might happen if capacity fade is too aggressive making Q_points very close or non-monotonic.
        print(f"Warning: Q_points not strictly increasing for cycle {cycle_num}. Skipping.")
        return None, None, None

    interpolator = PchipInterpolator(q_points_unique, v_points_unique)
    q_fine = np.linspace(q_points_unique.min(), q_points_unique.max(), fine_points)
    v_fine = interpolator(q_fine)

    # Calculate dQ/dV
    # We want V on x-axis, dQ/dV on y-axis.
    # For discharge: Q increases, V decreases.
    # To get positive peaks for -dQ/dV, we can:
    # 1. Calculate dQ/dV (will be negative) and then plot its negation.
    # 2. Or, use V_increasing and corresponding Q_decreasing.

    # Method 1:
    dv = np.diff(v_fine)
    dq = np.diff(q_fine)

    # Voltage for plotting dQ/dV (midpoints)
    v_for_dqdv = (v_fine[:-1] + v_fine[1:]) / 2

    # Filter out points where dV is zero or positive (for discharge) to avoid infinity or wrong sign
    # and ensure dQ is positive
    valid_indices = (dv < -1e-6) & (dq > 1e-9)  # dV should be negative, dQ positive

    if not np.any(valid_indices):
        # print(f"Warning: No valid dV segments for ICA in cycle {cycle_num}. Skipping.")
        return None, None, None

    dq_dv = dq[valid_indices] / dv[valid_indices]
    v_for_dqdv_filtered = v_for_dqdv[valid_indices]

    # We want positive peaks for -dQ/dV
    ica_peaks_positive = -dq_dv

    # Smooth the ICA curve (optional, but good for real data, can help with synthetic too)
    if len(ica_peaks_positive) > 11:  # SavGol filter needs window < data length
        window_length = min(11,
                            len(ica_peaks_positive) - (1 if (len(ica_peaks_positive) % 2 == 0) else 0))  # must be odd
        ica_peaks_positive_smooth = savgol_filter(ica_peaks_positive, window_length, 3)
    else:
        ica_peaks_positive_smooth = ica_peaks_positive

    return v_fine, q_fine, v_for_dqdv_filtered, ica_peaks_positive_smooth


# --- Define Base Q and V points for a "fresh" battery ---
# These points are chosen to create characteristic plateaus in the V-Q curve,
# which will translate to peaks in the dQ/dV vs V curve.
q_points_base = np.array([0, 0.05, 0.1, 0.3, 0.4, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5])  # Capacity in Ah
v_points_base = np.array([4.2, 4.15, 3.95, 3.85, 3.83, 3.75, 3.73, 3.55, 3.53, 3.35, 3.0, 2.5])  # Voltage in V

# --- Generate and Plot ICA for the first (fresh) cycle (Plot b) ---
v_cycle0, q_cycle0, v_for_ica0, ica0 = generate_synthetic_ica(q_points_base, v_points_base, cycle_num=0)

if v_for_ica0 is not None:
    plt.figure(figsize=(10, 6))

    # Plot a) Voltage vs Capacity (simulated)
    ax1 = plt.subplot(2, 1, 1)
    if v_cycle0 is not None:
        ax1.plot(q_cycle0, v_cycle0, label='Synthetic Discharge Curve (Cycle 0)')
    ax1.set_xlabel('Capacity (Ah)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('a) Synthesized Voltage vs. Capacity Profile')
    ax1.grid(True)
    ax1.legend()

    # Plot b) dQ/dV vs V
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(v_for_ica0, ica0, label='dQ/dV vs V (Cycle 0)')
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('-dQ/dV (Ah/V)')
    ax2.set_title('b) Incremental Capacity Analysis (ICA)')
    ax2.grid(True)
    ax2.legend()
    # Optional: invert x-axis for typical discharge ICA view
    # ax2.invert_xaxis()

    plt.tight_layout()
    plt.show()
else:
    print("Could not generate ICA for the initial cycle.")

# --- Part 2: 3D Plot of Peak Evolution with Aging ---
num_simulated_cycles = 50
all_peak_voltages = []
all_peak_heights = []
all_cycle_numbers_for_peaks = []

# Define initial voltage windows to track specific peaks (approximate based on expected plateaus)
# These windows might need adjustment based on how the synthetic aging shifts peaks.
peak_windows_initial_V = [(3.7, 3.9), (3.4, 3.6)]
num_peaks_to_track = len(peak_windows_initial_V)

for cycle_n in range(num_simulated_cycles):
    v_cycle, q_cycle, v_for_ica, ica_curve = generate_synthetic_ica(
        q_points_base, v_points_base,
        cycle_num=cycle_n,
        total_cycles=num_simulated_cycles
    )

    if v_for_ica is None or ica_curve is None or len(ica_curve) < 3:
        # print(f"Skipping cycle {cycle_n} for peak analysis due to insufficient ICA data.")
        continue

    # Find peaks in the current cycle's ICA curve
    # Height and distance parameters might need tuning for real data
    peaks_indices, properties = find_peaks(ica_curve, height=0.1, distance=5)

    if len(peaks_indices) > 0:
        current_cycle_peak_V = v_for_ica[peaks_indices]
        current_cycle_peak_H = ica_curve[peaks_indices]

        # Attempt to track the N most prominent peaks or peaks within evolving windows
        # For simplicity here, let's take up to num_peaks_to_track most prominent ones
        # and assume their relative order roughly corresponds to the initial windows

        sorted_peaks_indices = np.argsort(current_cycle_peak_H)[::-1]  # Sort by height, descending

        for i in range(min(num_peaks_to_track, len(sorted_peaks_indices))):
            peak_idx_in_sorted = sorted_peaks_indices[i]
            actual_peak_idx = peaks_indices[peak_idx_in_sorted]  # original index in ica_curve

            # Heuristic: Match to initial peak windows based on current voltage
            # This is a very simplified tracking mechanism
            v_peak = v_for_ica[actual_peak_idx]
            h_peak = ica_curve[actual_peak_idx]

            # Simplistic association: if we have two windows, assign first found prominent peak to first window etc.
            # This is a placeholder for more robust peak tracking.
            # For this demo, just store them in order found (most prominent first)
            all_peak_voltages.append(v_peak)
            all_peak_heights.append(h_peak)
            all_cycle_numbers_for_peaks.append(cycle_n)

if not all_peak_voltages:
    print("\nNo peaks were successfully tracked across simulated cycles for the 3D plot.")
else:
    # Create 3D plot for peak evolution
    fig_3d = plt.figure(figsize=(12, 9))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Scatter plot of all detected peaks
    # Color by cycle number to see evolution
    scatter = ax_3d.scatter(all_peak_voltages, all_peak_heights, all_cycle_numbers_for_peaks,
                            c=all_cycle_numbers_for_peaks, cmap='viridis', marker='o', s=20)

    ax_3d.set_xlabel('Peak Voltage (V)')
    ax_3d.set_ylabel('Peak -dQ/dV (Ah/V)')
    ax_3d.set_zlabel('Cycle Number (Aging)')
    ax_3d.set_title('3D Plot: Evolution of ICA Peaks with Synthetic Aging')

    # Add a color bar
    cbar = fig_3d.colorbar(scatter, ax=ax_3d, shrink=0.5, aspect=10)
    cbar.set_label('Cycle Number')

    plt.tight_layout()
    plt.show()

    print(f"\nCollected {len(all_peak_voltages)} peak points across {num_simulated_cycles} simulated cycles.")


# In[ ]:





# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Load the metadata.csv file (assuming it's already uploaded and available in the environment)
try:
    metadata_df = pd.read_csv(r"C:\Users\ASUS\Downloads\data(1)\cleaned_dataset\metadata.csv")
    print("Successfully loaded metadata.csv")
except FileNotFoundError:
    print("Error: metadata.csv not found. Please ensure the file is uploaded.")
    exit() # Exit if the file is not found
except Exception as e:
    print(f"An error occurred while loading metadata.csv: {e}")
    exit() # Exit for other loading errors

# --- Data Preprocessing ---

# Convert 'start_time' to a usable datetime format
def parse_time_array(time_str):
    try:
        if not time_str or time_str.strip() == '[]':
            return pd.NaT

        # Convert string representation to a list of floats (handling scientific notation and dots)
        parts = [float(p) for p in time_str.strip('[] ').replace('.', ' ').split() if p.strip()]

        if len(parts) >= 6:
            year, month, day, hour, minute, second_float = parts[:6]
            second = int(second_float)
            microsecond = int((second_float - second) * 1_000_000)
            return pd.Timestamp(year=int(year), month=int(month), day=int(day),
                                hour=int(hour), minute=int(minute), second=second,
                                microsecond=microsecond)
        else:
            return pd.NaT
    except (ValueError, TypeError, IndexError):
        return pd.NaT

# Apply the parsing function to the original DataFrame
metadata_df['start_datetime'] = metadata_df['start_time'].apply(parse_time_array)

# Force 'Capacity', 'Re', 'Rct', 'ambient_temperature' to numeric right at the beginning
# Any non-convertible values will become NaN
metadata_df['Capacity'] = pd.to_numeric(metadata_df['Capacity'], errors='coerce')
metadata_df['Re'] = pd.to_numeric(metadata_df['Re'], errors='coerce')
metadata_df['Rct'] = pd.to_numeric(metadata_df['Rct'], errors='coerce')
metadata_df['ambient_temperature'] = pd.to_numeric(metadata_df['ambient_temperature'], errors='coerce')


# Sort the original DataFrame by battery_id and start_datetime for proper forward-fill
metadata_df.sort_values(by=['battery_id', 'start_datetime'], inplace=True)

# Forward-fill 'Re' and 'Rct' within each battery group.
# This will propagate the last known impedance value to subsequent cycles until a new one is recorded.
metadata_df['Re_ffill'] = metadata_df.groupby('battery_id')['Re'].ffill()
metadata_df['Rct_ffill'] = metadata_df.groupby('battery_id')['Rct'].ffill()

# Now filter for 'discharge' cycles after ffill, ensuring impedance features are present
discharge_df = metadata_df[metadata_df['type'] == 'discharge'].copy()

# Drop rows where 'Capacity' is NaN (after coerce, these were truly non-numeric or missing)
discharge_df.dropna(subset=['Capacity'], inplace=True)

# Drop rows where 'start_datetime' parsing might have resulted in NaT for discharge cycles
discharge_df.dropna(subset=['start_datetime'], inplace=True)

# Drop rows where Re_ffill or Rct_ffill might still be NaN
# This happens if a battery's first cycle is discharge and there's no preceding impedance cycle.
discharge_df.dropna(subset=['Re_ffill', 'Rct_ffill'], inplace=True)

# Drop rows where ambient_temperature might still be NaN (after coerce)
discharge_df.dropna(subset=['ambient_temperature'], inplace=True)

# Recalculate cycle_number for each battery after filtering and dropping NaNs
# This ensures cycle numbers are sequential and correct for the filtered discharge data
discharge_df['cycle_number'] = discharge_df.groupby('battery_id').cumcount() + 1


# --- Feature Selection ---
features = ['cycle_number', 'ambient_temperature', 'Re_ffill', 'Rct_ffill']
target = 'Capacity'

X = discharge_df[features]
y = discharge_df[target]

# --- Model Training and Evaluation ---
# Splitting data by battery_id to ensure generalization, using 80% for training and 20% for testing.
unique_batteries = discharge_df['battery_id'].unique()

# Check if there are enough unique batteries for a meaningful split
if len(unique_batteries) < 2:
    print("Warning: Not enough unique batteries for a train/test split by battery_id. Splitting randomly by rows instead.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Store battery_id for plotting even in random split case
    plot_battery_ids = discharge_df.loc[y.index, 'battery_id']
else:
    train_batteries, test_batteries = train_test_split(unique_batteries, test_size=0.2, random_state=42)

    X_train = discharge_df[discharge_df['battery_id'].isin(train_batteries)][features]
    y_train = discharge_df[discharge_df['battery_id'].isin(train_batteries)][target]
    X_test = discharge_df[discharge_df['battery_id'].isin(test_batteries)][features]
    y_test = discharge_df[discharge_df['battery_id'].isin(test_batteries)][target]
    plot_battery_ids = discharge_df.loc[y_test.index, 'battery_id'] # Get battery_id for the test set

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# --- Visualization of Predictions ---
# Prepare data for plotting to include battery_id for hue
plot_df = pd.DataFrame({
    'cycle_number': X_test['cycle_number'],
    'Actual Capacity': y_test,
    'Predicted Capacity': y_pred,
    'battery_id': plot_battery_ids
})

plt.figure(figsize=(12, 6))
sns.lineplot(data=plot_df, x='cycle_number', y='Actual Capacity', hue='battery_id', marker='o')
sns.lineplot(data=plot_df, x='cycle_number', y='Predicted Capacity', hue='battery_id', marker='x', linestyle='--')

plt.title('Actual vs. Predicted Capacity for Test Batteries')
plt.xlabel('Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.legend(title='Battery ID')
plt.tight_layout()
plt.show()

# --- Feature Importance ---
feature_importances = model.feature_importances_
features_df_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
features_df_importance = features_df_importance.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(features_df_importance)


# In[ ]:




