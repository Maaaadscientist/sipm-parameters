import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from scipy.stats import linregress
import argparse

# Define constants
ELEMENTARY_CHARGE = 1.602e-19  # Coulombs

# Set up argument parser
parser = argparse.ArgumentParser(description='Process SiPM data.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--tsn', type=int, required=True, help='TSN selected')
parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

database_path = config['database']['path']
gain_factors_path = config['gain_factors']['path']
tsn_selected = args.tsn
output_path = args.output

# Load gain factors
gain_factors_df = pd.read_csv(gain_factors_path)
gain_factors = gain_factors_df.set_index(['pcb_pos', 'ch'])['amp_gain'].to_dict()

# Connect to the SQLite database
conn = sqlite3.connect(database_path)

# Query to get the available channels for the given tsn
query_channels = f"""
SELECT DISTINCT ch
FROM csv
WHERE tsn = {tsn_selected};
"""

# Execute the query and load the results into a DataFrame
channels_df = pd.read_sql_query(query_channels, conn)

# Extract the available channels as a list
available_channels = channels_df['ch'].tolist()

# Define the RMS function
def rms(series):
    return np.sqrt(np.mean(series**2))

# Initialize lists to store the headers and values for all channels
all_headers = []
all_values = []

# Initialize plot
plt.figure(figsize=(12, 6))

for ch in available_channels:
    ch_selected = ch
    
    # Query to select data for all voltage points for a specific SiPM channel, across all runs
    query = f"""
    SELECT vol, ov, gain, gain_err, pos
    FROM csv
    WHERE tsn = {tsn_selected} AND ch = {ch_selected}
    ORDER BY vol, ov;
    """
    
    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Drop data points where ov > 8V
    df = df[df['ov'] <= 8]
    
    # Calculate the absolute gain
    df['amp_gain'] = df.apply(lambda row: gain_factors.get((row['pos'], ch_selected)), axis=1)
    df['gain_abs'] = df['gain'] * 1e-12 / (ELEMENTARY_CHARGE * df['amp_gain'])
    df['gain_abs_err'] = df['gain_err'] * 1e-12 / (ELEMENTARY_CHARGE * df['amp_gain'])
    
    # Group by 'vol' and calculate the mean of 'ov' and other parameters
    grouped = df.groupby('vol').agg({
        'ov': 'mean',
        'gain_abs': 'mean',
        'gain_abs_err': rms
    }).reset_index()
    
    # Perform a linear fit
    slope, intercept, r_value, p_value, std_err = linregress(grouped['ov'], grouped['gain_abs'])
    
    # Define the range for ov from 2.0 to 7.1 with step of 0.1V
    ov_range = np.arange(2.0, 7.1, 0.1)
    
    # Calculate the gain using the linear fit
    gain_abs_interpolated = slope * ov_range + intercept
    
    # Calculate the interpolation errors
    gain_abs_err_interpolated = np.interp(ov_range, grouped['ov'], grouped['gain_abs_err'])
    
    # Print the interpolated values and errors
    gain_header = ",".join([f'gain_abs_{ov:.1f}' for ov in ov_range])
    gain_err_header = ",".join([f'gain_abs_err_{ov:.1f}' for ov in ov_range])
    gain_values = ",".join([f'{value:.4f}' for value in gain_abs_interpolated])
    gain_err_values = ",".join([f'{value:.4f}' for value in gain_abs_err_interpolated])
    
    # Combine headers and values with "tsn, ch, pos"
    combined_header = f"tsn,ch,pos,{gain_header},{gain_err_header}"
    combined_value = f"{int(tsn_selected)},{int(ch_selected)},{int(df['pos'].iloc[0])},{gain_values},{gain_err_values}"
    
    # Append headers and values to lists
    all_headers.append(combined_header)
    all_values.append(combined_value)
    
    # Plot the gain with error bars
    #plt.errorbar(ov_range, gain_abs_interpolated, yerr=gain_abs_err_interpolated, label=f'Channel {ch_selected}', capsize=3)
    print(f"gain: TSN-{int(tsn_selected)} Ch-{int(ch)} processed")
print("-----------------------------------------------------------------")

## Customize the plot
#plt.title(f'Gain vs Overvoltage for TSN {tsn_selected}')
#plt.xlabel('Overvoltage (V)')
#plt.ylabel('Gain (Absolute)')
#plt.legend()
#plt.grid(True)
#plt.show()

# Save headers and values to CSV
output_path = args.output
with open(output_path, 'w') as f:
    write_header = True
    for header, value in zip(all_headers, all_values):
        if write_header:
            f.write(header + '\n')
            write_header = False
        f.write(value + '\n')

# Close the connection to the database
conn.close()

