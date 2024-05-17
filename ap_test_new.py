import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import argparse
import yaml

# Define the RMS function
def rms(series):
    return np.sqrt(np.mean(series**2))

# Set up argument parser
parser = argparse.ArgumentParser(description='Process AP data.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--tsn', type=int, required=True, help='TSN selected')
parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

database_path = config['database']['path']
tsn_selected = args.tsn
output_path = args.output

# Connect to the SQLite database
conn = sqlite3.connect(database_path)

query_channels = f"""
SELECT ch
FROM csv
WHERE tsn = {tsn_selected}
GROUP BY ch
HAVING COUNT(DISTINCT vol) >= 5;
"""
# Execute the query and load the results into a DataFrame
channels_df = pd.read_sql_query(query_channels, conn)

# Extract the available channels as a list
available_channels = channels_df['ch'].tolist()

# Initialize lists to store the headers and values for all channels
all_headers = []
all_err_headers = []
all_values = []
all_err_values = []

# Loop over available channels
for ch in available_channels:
    ch_selected = ch
    # Query to select data for all voltage points for a specific SiPM channel, across all runs
    query = f"""
    SELECT vol, ov, alpha, alpha_err, ap_pe, ap_pe_err, pos
    FROM csv
    WHERE tsn = {tsn_selected} AND ch = {ch_selected}
    ORDER BY vol, ov;
    """

    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)

    # Set alpha and alpha_err to 0 for ov < 4
    df.loc[df['ov'] < 4, ['alpha', 'alpha_err']] = 0

    # Calculate ap_mean, ap_var, and ap_prob
    df['ap_mean'] = df['ap_pe'] * df['alpha'] / (1 - df['alpha'])
    df['ap_var'] = df['ap_pe'] * df['alpha'] / (1 - df['alpha'])**2
    df['ap_prob'] = df['alpha'] / (1 - df['alpha'])

    # Propagate errors for ap_mean, ap_var, and ap_prob
    df['ap_mean_err'] = np.sqrt(
        (df['ap_pe_err'] * df['alpha'] / (1 - df['alpha']))**2 +
        (df['ap_pe'] * df['alpha_err'] / (1 - df['alpha']))**2 +
        (df['ap_pe'] * df['alpha'] * df['alpha_err'] / (1 - df['alpha'])**2)**2
    )

    df['ap_var_err'] = np.sqrt(
        (df['ap_pe_err'] * df['alpha'] / (1 - df['alpha'])**2)**2 +
        (df['ap_pe'] * df['alpha_err'] / (1 - df['alpha'])**2)**2 +
        (2 * df['ap_pe'] * df['alpha'] * df['alpha_err'] / (1 - df['alpha'])**3)**2
    )

    df['ap_prob_err'] = np.sqrt(
        (df['alpha_err'] / (1 - df['alpha']))**2 +
        (df['alpha'] * df['alpha_err'] / (1 - df['alpha'])**2)**2
    )

    # Group by 'vol' and calculate the mean of 'ov' and other parameters
    grouped = df.groupby('vol').agg({
        'ov': 'mean',
        'ap_mean': 'mean',
        'ap_var': 'mean',
        'ap_prob': 'mean',
        'ap_mean_err': rms,
        'ap_var_err': rms,
        'ap_prob_err': rms
    }).reset_index()

    # Define the range for ov from 2.0 to 7.1 with step of 0.1V
    ov_range = np.arange(2.0, 7.1, 0.1)

    # Combine headers and values with "tsn, ch, pos"
    combined_header = f"tsn,ch,pos"
    combined_value = f"{int(tsn_selected)},{int(ch_selected)},{int(df['pos'].iloc[0])}"
    for key, err_key in [('ap_mean', 'ap_mean_err'), ('ap_var', 'ap_var_err'), ('ap_prob', 'ap_prob_err')]:
        if key == 'ap_prob':
            # Perform monotonic cubic interpolation (PCHIP) for ap_prob to ensure smoothness and monotonicity
            interp_func = PchipInterpolator(grouped['ov'], grouped[key])
            interp_err_func = PchipInterpolator(grouped['ov'], grouped[err_key])
        else:
            # Perform linear interpolation for the other parameters
            interp_func = interp1d(grouped['ov'], grouped[key], kind='linear', fill_value='extrapolate')
            interp_err_func = interp1d(grouped['ov'], grouped[err_key], kind='linear', fill_value='extrapolate')

        interpolated_values = interp_func(ov_range)
        interpolated_errors = interp_err_func(ov_range)

        # Set values to zero for the range below the first ov point
        interpolated_values[ov_range < grouped['ov'].min()] = 0
        interpolated_errors[ov_range < grouped['ov'].min()] = 0

        # Set values to the last point for the range beyond the last ov point
        last_value = grouped[key].iloc[-1]
        last_error = grouped[err_key].iloc[-1]
        interpolated_values[ov_range > grouped['ov'].max()] = last_value
        interpolated_errors[ov_range > grouped['ov'].max()] = last_error

        # Append the headers and values to the lists
        header = ",".join([f'{key}_{ov:.1f}' for ov in ov_range])
        err_header = ",".join([f'{err_key}_{ov:.1f}' for ov in ov_range])
        values = ",".join([f'{value:.4f}' for value in interpolated_values])
        errors = ",".join([f'{value:.4f}' for value in interpolated_errors])
        combined_header += ',' + header
        combined_header += ',' + err_header
        combined_value += ',' + values
        combined_value += ',' + errors
    all_headers.append(combined_header)
    all_values.append(combined_value)
    print(f"afterpulse: TSN-{int(tsn_selected)} Ch-{int(ch_selected)} processed")

print("-----------------------------------------------------------------")

# Save headers and values to CSV
output_path = args.output
with open(output_path, 'w') as f:
    write_header = True
    for header, value in zip(all_headers, all_values):
        if write_header:
            f.write(header + '\n')
            write_header = False
        f.write(value + '\n')

