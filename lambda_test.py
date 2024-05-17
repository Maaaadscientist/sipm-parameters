import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
import argparse
import yaml

# Define the RMS function
def rms(series):
    return np.sqrt(np.mean(series**2))

# Set up argument parser
parser = argparse.ArgumentParser(description='Process crosstalk data.')
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

# Query to get the available channels for the given tsn

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
for ch_selected in available_channels:
    # Query to select data for all voltage points for a specific SiPM channel, across all runs
    query = f"""
    SELECT vol, ov, lambda, lambda_err, pos
    FROM csv
    WHERE tsn = {tsn_selected} AND ch = {ch_selected}
    ORDER BY vol, ov;
    """

    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)

    # Drop data points where ov > 8V
    df = df[df['ov'] <= 8]

    # Group by 'vol' and calculate the mean of 'ov' and other parameters
    grouped = df.groupby('vol').agg({
        'ov': 'mean',
        'lambda': 'mean',
        'lambda_err': rms
    }).reset_index()

    # Define the range for ov from 2.0 to 7.1 with step of 0.1V
    ov_range = np.arange(2.0, 7.1, 0.1)

    # Perform a linear fit for extrapolation
    slope, intercept, r_value, p_value, std_err = linregress(grouped['ov'], grouped['lambda'])

    # Perform linear interpolation for lambda within the original data range
    lambda_interp = interp1d(grouped['ov'], grouped['lambda'], kind='linear', fill_value='extrapolate')
    lambda_interpolated = np.where((ov_range >= grouped['ov'].min()) & (ov_range <= grouped['ov'].max()), lambda_interp(ov_range), slope * ov_range + intercept)

    # Perform linear interpolation for lambda_err within the original data range
    lambda_err_interp = interp1d(grouped['ov'], grouped['lambda_err'], kind='linear', fill_value='extrapolate')
    lambda_err_interpolated = np.where((ov_range >= grouped['ov'].min()) & (ov_range <= grouped['ov'].max()), lambda_err_interp(ov_range), lambda_err_interp(grouped['ov'].max()))

    # Create headers and values strings for CSV output
    lambda_header = ",".join([f'lambda_{ov:.1f}' for ov in ov_range])
    lambda_err_header = ",".join([f'lambda_err_{ov:.1f}' for ov in ov_range])
    lambda_values = ",".join([f'{value:.4f}' for value in lambda_interpolated])
    lambda_err_values = ",".join([f'{value:.4f}' for value in lambda_err_interpolated])

    # Add TSN, channel, and position information
    combined_header = f"tsn,ch,pos,{lambda_header},{lambda_err_header}"
    combined_values = f"{int(tsn_selected)},{int(ch_selected)},{int(df['pos'].iloc[0])},{lambda_values},{lambda_err_values}"

    # Append headers and values to lists
    all_headers.append(combined_header)
    all_values.append(combined_values)

    print(f"crosstalk: TSN-{int(tsn_selected)} Ch-{int(ch_selected)} processed")
    # Plot the lambda with error bars
    #plt.figure(figsize=(12, 6))
    #plt.errorbar(ov_range, lambda_interpolated, yerr=lambda_err_interpolated, label=f'Channel {ch_selected}', capsize=3)
    #plt.title(f'Crosstalk (lambda) vs Overvoltage for TSN {tsn_selected}')
    #plt.xlabel('Overvoltage (V)')
    #plt.ylabel('Crosstalk (lambda)')
    #plt.legend()
    #plt.grid(True)
    #plt.show()
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

# Close the connection to the database
conn.close()
