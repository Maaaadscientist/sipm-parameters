import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.stats import linregress
import argparse

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
# Connect to the SQLite database
conn = sqlite3.connect(database_path)
# Query to get the available channels for the given tsn
tsn_selected = args.tsn
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
    SELECT vol, ov, dcr, dcr_err, pos
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
        'dcr': 'mean',
        'dcr_err': rms
    }).reset_index()
    
    # Check the maximum value of ov
    max_ov = round(grouped['ov'].max(), 1)
    min_ov = round(grouped['ov'].min(), 1)
    
    # Define the range for ov from 2.0 to 7.1 with step of 0.1V
    ov_range = np.arange(2.0, 7.1, 0.1)
    
    # Perform a linear fit for extrapolation
    slope, intercept, r_value, p_value, std_err = linregress(grouped['ov'], grouped['dcr'])
    
    # Perform linear interpolation for dcr within the original data range
    dcr_interp = interp1d(grouped['ov'], grouped['dcr'], kind='linear', fill_value='extrapolate')
    dcr_interpolated = np.where((ov_range >= min_ov) & (ov_range <= max_ov), dcr_interp(ov_range), slope * ov_range + intercept)
    
    # Perform linear interpolation for dcr_err within the original data range
    dcr_err_interp = interp1d(grouped['ov'], grouped['dcr_err'], kind='linear', fill_value='extrapolate')
    dcr_err_interpolated = np.where((ov_range >= min_ov) & (ov_range <= max_ov), dcr_err_interp(ov_range), dcr_err_interp(max_ov))
    
    # Print the interpolated values and errors
    dcr_header = ",".join([f'dcr_{ov:.1f}' for ov in ov_range])
    dcr_err_header = ",".join([f'dcr_err_{ov:.1f}' for ov in ov_range])
    dcr_values = ",".join([f'{value:.4f}' for value in dcr_interpolated])
    dcr_err_values = ",".join([f'{value:.4f}' for value in dcr_err_interpolated])
    
    # Combine headers and values with "tsn, ch, pos"
    combined_header = f"tsn,ch,pos,{dcr_header},{dcr_err_header}"
    combined_value = f"{int(tsn_selected)},{int(ch_selected)},{int(df['pos'].iloc[0])},{dcr_values},{dcr_err_values}"
    
    # Append headers and values to lists
    all_headers.append(combined_header)
    all_values.append(combined_value)
    
    # Plot the DCR with error bars
    #plt.errorbar(ov_range, dcr_interpolated, yerr=dcr_err_interpolated, label=f'Channel {ch_selected}', capsize=3)

    print(f"dcr: TSN-{int(tsn_selected)} Ch-{int(ch)} processed")
print("-----------------------------------------------------------------")
## Customize the plot
#plt.title(f'DCR vs Overvoltage for TSN {tsn_selected}')
#plt.xlabel('Overvoltage (V)')
#plt.ylabel('DCR')
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
