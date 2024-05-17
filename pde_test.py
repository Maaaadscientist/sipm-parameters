import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator, interp1d
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
reference_table_path = config['reference_table']['path']
# Load the updated reference table
ref_table = pd.read_csv(reference_table_path)
# Connect to the SQLite database
conn = sqlite3.connect(database_path)
## Set up argument parser

# Create a mapping from pos to PDE@421nm and its error
pos_to_pde = dict(zip(ref_table['IPCB.Pos'], ref_table['PDE@421nm']))
pos_to_pde_err = dict(zip(ref_table['IPCB.Pos'], ref_table['PDE@421nm_err']))

# Query to get the available channels for the given tsn
tsn_selected = args.tsn
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

for ch in available_channels:
    ch_selected = ch
    
    # Query to select data for all voltage points for a specific SiPM channel, across all runs
    query = f"""
    SELECT vol, ov, ref_mu, ref_mu_err, mu, mu_err, pos
    FROM csv
    WHERE tsn = {tsn_selected} AND ch = {ch_selected}
    ORDER BY vol, ov;
    """
    
    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Drop data points where ov > 8V
    df = df[df['ov'] <= 8]
    
    # Look up the absolute PDE based on pos and calculate the absolute PDE for the tested SiPM
    df['PDE@421nm'] = df['pos'].map(pos_to_pde)
    df['PDE@421nm_err'] = df['pos'].map(pos_to_pde_err)
    df['pde_abs'] = df['mu'] / df['ref_mu'] * df['PDE@421nm']
    
    # Calculate the pde_abs_err
    df['pde_abs_err'] = np.sqrt(
        (df['mu'] / df['ref_mu'])**2 * (df['PDE@421nm_err'] / df['PDE@421nm'])**2 +
        (df['PDE@421nm'] / df['ref_mu'])**2 * (df['mu_err'] / df['mu'])**2 +
        (df['mu'] * df['PDE@421nm'] / df['ref_mu']**2)**2 * (df['ref_mu_err'] / df['ref_mu'])**2
    )
    
    # Group by 'vol' and calculate the mean of 'ov' and other parameters
    grouped = df.groupby('vol').agg({
        'ov': 'mean',
        'pde_abs': 'mean',
        'pde_abs_err': rms
    }).reset_index()
    
    # Add a point where ov = 0 and PDE = 0, with error 0
    grouped = pd.concat([pd.DataFrame({'vol': [grouped['vol'].min() - 1], 'ov': [0], 'pde_abs': [0], 'pde_abs_err': [0]}), grouped])
    
    # Check the maximum value of ov
    max_ov = round(grouped['ov'].max(), 1)
    if max_ov < 7.1:
        last_point = grouped.iloc[-1]
        new_point = pd.DataFrame({'vol': [last_point['vol']], 'ov': [7.1], 'pde_abs': [last_point['pde_abs']], 'pde_abs_err': [last_point['pde_abs_err']]})
        grouped = pd.concat([grouped, new_point])
        max_ov = 7.1  # Update max_ov to 7V

    # Separate the data for spline and linear interpolation
    grouped_spline = grouped[grouped['ov'] <= 6]
    grouped_linear = grouped[grouped['ov'] > 6]
    
    # Check the number of data points for spline interpolation
    if len(grouped_spline) > 1:
        # Perform monotonic cubic interpolation (PCHIP) for the absolute PDE where ov <= 6V
        pchip = PchipInterpolator(grouped_spline['ov'], grouped_spline['pde_abs'])
        ov_spline = np.linspace(grouped_spline['ov'].min(), grouped_spline['ov'].max(), 1000)
        pde_abs_spline = pchip(ov_spline)
    else:
        # Use linear interpolation if there are not enough points for a spline
        linear_interp_spline = interp1d(grouped_spline['ov'], grouped_spline['pde_abs'], kind='linear')
        ov_spline = np.linspace(grouped_spline['ov'].min(), grouped_spline['ov'].max(), 1000)
        pde_abs_spline = linear_interp_spline(ov_spline)
    
    # Perform linear interpolation between the last point of ov <= 6V and the last point of all points
    if len(grouped_linear) > 0:
        ov_linear = np.linspace(grouped_spline['ov'].max(), grouped['ov'].max(), 100)
        linear_interp = interp1d([grouped_spline['ov'].max(), grouped['ov'].max()], 
                                 [grouped_spline['pde_abs'].iloc[-1], grouped['pde_abs'].iloc[-1]], 
                                 kind='linear')
        pde_abs_linear = linear_interp(ov_linear)
        
        # Combine the results
        ov_combined = np.concatenate((ov_spline, ov_linear))
        pde_abs_combined = np.concatenate((pde_abs_spline, pde_abs_linear))
    else:
        ov_combined = ov_spline
        pde_abs_combined = pde_abs_spline
    # Perform spline interpolation for the errors where ov <= 6V
    if len(grouped_spline) > 1:
        pchip_err = PchipInterpolator(grouped_spline['ov'], grouped_spline['pde_abs_err'])
        pde_abs_err_spline = pchip_err(ov_spline)
    else:
        linear_interp_err_spline = interp1d(grouped_spline['ov'], grouped_spline['pde_abs_err'], kind='linear')
        pde_abs_err_spline = linear_interp_err_spline(ov_spline)
    
    # Combine the error results for the entire range
    if len(grouped_linear) > 0:
        linear_interp_err = interp1d([grouped_spline['ov'].max(), grouped['ov'].max()],
                                     [grouped_spline['pde_abs_err'].iloc[-1], grouped['pde_abs_err'].iloc[-1]], 
                                     kind='linear')
        ov_err_linear = np.linspace(grouped_spline['ov'].max(), grouped['ov'].max(), 100)
        pde_abs_err_linear = linear_interp_err(ov_err_linear)
        
        # Combine the spline and linear error parts
        ov_err_combined = np.concatenate((ov_spline, ov_err_linear))
        pde_abs_err_combined = np.concatenate((pde_abs_err_spline, pde_abs_err_linear))
    else:
        ov_err_combined = ov_spline
        pde_abs_err_combined = pde_abs_err_spline
    
    # Define the range for ov from 2.0 to 8.0 with step of 0.1V
    ov_range = np.arange(2.0, 7.1, 0.1)
    
    # Interpolate the absolute PDE and errors over the defined range
    pde_abs_interpolated = np.interp(ov_range, ov_combined, pde_abs_combined)
    pde_abs_err_interpolated = np.interp(ov_range, ov_err_combined, pde_abs_err_combined)
    
    # Calculate the interpolation error
    linear_interp = interp1d(grouped['ov'], grouped['pde_abs'], kind='linear')
    pde_abs_linear_interpolated = linear_interp(ov_range)
    interpolation_error = pde_abs_interpolated - pde_abs_linear_interpolated
    #print(interpolation_error)
    
    # Combine the errors using RMS
    combined_err = np.sqrt(pde_abs_err_interpolated**2 + interpolation_error**2)
    #combined_err = pde_abs_err_interpolated
    
    # Format the interpolated values and errors to 4 decimal places
    pde_abs_interpolated_formatted = [f'{value:.4f}' for value in pde_abs_interpolated]
    combined_err_formatted = [f'{value:.4f}' for value in combined_err]
    # Create the headers
    pde_header = ",".join([f'pde_{ov:.1f}' for ov in ov_range])
    pde_err_header = ",".join([f'pde_err_{ov:.1f}' for ov in ov_range])
    
    # Create the value strings
    pde_values = ",".join(pde_abs_interpolated_formatted)
    pde_err_values = ",".join(combined_err_formatted)
    
    # Combine headers and values with "tsn, ch, pos"
    combined_header = f"tsn,ch,pos,{pde_header},{pde_err_header}"
    combined_value = f"{int(tsn_selected)},{int(ch_selected)},{int(df['pos'].iloc[0])},{pde_values},{pde_err_values}"
    
    # Append headers and values to lists
    all_headers.append(combined_header)
    all_values.append(combined_value)
        
    # Plot the PDE with error bars
    #plt.errorbar(ov_range, pde_abs_interpolated, yerr=combined_err, label=f'Channel {ch_selected}', capsize=3)
    ## Customize the plot
    #plt.title(f'PDE vs Overvoltage for TSN {tsn_selected} CH{ch_selected}')
    #plt.xlabel('Overvoltage (V)')
    #plt.ylabel('PDE')
    #
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    print(f"pde: TSN-{int(tsn_selected)} Ch-{int(ch)} processed")
print("-----------------------------------------------------------------")
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
