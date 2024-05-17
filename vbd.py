import sqlite3
import pandas as pd
import numpy as np
import argparse
import yaml

# Set up argument parser
parser = argparse.ArgumentParser(description='Calculate breakdown voltage.')
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
all_headers = ["tsn", "ch", "pos", "vbd", "vbd_err", "vbd_diff"]
all_values = []

# Loop over available channels
for ch_selected in available_channels:
    # Query to select data for breakdown voltage for a specific SiPM channel
    query = f"""
    SELECT vbd, vbd_err, vbd_diff, pos
    FROM csv
    WHERE tsn = {tsn_selected} AND ch = {ch_selected}
    LIMIT 1;
    """

    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)

    # Append the results to the list
    if not df.empty:
        vbd = df['vbd'].iloc[0]
        vbd_err = df['vbd_err'].iloc[0]
        vbd_diff = df['vbd_diff'].iloc[0]
        pos = df['pos'].iloc[0]
        values = f"{int(tsn_selected)},{int(ch_selected)},{int(pos)},{vbd:.4f},{vbd_err:.5f},{vbd_diff:.4f}"
        all_values.append(values)

    print(f"vbd: TSN-{int(tsn_selected)} Ch-{int(ch_selected)} processed")
print("---------------------------Finished------------------------------")
# Write to the output CSV file
with open(output_path, 'w') as f:
    f.write(",".join(all_headers) + '\n')
    for values in all_values:
        f.write(values + '\n')

# Close the connection to the database
conn.close()

