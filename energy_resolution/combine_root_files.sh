#!/bin/bash

# Associative array to hold filenames grouped by the overvoltage point
declare -A file_groups

# Loop over all .root files
for file in outputs/*.root; do
    # Extract the overvoltage point (assuming the format is always like 'hist_X.Y_Z.root')
    ov_point=$(echo "$file" | awk -F '_' '{print $2}')
    echo $file

    # Append the filename to the appropriate group in the associative array
    file_groups[$ov_point]+=" $file"
done

# Loop over the file groups and perform merging
for ov_point in "${!file_groups[@]}"; do
    files=${file_groups[$ov_point]}
    output_file="merged_$ov_point.root"  # Naming the output file

    # Your merge command goes here
    hadd -f merged/$output_file $files

    echo "Merged files for overvoltage point $ov_point into $output_file"
done

