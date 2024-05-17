#!/bin/bash
# Directory containing the job scripts
jobs_dir=

# Argument handling for the specific job script to execute
if [ $# -ne 1 ]; then
    echo "Usage: $0 [ProcId]"
    exit 1
fi

procid=$1  # The provided ProcId determines which job script to run

# Get an array of job scripts
scripts=( $(ls $jobs_dir/*.sh) )

# Calculate total available jobs
total_jobs=${#scripts[@]}

# Determine the job script to execute based on ProcId
if [ $procid -ge 0 ] && [ $procid -lt $total_jobs ]; then
    script_to_run="${scripts[$procid]}"
    echo "Running job script: $script_to_run"
    bash "$script_to_run"
else
    echo "Error: ProcId $procid is out of range. Only $total_jobs jobs are available."
    exit 1
fi  
