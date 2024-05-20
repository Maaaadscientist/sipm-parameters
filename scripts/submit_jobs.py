import sqlite3
import pandas as pd
import argparse
import yaml
import os
import subprocess

# Set up argument parser
parser = argparse.ArgumentParser(description='Prepare job submission scripts for cluster.')
parser.add_argument('--job_dir', type=str, required=True, help='Directory to save job scripts')
args = parser.parse_args()

job_dir = os.path.abspath(args.job_dir)
wrapper_name = 'wrapper_script.sh'#os.path.abspath(args.wrapper)
wrapper_script_path = f'{job_dir}/{wrapper_name}'

submit = input('Submit jobs? Y/N\n')
if submit.lower() == 'y':
    # Write the wrapper script with the actual job_dir path
    wrapper_script_content = """#!/bin/bash
# Directory containing the job scripts
jobs_dir={job_dir}/scripts

# Argument handling for the specific job script to execute
if [ $# -ne 1 ]; then
    echo "Usage: $0 [ProcId]"
    exit 1
fi

procid=$1  # The provided ProcId determines which job script to run

# Get an array of job scripts
scripts=( $(ls $jobs_dir/*.sh) )

# Calculate total available jobs
total_jobs=${{#scripts[@]}}

# Determine the job script to execute based on ProcId
if [ $procid -ge 0 ] && [ $procid -lt $total_jobs ]; then
    script_to_run="${{scripts[$procid]}}"
    echo "Running job script: $script_to_run"
    bash "$script_to_run"
else
    echo "Error: ProcId $procid is out of range. Only $total_jobs jobs are available."
    exit 1
fi
""".format(job_dir=job_dir)
    
    with open(wrapper_script_path, 'w') as wrapper_script:
        wrapper_script.write(wrapper_script_content)
    
    os.chmod(wrapper_name, 0o755)
    
    # Count the number of job scripts
    num_job_scripts = len([name for name in os.listdir(job_dir+'/scripts') if name.startswith('job_') and name.endswith('.sh')])
    
    # Submit the jobs using hep_sub
    submit_command = f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
    subprocess.run(submit_command, shell=True, check=True)
    
    print(f"Submitted {num_job_scripts} jobs using hep_sub.")
