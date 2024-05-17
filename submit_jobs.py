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
wrapper_script_path = 'wrapper_script.sh'#os.path.abspath(args.wrapper)

submit = input('Submit jobs? Y/N\n')
if submit.lower() == 'y':
    # Count the number of job scripts
    num_job_scripts = len([name for name in os.listdir(job_dir+'/scripts') if name.startswith('job_') and name.endswith('.sh')])
    
    # Submit the jobs using hep_sub
    submit_command = f"hep_sub -e /dev/null -o /dev/null {job_dir}/{wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
    subprocess.run(submit_command, shell=True, check=True)
    
    print(f"Submitted {num_job_scripts} jobs using hep_sub.")
