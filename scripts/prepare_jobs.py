import sqlite3
import pandas as pd
import argparse
import yaml
import os
import subprocess

# Set up argument parser
parser = argparse.ArgumentParser(description='Prepare job submission scripts for cluster.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--job_dir', type=str, required=True, help='Directory to save job scripts')
args = parser.parse_args()

# Get absolute paths
config_path = os.path.abspath(args.config)
job_dir = os.path.abspath(args.job_dir)
wrapper_name = 'wrapper_script.sh'#os.path.abspath(args.wrapper)

# Load configuration from YAML file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

database_path = config['database']['path']

# Create job directory if it does not exist
os.makedirs(job_dir + '/scripts', exist_ok=True)
os.makedirs(job_dir + '/outputs', exist_ok=True)
os.makedirs(job_dir + '/tmps', exist_ok=True)

# Connect to the SQLite database
conn = sqlite3.connect(database_path)

# Query to get the unique list of tsn values
query_tsn = "SELECT DISTINCT tsn FROM csv;"
tsn_df = pd.read_sql_query(query_tsn, conn)

# Close the connection to the database
conn.close()

# Get the list of tsn values
tsn_list = tsn_df['tsn'].tolist()

# Define the list of Python scripts to be run for each tsn
python_scripts = ['pde_test.py', 'dcr_test.py', 'lambda_test.py', 'ap_test_new.py', 'gain_test.py', 'vbd.py']
output_prefixes = ['pde', 'dcr', 'lambda', 'ap', 'gain', 'vbd']

# Get absolute paths of the Python scripts
python_scripts_paths = [os.path.abspath(script) for script in python_scripts]
combine_script_path = os.path.abspath('combine_all_csv.py')

python_path = 'python3'
# Create a bash script for each tsn
for tsn in tsn_list:
    tsn = int(tsn)
    job_script_path = os.path.join(job_dir,'scripts', f'job_{tsn}.sh')
    with open(job_script_path, 'w') as job_script:
        job_script.write("#!/bin/bash\n\n")
        csv_args = ''
        job_script.write(f"cd {job_dir}/tmps\n")
        job_script.write("sleep 3\n")
        for script_path, prefix in zip(python_scripts_paths, output_prefixes):
            output_file = f"{prefix}_{tsn}.csv"
            job_script.write(f"{python_path} {script_path} --config {config_path} --tsn {tsn} --output {output_file}\n")
            csv_args += f'{output_file} '
            job_script.write('sleep 1\n')
        job_script.write(f"{python_path} {combine_script_path} {csv_args} {job_dir}/outputs/output_{tsn}.csv -k tsn,ch\n")
        job_script.write('sleep 1\n')
        job_script.write(f'rm *_{tsn}.csv')
    
    # Make the bash script executable
    os.chmod(job_script_path, 0o755)

print(f"Job scripts created in directory: {job_dir}")

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
    
    os.chmod(wrapper_script_path, 0o755)
    
    # Count the number of job scripts
    num_job_scripts = len([name for name in os.listdir(job_dir+'/scripts') if name.startswith('job_') and name.endswith('.sh')])
    
    # Submit the jobs using hep_sub
    submit_command = f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
    subprocess.run(submit_command, shell=True, check=True)
    
    print(f"Submitted {num_job_scripts} jobs using hep_sub.")
