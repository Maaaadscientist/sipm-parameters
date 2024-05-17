import os
import argparse
import subprocess

def create_wrapper_script(job_dir, wrapper_name):
    wrapper_script_path = f'{job_dir}/{wrapper_name}'
    wrapper_script_content = f"""#!/bin/bash
# Directory containing the job scripts
jobs_dir={job_dir}/resubmit_jobs

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
"""
    with open(wrapper_script_path, 'w') as wrapper_script:
        wrapper_script.write(wrapper_script_content)
    
    os.chmod(wrapper_script_path, 0o755)
    return wrapper_script_path

def get_tsn_from_filename(filename, prefix, suffix):
    return filename.replace(prefix, "").replace(suffix, "")

def resubmit_failed_jobs(job_dir):
    scripts_dir = os.path.join(job_dir, "scripts")
    outputs_dir = os.path.join(job_dir, "outputs")
    
    script_prefix = "job_"
    script_suffix = ".sh"
    output_prefix = "output_"
    output_suffix = ".csv"

    # List all job scripts
    job_scripts = [f for f in os.listdir(scripts_dir) if f.startswith(script_prefix) and f.endswith(script_suffix)]
    job_tsns = [get_tsn_from_filename(script, script_prefix, script_suffix) for script in job_scripts]

    # List all output files
    output_files = [f for f in os.listdir(outputs_dir) if f.startswith(output_prefix) and f.endswith(output_suffix)]
    output_tsns = [get_tsn_from_filename(output, output_prefix, output_suffix) for output in output_files]

    # Determine failed jobs
    failed_tsns = set(job_tsns) - set(output_tsns)
    failed_jobs = [os.path.join(scripts_dir, f"{script_prefix}{tsn}{script_suffix}") for tsn in failed_tsns]

    # Print and resubmit failed jobs
    os.makedirs(job_dir + '/resubmit_jobs', exist_ok=True)
    for job_script in failed_jobs:
        os.system(f'cp {job_script} {job_dir}/resubmit_jobs')
    print(f"Found {len(failed_jobs)} failed jobs.")
    submit = input('Resubmit jobs? Y/N\n')
    if submit.lower() == 'y':
        wrapper_path = create_wrapper_script(job_dir, 'resubmit_wrapper.sh')
        # Count the number of job scripts
        num_job_scripts = len([name for name in os.listdir(job_dir+'/resubmit_jobs') if name.startswith('job_') and name.endswith('.sh')])
        
        # Submit the jobs using hep_sub
        submit_command = f"hep_sub -e /dev/null -o /dev/null {wrapper_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
        subprocess.run(submit_command, shell=True, check=True)
        
        print(f"Submitted {num_job_scripts} jobs using hep_sub.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resubmit failed jobs by checking output files.')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory containing the job scripts and outputs.')
    
    args = parser.parse_args()
    resubmit_failed_jobs(args.job_dir)

