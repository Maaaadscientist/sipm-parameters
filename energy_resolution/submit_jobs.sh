#!/bin/bash

# Set the default directory to "jobs"
directory="jobs"

# Check if the directory path argument is provided
if [[ $# -eq 1 ]]; then
  # Retrieve the directory path argument
  directory="$1"
elif [[ $# -gt 1 ]]; then
  echo "Error: Too many arguments provided."
  echo "Usage: $0 [directory_path]"
  exit 1
fi

# Directory to be added to PATH
htcondor_dir="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin"

# Check if the directory is already in the PATH
if echo "$PATH" | grep -q "$htcondor_dir"; then
  echo "Directory already exists in PATH. No changes needed."
else
  # Add the directory to the PATH variable
  export PATH="$htcondor_dir:$PATH"
  echo "Directory added to PATH."
fi

# Ask user if they want to keep logs
echo "Do you want to keep logs? (yes/no)"
read keep_logs

# Check and create the "logs" directory if needed
if [[ $keep_logs == "yes" ]]; then
  if [[ ! -d "logs" ]]; then
    mkdir "logs"
    echo "\"logs\" directory created."
  fi
fi

# Iterate over the files in the directory
for file in "$directory"/*.sh; do
  # Check if the file is a regular file
  if [[ -f "$file" ]]; then
    # Submit the file using hep_sub command
    if [[ $keep_logs == "yes" ]]; then
      # Extract filename without path
      filename=$(basename -- "$file")
      hep_sub "$file" -e "logs/${filename}.err" -o "logs/${filename}.out"
    else
      hep_sub "$file" -e /dev/null -o /dev/null
    fi
  fi
done

