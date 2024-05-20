import os
import shutil
import ROOT

def process_root_files(directory, file_pattern, hist_name, jobs_dir, resubmit_dir):
    failed_files = []

    # Ensure resubmit directory exists
    if not os.path.exists(resubmit_dir):
        os.makedirs(resubmit_dir)

    # List all files in the directory matching the pattern
    root_files = [f for f in os.listdir(directory) if f.endswith('.root') and file_pattern in f]

    # Loop over the files
    for filename in root_files:
        # Construct the full path to the file
        full_path = os.path.join(directory, filename)

        # Open the ROOT file
        root_file = ROOT.TFile(full_path, "READ")

        if root_file.IsOpen():
            print(f"Processing file: {filename}")

            # Get the histogram
            histogram = root_file.Get(hist_name)

            if histogram:
                # Calculate the integral
                integral = histogram.Integral()
                print(f"Integral of {hist_name} in {filename}: {integral}")
            else:
                print(f"Histogram {hist_name} not found in {filename}")
                failed_files.append(filename)

            # Close the ROOT file
            root_file.Close()
        else:
            print(f"Failed to open {filename}")
            failed_files.append(filename)
    # Process failed files
    for filename in failed_files:
        script_name = filename.replace('.root', '.sh')
        script_path = os.path.join(jobs_dir, script_name)
        resubmit_path = os.path.join(resubmit_dir, script_name)

        # Check if the script file exists
        if os.path.exists(script_path):
            # Copy the script to the resubmit directory
            shutil.copy(script_path, resubmit_path)
            print(f"Copied {script_name} to resubmit directory.")
        else:
            print(f"Script {script_name} not found.")

if __name__ == "__main__":
    directory = "outputs"  # Update this with your directory path
    file_pattern = "hist_"
    hist_name = "dcr_poisson"
    jobs_dir = "jobs"
    resubmit_dir = "jobs_resubmit"
    process_root_files(directory, file_pattern, hist_name, jobs_dir, resubmit_dir)


