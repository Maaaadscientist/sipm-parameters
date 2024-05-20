import os
import shutil

job_directory = 'jobs'  # Replace with your job directory path
csv_directory = 'outputs'  # Replace with your csv directory path
resubmit_directory = 'jobs_resubmit'  # Replace with your resubmit directory path

if not os.path.exists(resubmit_directory):
    os.makedirs(resubmit_directory)

for filename in os.listdir(job_directory):
    if filename.endswith('.sh'):
        root_name = filename.rsplit('.', 1)[0]#.strip(".sh")  # Remove the .sh extension
        csv_filename = root_name + '.root'
        csv_path = os.path.join(csv_directory, csv_filename)
        
        if os.path.exists(csv_path):
            print(f"job succeeded: {csv_path}")
            continue
        else:
            # .csv does not exist, copy .sh file to resubmit directory
            print(f"job failed: {csv_path}")
            shutil.copy(os.path.join(job_directory, filename), resubmit_directory)

print("Job scripts requiring resubmission have been copied to", resubmit_directory)

