import os
import sys

level = 1
job_split = 50
N_events = 1000
root_dir = "/junofs/users/wanghanwen/e_resolution"
output_dir = root_dir +"/outputs"
job_dir = root_dir + "/jobs"
script  = ''
script += '#!/bin/bash\n'
script += f'directory="{job_dir}"\n'
script += f'output_path="{output_dir}"\n'
script += 'if [ ! -d "$directory" ]; then\n'
script += '  echo "Directory does not exist. Creating directory..."\n'
script += '  mkdir -p "$directory"\n'
script += '  echo "Directory created."\n'
script += 'else\n'
script += '  echo "Directory already exists."\n'
script += 'fi\n'
script += 'if [ ! -d "$output_path" ]; then\n'
script += '  echo "Directory does not exist. Creating directory..."\n'
script += '  mkdir -p "$output_path"\n'
script += '  echo "Directory created."\n'
script += 'else\n'
script += '  echo "Directory already exists."\n'
script += 'fi\n'
script += 'source /workfs2/juno/wanghanwen/sipm-massive/env_lcg.sh\n'
script += 'PYTHON=$(which python3)\n'
script += 'cd $directory\n'

if not os.path.isdir(job_dir):
    os.makedirs(job_dir)

ov_list = [round(2.0 + 0.1 * i, 1) for i in range(51)]
ov_list.append(-1)
ov_list.append(-2)
for ov in ov_list:
    if ov == -2:
        ov_str = "max"
    elif ov == -1:
        ov_str = "typical"
    else:
        ov_str = f"{ov:0.1f}"
    for njob in range(job_split):
        job_content = f"$PYTHON /workfs2/juno/wanghanwen/sipm-massive/test/e_res.py --N {N_events} --energy 1 --ov {ov:.1f} --input /junofs/users/wanghanwen/bychannel.csv --output {output_dir}/hist_{ov_str}_{njob}.root --seed {njob} --level {level}"
        with open(f"{job_dir}/hist_{ov_str}_{njob}.sh", "w") as file1:
            file1.write(script)
            file1.write(job_content)
os.system(f"chmod a+x {job_dir}/*.sh")
os.system(f"cp energy_resolution/* {root_dir}")
