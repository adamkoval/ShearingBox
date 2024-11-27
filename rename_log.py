#!/home/akoval/miniconda3/envs/python3.11_conda_env/bin/python
import os
import sys
from datetime import datetime

file = sys.argv[1]

os.system(f"mv {file} output/run_logs/{file}_{datetime.isoformat(datetime.now())[:-7]}")

sys.exit()