#!/bin/bash
### General options
### -- set the job Name --
#BSUB -J SOMETHING
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 5GB
### -- set the walltime limit: hh:mm --
#BSUB -W 72:00
### -- set the email address --
#BSUB -u elygao@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -oo /zhome/06/8/135695/log%J.out
#BSUB -eo /zhome/06/8/135695/log%J.err

# Load necessary modules
module load mpi4py/3.1.3-python-3.10.2-openmpi-4.1.2
module load pandas/1.4.1-python-3.10.2
module load scipy/1.7.3-python-3.10.2
module load matplotlib/3.5.1-numpy-1.22.2-python-3.10.2

# Try to load scikit-learn module if available
module load scikit-learn/0.24.1-python-3.10.2 || echo "Module scikit-learn not found, using pip instead."

# Check if scikit-learn is installed and install if not
python3 -c "import sklearn" || pip install --user scikit-learn

# Assuming test_script.py is in the same directory as where you are running this job
# and your current directory is the right one where you want to execute the script
python3 test_script.py
