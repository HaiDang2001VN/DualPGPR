#!/bin/bash
#SBATCH --job-name=gruPGPR     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --output=/media02/lnnam/workspace/nhanh/PGPR-v1/logs/%j-%x.out
#SBATCH --error=/media02/lnnam/workspace/nhanh/PGPR-v1/logs/%j-%x.err
#SBATCH --mail-user=20120246@student.hcmus.edu.vn

# Load cuda
spack load cuda@11.8.0

# Config conda
eval "$(conda shell.bash hook)"
conda activate torch39

# Change dir
cd /media02/lnnam/workspace/nhanh/PGPR-v1

chmod +x ./run.sh

./run.sh