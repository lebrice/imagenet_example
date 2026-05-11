#!/bin/bash
#SBATCH --output=logs/%j/slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0:05:00

project_name="imagenet_example"
results_path="logs"
project_root="repos/imagenet_example"

# Minimal test job for cluv submit.
echo "hostname: $(hostname)"
echo "GIT_COMMIT=${GIT_COMMIT:?GIT_COMMIT is not set. Use 'cluv submit' to submit this job script.}"

# Setup the repo in $SLURM_TMPDIR, so the code can change in the project without affecting the job.
echo "Preparing the repo and virtual environment in $SLURM_TMPDIR"
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES --input=all bash -e <<END
cd $SLURM_TMPDIR
git clone $project_root
cd $SLURM_TMPDIR/$project_name
git checkout --detach $GIT_COMMIT
exec uv sync
END

# These environment variables are used by torch.distributed and should ideally be set
# before running the python script, or at the very beginning of the python script.
# Master address is the hostname of the first node in the job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

# Run the actual job command passed as an argument ('python main.py' for example)
echo "Running command: $@"
# Note: This `--gres-flags=allow-task-sharing` is required to allow tasks on the same node to access
# GPUs allocated to other tasks on that node. Without this flag, --gpus-per-task=1 would isolate
# each task to only see its own GPU, which can cause some mysterious NCCL errors.
srun --gres-flags=allow-task-sharing uv --directory=$SLURM_TMPDIR/$project_name run "$@"

# Copy results (if any) from the local storage back to the results dir (eg in $SCRATCH)
echo "Copying logs from $SLURM_TMPDIR/$project_name/$results_path to $project_root/$results_path"
srun --ntasks-per-node=1 rsync --update --recursive $SLURM_TMPDIR/$project_name/$results_path $project_root/
