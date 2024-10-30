#!/bin/bash
#SBATCH --job-name=5
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gypsum-rtx8000             # Partition to submit to (serial_requeue)
#SBATCH --mem=65536               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/work/pi_huiguan_umass_edu/mingcan/oslog/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=/work/pi_huiguan_umass_edu/mingcan/oslog/run_logs_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=6
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mingcanxiang@umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module load cuda/11.3.1
set -x -e
##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh" ]; then
        . "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh"
    else
        export PATH="/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
conda activate /home/mingcanxiang_umass_edu/.conda/envs/MTL
sleep 1

cd /home/mingcanxiang_umass_edu/Pruning/SNIP

python snip_train.py \
    --architecture mobilenetv2 \
    --batch_size 16 \
    --iters 20000 \
    --prune_ratio 0.5 \
    --save_dir /home/mingcanxiang_umass_edu/work_path/snip/mobilenetv2_0401/0.5/

sleep 1
exit