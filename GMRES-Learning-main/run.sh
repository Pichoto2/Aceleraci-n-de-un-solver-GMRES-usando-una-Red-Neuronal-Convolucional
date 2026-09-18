#!/bin/bash
#SBATCH --job-name=MLGMRES
#SBATCH --output=%j.out
#SBATCH --error=%j.err
####SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --exclusive
####SBATCH --qos=vip
####SBATCH --ntasks-per-node=16
####SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
####SBATCH --nodelist=acp03
carpeta=30kProblemasDim28_e001_EachIt

#idSleepJob=$(sbatch --parsable run_sleep.sh)
mkdir $carpeta
mv ${SLURM_JOB_ID}.out $carpeta
mv ${SLURM_JOB_ID}.err $carpeta
cp -r src_dir $carpeta
echo "STARTING SLURM JOB"
echo ""
source ~/.bashrc
conda activate myenv
date
./monitor_GPU_jorge/gpu_logger.sh 1 ./$carpeta/gpu_log.csv & LOGGER_PID=$!
time python demo.py $carpeta
#time python demoCPU.py
#scancel $idSleepJob
#rm slurm-$idSleepJob.out
echo
date
kill -9 $LOGGER_PID
echo "FINISHED SLURM JOB"
