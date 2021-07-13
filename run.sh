#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH --ntasks-per-node=3 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2 # number of cores per task
#SBATCH --gres=gpu:3
#SBATCH --nodelist=atlas # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /work/yyaoqing/oliver/nlp
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate flbackdoor
export PYTHONUNBUFFERED=1


## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --s_norm 20 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug1.log 2> ./logs/debug1.err &
## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --s_norm 10 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug2.log 2> ./logs/debug2.err &
## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --s_norm 5 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug3.log 2> ./logs/debug3.err &
## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --s_norm 1 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug4.log 2> ./logs/debug4.err &
## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --s_norm 0.5 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug5.log 2> ./logs/debug5.err
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python training_adver_update.py --s_norm 2 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug6.log 2> ./logs/debug6.err &
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python training_adver_update.py --s_norm 3 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug7.log 2> ./logs/debug7.err &
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python training_adver_update.py --s_norm 4 --run_slurm 1 --sentence_id_list 0 --start_epoch 1 --diff_privacy True > ./logs/debug8.log 2> ./logs/debug8.err 

wait
date
