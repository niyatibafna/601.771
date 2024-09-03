#!/bin/bash
#$ -N roberta
#$ -wd /export/b08/nbafna1/projects/courses/601.771-ssl/
#$ -m e
#$ -j y -o /export/b08/nbafna1/projects/courses/601.771-ssl/slurm_logs/roberta.log

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=50G,mem_free=50G,gpu=1,hostname=!c08*&!c07*&!c04*&!c24*&!c25*&c*

# Submit to GPU queue
#$ -q g.q

source ~/.bashrc
which python

conda deactivate
conda activate basic
which python

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu -n 1

echo "HOSTNAME: $(hostname)"
echo
echo CUDA in ENV:
env | grep CUDA
echo
echo SGE in ENV:
env | grep SGE

set -x # print out every command that's run with a +
nvidia-smi


cd /export/b08/nbafna1/projects/courses/601.771-ssl/
## SCRIPT TO RUN
python roberta.py
