#!/bin/sh 


### General options

### –- specify queue --
#BSUB -q gpuv100
##BSUB -q gpua100

### -- set the job Name --
#BSUB -J 241268-deep-learning

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00

# request system-memory (per core)
#BSUB -R "rusage[mem=2GB]"

##BSUB -R "select[gpu80gb]"

### -- Specify how the cores are distributed across nodes --
# The following means that all the cores must be on one single host
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file --
#BSUB -o job_out/resnet34unet/%J/all_%J.out
#BSUB -e job_out/resnet34unet/%J/all_%J.err

# -- end of LSF options --




MODEL=resnet34unet
LOSS=cross_entropy


REPO=${HOME}/deep-learning-project


OUT=${REPO}/job_out/${MODEL}/${LSB_JOBID}
mkdir -p ${OUT}

# Activate venv
module load python3/3.10.14
source ${REPO}/.venv/bin/activate


##### TRAINING #####
python3 ${REPO}/src/train_resnet.py ${OUT}


#### FINISHING ####
# mv ${REPO}/job_out/${MODEL}/all_${LSB_JOBID}* ${OUT}

