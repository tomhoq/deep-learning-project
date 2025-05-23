#!/bin/sh 


### General options

### –- specify queue --
#BSUB -q gpuv100
##SUB -q gpua100

### -- set the job Name --
#BSUB -J 241268-yolo

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request system-memory (per core)
#BSUB -R "rusage[mem=4GB]"

### -- Specify how the cores are distributed across nodes --
# The following means that all the cores must be on one single host
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o job_out/yolo_all_%J.out
#BSUB -e job_out/yolo_all_%J.err

# -- end of LSF options --





LR=0.00002
WD=0.0005
EPOCHS=20
BS=32
LN=0.5
LC=5





MODEL=yolo
REPO=${HOME}/deep-learning-project

OUT=${REPO}/job_out/${MODEL}/${LSB_JOBID}
mkdir -p ${OUT}

# Activate venv
module load python3/3.10.14
source ${REPO}/.venv/bin/activate


##### TRAINING #####
python3 ${REPO}/src/train_yolo.py --out-path ${OUT} --epochs ${EPOCHS} --learning-rate ${LR} --weight-decay ${WD} --batch-size ${BS} --lambda-noobj ${LN} --lambda-coord ${LC} 


##### EVALUATION #####
if [[ ! -d ${OUT}/evaluation ]]; then
    mkdir ${OUT}/evaluation
fi

python3 ${REPO}/src/evaluate_yolo.py ${OUT} 5


# ##### SUBMISSION #####
# python3 ${REPO}/src/make_submission_yolo.py ${OUT}

# printf "\n[*] Submitting to Kaggle\n"
# kaggle competitions submit -c airbus-ship-detection -f ${OUT}/submission.csv -m "Automatic submission YOLO ${LSB_JOBID} - With ${LOSS}"