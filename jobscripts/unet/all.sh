#!/bin/sh 


### General options

### –- specify queue --
##BSUB -q gpuv100
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J 241268-deep-learning

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00
# request 12GB of system-memory
#BSUB -R "rusage[mem=12GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s242168@dtu.dk

### -- send notification at start --
##BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify how the cores are distributed across nodes --
# The following means that all the cores must be on one single host
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_out/gpu_%J.out
#BSUB -e job_out/gpu_%J.err

# -- end of LSF options --



MODEL=unet

# LOSS=bce
# LOSS=dice
LOSS=jaccard

REPO=/zhome/82/4/212615/deep-learning-project

# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
	mkdir ${REPO}/job_out
fi

OUT=${REPO}/job_out/${MODEL}/${LSB_JOBID}
mkdir -p ${OUT}

# Activate venv
module load python3/3.10.14
source ${REPO}/.venv/bin/activate


##### TRAINING #####
python3 ${REPO}/src/train.py ${MODEL} ${LOSS} ${OUT}


##### EVALUATION #####
if [[ ! -d ${OUT}/evaluation ]]; then
    mkdir ${OUT}/evaluation
fi

python3 ${REPO}/src/evaluate.py ${MODEL} ${OUT} 5


##### SUBMISSION #####
python3 ${REPO}/src/make_submission.py ${MODEL} ${OUT}

printf "\n[*] Submitting to Kaggle\n"
kaggle competitions submit -c airbus-ship-detection -f ${OUT}/submission.csv -m "Automatic submission ${LSB_JOBID}"


##### FINISHING #####
# Move job stdout/stderr to correct folder
mv ${REPO}/job_out/gpu_${LSB_JOBID}* ${OUT}