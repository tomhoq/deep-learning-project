#!/bin/sh 


### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J 241268-deep-learning

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=12GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s242168@dtu.dk

### -- send notification at start --
##BSUB -B

### -- send notification at completion--
##BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_out/submission_%J.out
#BSUB -e job_out/submission_%J.err

# -- end of LSF options --


MODEL=unet_resnet34

REPO=${HOME}/deep-learning-project
OUT=$(find ${REPO}/job_out/${MODEL} -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1 | sed 's#.*/##p' | head -n 1)  # Get the latest run

# Activate venv
module load python3/3.10.14
source ${REPO}/.venv/bin/activate

# Run submission
python3 ${REPO}/src/make_submission_unet.py ${MODEL} ${REPO}/job_out/${MODEL}/${OUT}

# Submit to Kaggle
printf "\n[*] Submitting to Kaggle\n"
kaggle competitions submit -c airbus-ship-detection -f ${REPO}/job_out/${MODEL}/${OUT}/submission.csv -m "Automatic submission ${LSB_JOBID} - ${MODEL}"