#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=generate-pycbc

#SBATCH -A mlgw

# number of requested nodes
#SBATCH --nodes=2
##SBATCH --ntasks=1

#SBATCH --exclusive

##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=80
##SBATCH --mem=64G
#SBATCH --part=cpu1

# requested runtime
#SBATCH --time=48:00:00

# Path to the standard output and error files relative to the working directory
#SBATCH --output=/veracruz/home/g/ggoncalves/gw_classification/jobs/job_generate_%j.log

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ml load OpenBLAS
ml load Python
ml load OpenMPI
ml load libsndfile
ml load HDF5

HOME_DIR=/veracruz/home/g/ggoncalves/gw_classification

# Install requirements
python -m venv ${HOME_DIR}/utils/venv
source ${HOME_DIR}/utils/venv/bin/activate

pip install --upgrade pip==21.3.1
pip install -r ${HOME_DIR}/requirements.txt

N=160
POLARIZATION=hp
MODEL_FILE=${HOME_DIR}/utils/models.csv

TYPE_DS="train"

JOB_DIR=${HOME_DIR}/utils

SCRIPT1=${JOB_DIR}/generate_mpi.py

# SCRIPT2=${JOB_DIR}/spec_ex.py
# SCRIPT3=${JOB_DIR}/merge_multi.py

OUT_DIR=/veracruz/projects/m/mlgw/class_dataset

for EOS in BSk20 TM1 SLY9

do

EOS_DIR=${OUT_DIR}/${TYPE_DS}/${EOS}
mkdir -p ${EOS_DIR}

mpirun python ${SCRIPT1} ${N} \
-w ${EOS_DIR} \
-p ${POLARIZATION} \
--model-name ${EOS} \
--model-file ${MODEL_FILE} \
--process
done

# /home/goncalo/gw_env/bin/python ${SCRIPT2} -w ${OUT_DIR}/${TYPE_DS}
# /home/goncalo/gw_env/bin/python ${SCRIPT3} -w ${OUT_DIR}/${TYPE_DS}
