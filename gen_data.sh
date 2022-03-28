#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=generate-pycbc

#SBATCH -A mlgw

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --part=cpu2
#SBATCH --exclusive

# requested runtime
#SBATCH --time=24:00:00

# Path to the standard output and error files relative to the working directory
#SBATCH --output=/veracruz/home/g/ggoncalves/gw_classification/jobs/job_generate_%j.log

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ml load Python
ml load libsndfile
ml load HDF5

HOME_DIR=/veracruz/home/g/ggoncalves/gw_classification
SCRIPT_DIR=${HOME_DIR}/utils
OUT_DIR=/veracruz/projects/m/mlgw/class_dataset

# Install requirements
python -m venv ${HOME_DIR}/utils/venv
source ${HOME_DIR}/utils/venv/bin/activate

pip install --upgrade pip
pip install -r ${HOME_DIR}/requirements.txt

N=100000
POLARIZATION=hp
MODEL_FILE=${SCRIPT_DIR}/models.csv
TYPE_DS="train"

SCRIPT_1=${SCRIPT_DIR}/generate_joblib.py
SCRIPT_2=${SCRIPT_DIR}/merge_multi.py
SCRIPT_3=${SCRIPT_DIR}/spec_ex.py


for EOS in DD2 NL3
do

EOS_DIR=${OUT_DIR}/${TYPE_DS}/${EOS}
mkdir -p ${EOS_DIR}

python ${SCRIPT_1} ${N} \
-w ${EOS_DIR} \
-p ${POLARIZATION} \
--model-name ${EOS} \
--model-file ${MODEL_FILE} \
--process
done

python ${SCRIPT_2} -w ${OUT_DIR}/${TYPE_DS}
python ${SCRIPT_3} -w ${OUT_DIR}/${TYPE_DS}

rm -rf ${HOME_DIR}/utils/venv
