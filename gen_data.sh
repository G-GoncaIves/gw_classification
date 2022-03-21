#!/bin/bash
source /home/goncalo/gw_env/bin/activate

N=16
POLARIZATION=hp
MODEL_FILE="/home/goncalo/GW_pycbc/AST/modules/models.csv"
TYPE_DS="train"

JOB_DIR=/home/goncalo/gw_classification/utils

SCRIPT1=${JOB_DIR}/generate_mpi.py
SCRIPT2=${JOB_DIR}/spec_ex.py
SCRIPT3=${JOB_DIR}/merge_multi.py

OUT_DIR=/home/goncalo/gw_classification/dataset

for EOS in BSk20 TM1 SLY9

do

EOS_DIR=${OUT_DIR}/${TYPE_DS}/${EOS}
mkdir -p ${EOS_DIR}

mpirun /home/goncalo/gw_env/bin/python ${SCRIPT1} ${N} \
-w ${EOS_DIR} \
-p ${POLARIZATION} \
--model-name ${EOS} \
--model-file ${MODEL_FILE} \
--process
done

/home/goncalo/gw_env/bin/python ${SCRIPT2} -w ${OUT_DIR}/${TYPE_DS}
/home/goncalo/gw_env/bin/python ${SCRIPT3} -w ${OUT_DIR}/${TYPE_DS}
