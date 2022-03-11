#!/bin/bash
source /home/goncalo/gw_env/bin/activate

N=80
POLARIZATION=hp
MODEL_FILE="/home/goncalo/GW_pycbc/AST/modules/models.csv"
PROCESS=False
TYPE_DS="valid"

JOB_DIR=/home/goncalo/GW_pycbc/AST/modules
SCRIPT=${JOB_DIR}/generate_mpi.py


for EOS in BSk20 TM1 SLY9 NL3 GM1 DDHd DD2 BSR6 BSR2 BSk21

do
mkdir test_dataset/${TYPE_DS}/${EOS}

EOS_DIR=${JOB_DIR}/test_dataset/${TYPE_DS}/${EOS}

mpirun /home/goncalo/gw_env/bin/python ${SCRIPT} \
--process ${N} \
-w ${EOS_DIR} \
-p ${POLARIZATION} \
--model-name ${EOS} \
--model-file ${MODEL_FILE} \
--process
done

/home/goncalo/gw_env/bin/python test_md.py
/home/goncalo/gw_env/bin/python merge_multi.py