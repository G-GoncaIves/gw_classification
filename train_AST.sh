#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name="ast_train"

#SBATCH -A mlgw

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --part=gpu

#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# requested runtime
#SBATCH --time=24:00:00

# Path to the standard output and error files relative to the working directory
#SBATCH --output=/veracruz/home/g/ggoncalves/gw_classification/jobs/job_train_%j.log

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

# Cluster References:
OUT_DIR=/veracruz/projects/m/mlgw/
HOME_DIR=/veracruz/home/g/ggoncalves/gw_classification


ml load Python
ml load HDF5
ml load libsndfile

python -m venv ${HOME_DIR}/AST/venv
source ${HOME_DIR}/AST/venv/bin/activate

pip install --upgrade pip
#pip install -r ${HOME_DIR}/AST/requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm
pip install sklearn
pip install ast
pip install timm==0.4.5
pip install h5py
pip install numpy 
pip install librosa
pip install matplotlib
pip install wget
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Params:
model=AST

# Training Parameters:
lr=1e-4
epochs=1

# Classifier Parameters:
freqm=128
timem=128
fstride=10
tstride=10

# DataLoader Parameters:
batch_size=500
num_workers=1

# DataSet Paramaeters:
n_class=3
data_type=spectrograms

# File References:
exp_dir=${OUT_DIR}/output_files # Path to dir used to store all outputs
tr_data=${OUT_DIR}/class_dataset/train/dataset.h5 # Path to data used for trainig
te_data=${OUT_DIR}/class_dataset/train/dataset.h5  # Path to data used for testing

SCRIPT=${HOME_DIR}/run.py

CUDA_CACHE_DISABLE=1 python -W ignore ${SCRIPT} $model \
--data_train $tr_data \
--num_workers $num_workers \
--data_val $te_data \
--exp_dir $exp_dir \
--lr $lr \
--epochs $epochs \
--batch_size $batch_size \
--save_model \
--freqm $freqm \
--timem $timem \
--tstride $tstride \
--fstride $fstride \
--data_shuffle \
--n_class $n_class \
--imagenet_pretrain \
--audioset_pretrain \
--data_type $data_type \
--verbose
