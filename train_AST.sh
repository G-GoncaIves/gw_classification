# Params:
model=AST

# Training Parameters:
lr=1e-4
epochs=100

# Classifier Parameters:
freqm=128
timem=128
fstride=10
tstride=10

# DataLoader Parameters:
batch_size=50
num_workers=1

# DataSet Paramaeters:
n_class=3
data_type=spectrograms

# References:
exp_dir=/home/goncalo/gw_classification/AST/output_files # Path to dir used to store all outputs
tr_data=/home/goncalo/gw_classification/dataset/train/dataset.h5 # Path to data used for trainig
te_data=/home/goncalo/gw_classification/dataset/train/dataset.h5  # Path to data used for testing

CUDA_CACHE_DISABLE=1 python3 -W ignore run.py $model \
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
