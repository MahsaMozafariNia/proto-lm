#!/bin/bash

#SBATCH --job-name=Proto         # Job name
#SBATCH --cpus-per-task=2               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU

#SBATCH --output=%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer

model_name="Unso/roberta-large-finetuned-sst5"
max_seq_length=512
num_prototypes=1000
hidden_shape=1024
num_classes=5
cohsep_ratio=1
lr=3e-6
proto_training_weights=1
batch_size=128
logger_dir="sst5_logs/setting1_logs"
checkpoint_dir="sst5_logs/setting1_logs"
config_subdir="0001"
max_epochs=40
num_gpu=1
lambda0=0.2
lambda1=0.4
lambda2=0.4


## Use pretrained   
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
apptainer run --nv "/home/mozafarinia/Documents/sif_file.sif" \
python "sst5_proto.py"  -model_name=$model_name -max_seq_length=$max_seq_length -num_prototypes=$num_prototypes -hidden_shape=$hidden_shape -num_classes=$num_classes -cohsep_ratio=$cohsep_ratio -lr=$lr \
-proto_training_weights=$proto_training_weights -batch_size=$batch_size -logger_dir=$logger_dir -checkpoint_dir=$checkpoint_dir -config_subdir=$config_subdir -max_epochs=$max_epochs -num_gpu=$num_gpu \
-lambda0=$lambda0 -lambda1=$lambda1 -lambda2=$lambda2




echo "====================================================="
date
