#!/bin/bash

# Job
#SBATCH -p 3090
##A100-80GB 3090
# Tasks
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1

# GPUs+
#SBATCH --gres=gpu:1

## node 지정하기
##SBATCH --nodelist=n1 # specify nodelist
#SBATCH --nodes=1 # number of nodes

# CMDs
echo "Start: `date`"

# GPU Setting
echo "UUID GPU List - original"
nvidia-smi -L # 실제 할당받은 gpu

UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"

# Our Docker Setting
docker build -t nnunet_batch .
docker run --rm --name nnunet_batch_run --shm-size 16G --gpus ${GPULIST} -v /home2/ych000/data/nnUNet_patch_ablation/nnUNet_trained_models:/data/nnUNet/nnUNet_trained_models -v /home2/ych000/data/nnUNet_patch_ablation/nnUNet_preprocessed:/data/nnUNet/nnUNet_preprocessed -v /home2/ych000/data/nnUNet_patch_ablation/nnUNet_raw_data_base:/data/nnUNet/nnUNet_raw_data_base nnunet_batch


##sinfo -O "Partition:12,Nodes:5,Nodelist:20,Gres:22,GresUsed:34,features_act:25"