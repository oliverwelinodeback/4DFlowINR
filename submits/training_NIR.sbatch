#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user jebisbal@uc.cl
#SBATCH --output /proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/outputfiles/all_output_train3.out
#SBATCH --error /proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/outputfiles/all_output_train3.err
# export SSL_CERT_FILE="/proj/multipress/users/x_javbi/ICUNET/SebastianJofre/cacert.pem"
# cd  /proj/multipress/users/x_javbi/SebastianJofre/

export SSL_CERT_FILE="/proj/multipress/users/x_javbi/SebastianJofre/cacert.pem"
cd /proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/src
apptainer exec --nv /proj/multipress/users/x_oliwe/torch_2.2.1-gpu.sif python trainer_wandb.py

# singularity run --nv --env WANDB_API_KEY=$WANDB_API_KEY /proj/multipress/users/x_javbi/unet2.sif python main.py 0