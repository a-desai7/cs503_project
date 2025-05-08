#!/bin/bash

# Example use: ./start_node.sh 120 1 (2 hours, 1 gpu, 64GB memory, 4 CPUs)

# Default values
TIME=${1:-120}
GPUS=${2:-1}
MEM=${3:-64}
CPUS=${4:-4}

srun -t $TIME --gres=gpu:$GPUS --mem=${MEM}G --cpus-per-task=$CPUS --pty bash