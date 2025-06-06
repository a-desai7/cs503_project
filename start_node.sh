#!/bin/bash

# Example use: ./start_node.sh 120 1 (2 hours, 1 gpu, 64GB memory, 4 CPUs)

# Default values
TIME=${1:-120}
GPUS=${2:-1}
MEM=${3:-64}
CPUS=${4:-8}

srun -t $TIME -A cs-503 --qos=cs-503 --gres=gpu:$GPUS --mem=${MEM}G --cpus-per-task=$CPUS --pty bash