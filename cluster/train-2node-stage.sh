#!/bin/bash
# usage: train-2node-stage.sh <node_rank> <task> [checkpoint] [max_iterations]
cd ~/robotics/IsaacLab
export OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
export NCCL_SOCKET_IFNAME=enp1s0f0np0 GLOO_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
CKPT="${3:+--checkpoint $3} ${4:+--max_iterations $4}"
.venv/bin/isaaclab train_multigpu --nnodes 2 --node_rank $1 --num_gpus 1 \
  --master_addr 192.168.100.11 --master_port 29500 \
  --rl_library rsl_rl --task $2 --num_envs 4096 $CKPT
echo TRAIN_EXIT=$?
