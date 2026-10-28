#!/bin/bash
# finish the wheeled hit stage (resumed after the NCCL stall at 2209), then run the wheeled target stage
R=192.168.100.11
LOGS=$HOME/robotics/IsaacLab/logs/rsl_rl/badminton_kuka_allegro
~/robotics/supervise-stage.sh Badminton-MobileHit-KukaAllegro mobile_hit 5500 train-2node-wheeled-hit \
  $LOGS/2026-10-26_00-30-14_mobile_hit/model_2200.pt || exit 1
HIT=$(ssh $R "ls $LOGS/*_mobile_hit/model_*.pt" | awk -F'model_' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
ssh $R "cp $HIT ~/robotics/wheeled_hit.pt"
echo "hit final: $HIT"
TEND=$(( $(basename $HIT .pt | sed s/model_//) + 4000 ))
~/robotics/supervise-stage.sh Badminton-MobileTarget-KukaAllegro mobile_target $TEND train-2node-wheeled-target \
  $HOME/robotics/wheeled_hit.pt
