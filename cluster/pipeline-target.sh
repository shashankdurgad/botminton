#!/bin/bash
# run the wheeled target stage from the final wheeled hit checkpoint (hit finished 2026-10-26 05:00)
R=192.168.100.11
LOGS=$HOME/robotics/IsaacLab/logs/rsl_rl/badminton_kuka_allegro
HIT=$(ssh $R "ls $LOGS/*_mobile_hit/model_*.pt" | awk -F'model_' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
ssh $R "cp $HIT ~/robotics/wheeled_hit.pt"
echo "hit final: $HIT"
TEND=$(( $(basename $HIT .pt | sed s/model_//) + 4000 ))
~/robotics/supervise-stage.sh Badminton-MobileTarget-KukaAllegro mobile_target $TEND train-2node-wheeled-target \
  $HOME/robotics/wheeled_hit.pt
