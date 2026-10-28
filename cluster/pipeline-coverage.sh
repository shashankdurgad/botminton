#!/bin/bash
# run the wheeled full-court stage from the final wheeled target checkpoint (target finished 2026-10-26 14:32)
R=192.168.100.11
LOGS=$HOME/robotics/IsaacLab/logs/rsl_rl/badminton_kuka_allegro
TGT=$(ssh $R "ls $LOGS/*_mobile_target/model_*.pt" | awk -F'model_' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
ssh $R "cp $TGT ~/robotics/wheeled_target.pt"
echo "target final: $TGT"
CEND=$(( $(basename $TGT .pt | sed s/model_//) + 4000 ))
~/robotics/supervise-stage.sh Badminton-MobileCoverage-KukaAllegro mobile_coverage $CEND train-2node-wheeled-coverage \
  $HOME/robotics/wheeled_target.pt
