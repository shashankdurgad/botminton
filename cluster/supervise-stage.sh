#!/bin/bash
# usage: supervise-stage.sh <task> <run_name> <end_iteration> <log_prefix> <first_checkpoint>
# Runs a 2-node stage and, if it crashes (e.g. an NCCL timeout from a stalled rank), resumes both ranks from the
# newest checkpoint of that run, up to 5 times. Each attempt logs to ~/robotics/<log_prefix>.<attempt>.log on both
# nodes. A checkpoint from the same run resumes with max_iterations = end_iteration - its iteration; a checkpoint
# from the previous stage starts the stage with the task's default length. Prints STAGE_DONE or STAGE_FAILED.
TASK=$1; RUN=$2; END=$3; LOG=$4; CKPT=$5
R=192.168.100.11
LOGS=$HOME/robotics/IsaacLab/logs/rsl_rl/badminton_kuka_allegro
for attempt in 0 1 2 3 4 5; do
  if [ $attempt -gt 0 ]; then
    LATEST=$(ssh $R "ls $LOGS/*_$RUN/model_*.pt 2>/dev/null" | awk -F'model_' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$LATEST" ] && CKPT=$LATEST
  fi
  MAXI=""
  if [[ "$CKPT" == *_$RUN/model_* ]]; then MAXI=$((END - $(basename "$CKPT" .pt | sed 's/model_//'))); fi
  ssh $R "cp $CKPT ~/robotics/stage_resume.pt" && rsync -a $R:robotics/stage_resume.pt ~/robotics/stage_resume.pt
  tmux kill-session -t train-2node 2>/dev/null; ssh $R 'tmux kill-session -t train-2node 2>/dev/null'
  pkill -f '[t]rain_multigpu' 2>/dev/null; ssh $R "pkill -f '[t]rain_multigpu'"; sleep 20
  L=$LOG.$attempt.log
  echo "$(date +%H:%M) attempt $attempt: $TASK from $CKPT ${MAXI:+(+$MAXI it)} -> $L"
  ssh $R "tmux new -d -s train-2node 'bash ~/robotics/train-2node-stage.sh 0 $TASK $HOME/robotics/stage_resume.pt $MAXI 2>&1 | tee ~/robotics/$L'"
  tmux new -d -s train-2node "bash ~/robotics/train-2node-stage.sh 1 $TASK $HOME/robotics/stage_resume.pt $MAXI 2>&1 | tee ~/robotics/$L"
  until ssh $R "grep -q TRAIN_EXIT ~/robotics/$L 2>/dev/null"; do sleep 60; done
  if ssh $R "grep -q TRAIN_EXIT=0 ~/robotics/$L"; then echo "$(date +%H:%M) STAGE_DONE"; exit 0; fi
  echo "$(date +%H:%M) crashed at $(ssh $R "grep -E 'Learning iteration' ~/robotics/$L | tail -1 | grep -oE '[0-9]+/[0-9]+'")"
done
echo STAGE_FAILED; exit 1
