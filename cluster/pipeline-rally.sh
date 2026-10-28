#!/bin/bash
# self-play rounds: train the rally stage on the current return bank, then re-record the bank (and the rally
# statistics) with the new policy. Bank 0 comes from wheeled_coverage.pt. Qwen on m1 is stopped for the duration.
R=192.168.100.11
LOGS=$HOME/robotics/IsaacLab/logs/rsl_rl/badminton_kuka_allegro
REC="cd ~/robotics/badminton_rl && OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ../IsaacLab/.venv/bin/python scripts/record_rally.py"
CKPT=${CKPT:-$HOME/robotics/wheeled_coverage.pt}
START=${START:-13497}
docker stop qwen38-flash
for round in ${ROUNDS:-1 2}; do
  rsync -a $R:robotics/return_bank.npz ~/robotics/return_bank.npz
  ssh $R "cp ~/robotics/return_bank.npz ~/robotics/return_bank.r$((round-1)).npz"
  END=$((START + 3000))
  ~/robotics/supervise-stage.sh Badminton-MobileRally-KukaAllegro mobile_rally $END train-2node-rally-r$round $CKPT || break
  CKPT=$(ssh $R "ls $LOGS/*_mobile_rally/model_*.pt" | awk -F'model_' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
  START=$(basename $CKPT .pt | sed s/model_//)
  ssh $R "cp $CKPT ~/robotics/rally_r$round.pt"
  echo "$(date +%H:%M) round $round final: $CKPT"
  ssh $R "$REC --checkpoint ~/robotics/rally_r$round.pt --pairs 512 --rallies 1 --stat_rallies 2000 --bank_size 8000 \
    --serve_difficulty 0.7 --out /tmp/rally_r$round.npz --bank_out ~/robotics/return_bank.npz > ~/robotics/bank$round.log 2>&1"
  ssh $R "grep -E '^RALLY' ~/robotics/bank$round.log"
done
docker start qwen38-flash
echo PIPELINE_DONE
