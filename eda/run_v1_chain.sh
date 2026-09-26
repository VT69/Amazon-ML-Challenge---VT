#!/usr/bin/env bash
# Overnight chain: features -> token log-odds -> model all_v1 -> test scoring -> versioned submissions.
set -e
cd "/e/Amazon ML Challenge/eda"
export PYTHONIOENCODING=utf-8
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a out/chain_v1.txt; }

log "waiting for feature builder"
until grep -q "build done" out/14_log_d.txt; do sleep 60; done
log "token log-odds (16_te)"
python -u 16_te.py > out/16_te_log.txt 2>&1
log "training all_v1"
FAST=1 python -u 15_train.py all_v1 > out/15_all_v1.txt 2>&1
grep -v Warn out/15_all_v1.txt | grep -v "^\[" | tail -5 | tee -a out/chain_v1.txt
log "waiting for test candidates"
until grep -q "^done" out/13_test_log.txt 2>/dev/null; do sleep 60; done
log "scoring test candidates"
python -u 18_predict.py test all_v1 > out/18_all_v1.txt 2>&1
BEST=$(python - <<'EOF'
import sys; sys.path.insert(0, ".")
from common import *
from evalf import sweep
vp = pl.read_parquet(f"{PQ}/val_pred_all_v1.parquet"); gt = gt_pairs().select("s1", "rid")
ids = pl.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id"]).filter(pl.col("entity_id").str.slice(3).cast(pl.Int64).mod(40) == 0)["entity_id"]
r = sweep(vp, gt, ids, ts=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3])
print(max(r, key=lambda x: x[1])[0])
EOF
)
log "best val threshold $BEST"
python -u 19_make_submission.py --tag all_v1 --t "$BEST" --note "first full model, best local threshold" >> out/19_log.txt 2>&1
python -u 19_make_submission.py --tag all_v1 --t 0.2 --note "precision-leaning threshold" >> out/19_log.txt 2>&1
python -u 19_make_submission.py --tag all_v1 --t 0.5 --note "strongly precision-leaning threshold" >> out/19_log.txt 2>&1
log "submissions written"; tail -5 ../submissions/INDEX.md | tee -a out/chain_v1.txt
