#!/usr/bin/env bash
# Model v2: synthetic sibling clusters + group features + full-quality training -> test scoring -> versions.
set -e
cd "/e/Amazon ML Challenge/eda"
export PYTHONIOENCODING=utf-8 FEATSUF=2
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a out/chain_v2.txt; }
log "waiting for synthetic generation"
until grep -q "synthetic done" out/22_synth_log.txt 2>/dev/null; do sleep 30; done
log "building v2 features (train + test-like validation)"
python -u 14_build.py > out/14_v2_log.txt 2>&1
log "token log-odds for v2 features"
python -u 16_te.py > out/16_te_v2_log.txt 2>&1
log "training all_v2 (full quality)"
FAST=0 python -u 15_train.py all_v2 > out/15_all_v2.txt 2>&1
grep -v Warn out/15_all_v2.txt | grep -v "^\[" | tail -5 | tee -a out/chain_v2.txt
log "scoring test with all_v2"
python -u 18_predict.py test all_v2 > out/18_all_v2.txt 2>&1
for t in 0.7 0.8 0.9; do
  python -u 19_make_submission.py --tag all_v2 --t $t --note "v2: synthetic sibling clusters + group features + full-quality LGB" >> out/19_v2_log.txt 2>&1
done
log "v2 submissions written"; grep "all_v2" ../submissions/INDEX.md | tee -a out/chain_v2.txt
