#!/usr/bin/env bash
# Stronger model on the v1 feature set (the one that scored best on test): full-quality LightGBM -> test scoring -> versions.
set -e
cd "/e/Amazon ML Challenge/eda"
export PYTHONIOENCODING=utf-8 FEATSUF=""
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a out/chain_v3.txt; }
log "training all_v3 (full quality, v1 features)"
FAST=0 python -u 15_train.py all_v3 > out/15_all_v3.txt 2>&1
grep -v Warn out/15_all_v3.txt | grep -v "^\[" | tail -4 | tee -a out/chain_v3.txt
log "scoring test with all_v3"
python -u 18_predict.py test all_v3 > out/18_all_v3.txt 2>&1
log "building versions"
python -u 19_make_submission.py --tag all_v3 --t 0.8 --t-country "France=0.9" --note "v3: full-quality LGB on v1 features" > out/19_v3a.txt 2>&1
python -u 19_make_submission.py --tag all_v3 --t 0.8 --t-country "France=0.9" --rule cluster_mod --note "v3 + cluster rules (moderate)" > out/19_v3b.txt 2>&1
log "done"; grep "all_v3" ../submissions/INDEX.md | tee -a out/chain_v3.txt
