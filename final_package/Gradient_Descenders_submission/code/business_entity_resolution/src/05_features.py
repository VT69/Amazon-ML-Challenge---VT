"""Build pair-feature tables from cached train candidates.
Query fold = fold of its true S1 (if matched) else of its own id  (fold = numeric id % 40).
  val   : fold 0, all queries & all candidates  -> {PQ}/feat_val/*.parquet
  train : folds 1..5, 25% of queries           -> {PQ}/feat_train/*.parquet
Usage: python 14_build.py [max_files] [max_queries_per_file]   (limits only for smoke tests)"""
import sys, os, glob, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from feats import s1_side, r_side, pair_features
from group import group_table, add_group
import numpy as np, psutil
max_files = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
NF, TRAIN_FOLDS, TRAIN_FRAC = 40, range(1, 6), 0.25
SUF = os.environ.get("FEATSUF", "")          # "" = v1 feature set, "2" = v2 (+ synthetic siblings, + group features)
DV, DT = f"feat{SUF}_val", f"feat{SUF}_train"
GROUPS = {}
proc = psutil.Process(); mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
idnum = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
truth = gt_pairs().select("rid", pl.col("s1").alias("true_s1"))
for d in [DV, DT]: os.makedirs(f"{PQ}/{d}", exist_ok=True)
def process(country, files):
    if country not in SIDES:
        t0 = time.time(); SIDES[country] = s1_side("train", country); print(country, "S1 side", round(time.time() - t0), "s", mem(), flush=True)
    S, idf = SIDES[country]
    for fn in files:
        base = os.path.basename(fn)
        if os.path.exists(f"{PQ}/{DT}/{base}"): continue
        if time.time() - os.path.getmtime(fn) < 120: continue      # may still be being written
        t1 = time.time()
        C = pl.read_parquet(fn).join(truth, on="rid", how="left")
        C = C.with_columns(pl.when(pl.col("true_s1").is_not_null()).then(idnum("true_s1")).otherwise(idnum("rid")).mod(NF).alias("fold"),
                           (pl.col("s1") == pl.col("true_s1")).fill_null(False).alias("y"))
        q = C.select("rid", "fold").unique()
        if max_q: q = q.sample(min(max_q, q.height), seed=0)
        val_q = q.filter(pl.col("fold") == 0)
        tr_q = q.filter(pl.col("fold").is_in(list(TRAIN_FOLDS))).sample(fraction=TRAIN_FRAC, seed=1)
        for name, qs in [(DV, val_q), (DT, tr_q)]:
            sub = C.join(qs.select("rid"), on="rid", how="semi")
            Q = r_side("train", country, qs.select(pl.col("rid").alias("entity_id")))
            F = pair_features(sub.drop("fold"), Q, S, idf)
            if SUF:
                if country not in GROUPS: GROUPS[country] = group_table("train", country)
                F = add_group(F, GROUPS[country])
            F.write_parquet(f"{PQ}/{name}/{base}")
            print(f"  {base} {name}: q={qs.height} pairs={F.height} pos={F['y'].sum()}", flush=True)
        print(f"  {base} {time.time() - t1:.0f}s {mem()}", flush=True)


SIDES = {}
while True:
    cand_done = os.path.exists(f"{PQ}/train_cand_done.flag")
    todo = [f for f in sorted(glob.glob(f"{PQ}/train_cand/*.parquet"))[:max_files]
            if not os.path.exists(f"{PQ}/{DT}/{os.path.basename(f)}")]
    for country in ["India", "US"]:
        fs = [f for f in todo if os.path.basename(f).startswith(country + "_")]
        if fs: process(country, fs)
    left = [f for f in glob.glob(f"{PQ}/train_cand/*.parquet") if not os.path.exists(f"{PQ}/{DT}/{os.path.basename(f)}")]
    if (cand_done and not left) or max_q: break
    time.sleep(60)
print("build done", flush=True)
