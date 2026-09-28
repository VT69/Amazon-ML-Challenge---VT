"""Score every candidate pair of a split with a saved model.  python 18_predict.py <split> <model_tag>
Streams candidate chunks in sub-batches (~60k queries, ~1.4M pairs) so peak RAM stays ~3-4 GB.
Output: data/pred_<split>_<tag>/<chunk>.parquet with (rid, s1, prob, p_rank) for ALL candidates, so any
threshold / decision rule can be applied later without re-running the model."""
import sys, os, glob, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from feats import s1_side, r_side, pair_features
from te import te_features
from group import group_table, add_group
import numpy as np, lightgbm as lgb, psutil
split, tag = sys.argv[1], sys.argv[2]
SUB = 60_000
proc = psutil.Process(); mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
m = lgb.Booster(model_file=f"{PQ}/models/lgb_{tag}.txt")
feats = m.feature_name()
use_te = any(f.startswith("te_") for f in feats)
use_grp = any(f.startswith("g_") for f in feats)
T = pl.read_parquet(f"{PQ}/te_table.parquet") if use_te else None
out = f"{PQ}/pred_{split}_{tag}"; os.makedirs(out, exist_ok=True)
# optional speed-up: only run the model on pairs another model already gave >= PREFILTER_MIN (others get prob 0).
# Features are still computed on the FULL candidate list, so within-record context features are unchanged.
PRE_TAG, PRE_MIN = os.environ.get("PREFILTER_TAG", ""), float(os.environ.get("PREFILTER_MIN", "0.001"))
# PREFILTER_LIST: optional file with chunk names to prefilter (the submitted v019 prefiltered exactly these 22 chunks;
# the other 20 were scored in full). Without it, PREFILTER_TAG applies to every chunk.
_pl = os.environ.get("PREFILTER_LIST", "")
PRE_CHUNKS = set(open(_pl).read().split()) if _pl else None
files = sorted(glob.glob(f"{PQ}/{split}_cand/*.parquet"))
for country in sorted({os.path.basename(f).split("_")[0] for f in files}):
    todo = [f for f in files if os.path.basename(f).startswith(country + "_") and not os.path.exists(f"{out}/{os.path.basename(f)}")]
    if not todo: continue
    t0 = time.time(); S, idf = s1_side(split, country); print(country, "S1 side", round(time.time() - t0), "s", mem(), flush=True)
    G = group_table(split, country) if use_grp else None
    for fn in todo:
        t1 = time.time(); C = pl.read_parquet(fn)
        rids = C.select("rid").unique(maintain_order=True)["rid"]
        parts = []
        for b in range(0, len(rids), SUB):
            r = rids.slice(b, SUB)
            sub = C.filter(pl.col("rid").is_in(r.implode()))
            Q = r_side(split, country, pl.DataFrame({"entity_id": r}))
            F = pair_features(sub, Q, S, idf)
            if use_grp:
                F = add_group(F, G)
            if use_te:
                F = F.join(te_features(F.select("rid", "s1"), split, country, T), on=["rid", "s1"], how="left")
            if PRE_TAG and (PRE_CHUNKS is None or os.path.basename(fn) in PRE_CHUNKS):
                pre = pl.read_parquet(f"{PQ}/pred_{split}_{PRE_TAG}/{os.path.basename(fn)}", columns=["rid", "s1", "prob"]).rename({"prob": "_pre"})
                F = F.join(pre, on=["rid", "s1"], how="left")
                keep = (F["_pre"].fill_null(1.0) >= PRE_MIN).to_numpy()
            else:
                keep = np.ones(F.height, dtype=bool)
            prob = np.zeros(F.height, dtype=np.float32)
            if keep.any():
                X = F.filter(pl.Series(keep)).select([pl.col(c).cast(pl.Float32) if c in F.columns else pl.lit(None, pl.Float32).alias(c) for c in feats]).to_numpy()
                prob[keep] = m.predict(X).astype(np.float32)
            parts.append(F.select("rid", "s1", "p_rank").with_columns(pl.Series("prob", prob)))
            del F, X, Q, sub
        pl.concat(parts).write_parquet(f"{out}/{os.path.basename(fn)}")
        print(f"  {os.path.basename(fn)} pairs={C.height} {time.time() - t1:.0f}s {mem()}", flush=True)
    del S, idf
print("done", flush=True)
