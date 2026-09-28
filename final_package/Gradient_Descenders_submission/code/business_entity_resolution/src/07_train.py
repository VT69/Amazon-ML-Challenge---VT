"""Train the LightGBM pair model on feat_train, evaluate macro F0.5 on the fold-0 validation S1 set.
Memory: ~7M x ~65 float32 (~1.8 GB) + LightGBM bins."""
import sys, os, glob, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from evalf import assign, macro_f05, sweep
import numpy as np, lightgbm as lgb, psutil
proc = psutil.Process(); mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
ID_COLS = ["rid", "s1", "true_s1", "y"]
NF = 40


TAG = sys.argv[1] if len(sys.argv) > 1 else "v1"
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""          # e.g. "India" to train/evaluate one country only


def load(d):
    out = []
    for f in sorted(glob.glob(f"{PQ}/{d}/{ONLY}*.parquet")):
        x = pl.read_parquet(f)
        te = f"{PQ}/feat_te/{d}/{os.path.basename(f)}"
        if os.path.exists(te):
            x = x.join(pl.read_parquet(te), on=["rid", "s1"], how="left")
        out.append(x)
    return pl.concat(out, how="diagonal_relaxed")


def X(df, feats):
    return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


SUF = os.environ.get("FEATSUF", "")
tr = load(f"feat{SUF}_train"); va = load(f"feat{SUF}_val")
feats = [c for c in tr.columns if c not in ID_COLS]
print("train", tr.shape, "pos", tr["y"].mean(), "| val", va.shape, mem(), flush=True)
dtr = lgb.Dataset(X(tr, feats), tr["y"].to_numpy(), feature_name=feats, free_raw_data=True)
del tr
FAST = os.environ.get("FAST", "1") == "1"      # fast settings while the CPU is shared with other jobs
es = va.filter(pl.col("rid").is_in(va["rid"].unique().sample(fraction=0.2 if FAST else 0.4, seed=5).implode()))
dva = lgb.Dataset(X(es, feats), es["y"].to_numpy(), reference=dtr); del es
params = dict(objective="binary", learning_rate=0.15 if FAST else 0.07, num_leaves=63 if FAST else 127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0, max_bin=63 if FAST else 127,
              num_threads=2 if FAST else 4, verbose=-1)
t0 = time.time()
m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(30 if FAST else 60), lgb.log_evaluation(25)])
print("trained", round(time.time() - t0), "s, best iter", m.best_iteration, mem(), flush=True)
os.makedirs(f"{PQ}/models", exist_ok=True); m.save_model(f"{PQ}/models/lgb_{TAG}.txt")
imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
print("top features:", [(f, round(g / 1e3)) for f, g in imp[:30]])

va = va.select("rid", "s1").with_columns(pl.Series("prob", m.predict(X(va, feats), num_iteration=m.best_iteration)))
va.write_parquet(f"{PQ}/val_pred_{TAG}.parquet")
gt = gt_pairs().select("s1", "rid")
s1_all = load_s1 = pl.scan_parquet(f"{PQ}/train_s1.parquet").select("entity_id", "country").collect()
s1_val = s1_all.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64).mod(NF) == 0)
res = sweep(va, gt, s1_val["entity_id"], ts=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
print("threshold sweep (t, macroF0.5):", res)
bt = max(res, key=lambda x: x[1])[0]
if ONLY: s1_val = s1_val.filter(pl.col("country") == ONLY)
res = sweep(va, gt, s1_val["entity_id"], ts=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]); print("sweep (restricted):", res); bt = max(res, key=lambda x: x[1])[0]
for c in ([ONLY] if ONLY else ["US", "India"]):
    ids = s1_val.filter(pl.col("country") == c)["entity_id"]
    f, d = macro_f05(assign(va, bt), gt, ids)
    sing = d.filter(pl.col("nt") == 0)["f"].mean(); non = d.filter(pl.col("nt") > 0)["f"].mean()
    print(f"{c}: F0.5={f:.4f}  singletons={sing:.4f}  non-singletons={non:.4f}  n={d.height}")
