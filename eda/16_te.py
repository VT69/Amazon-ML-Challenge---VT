"""Out-of-fold token log-odds for name words present in only one side of a pair.
Table source: top-1 candidate pairs (p_rank == 0) of queries in folds 6..39 (disjoint from train folds 1..5 and val fold 0).
  lo_extra(w): does w appearing in the QUERY name but not in the S1 name indicate a non-match?
  lo_miss(w) : same for w in the S1 name but not in the query name.
Then writes per-pair aggregates for every feat_train / feat_val file to feat_te/<dir>/<file>.
Unknown words (e.g. new French vocabulary) get null = neutral."""
import sys, os, glob; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from te import diff_tokens, te_features
import numpy as np
NF, ALPHA = 40, 20.0
idnum = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
truth = gt_pairs().select("rid", pl.col("s1").alias("true_s1"))

if not os.path.exists(f"{PQ}/te_table.parquet"):
    parts = []
    for fn in sorted(glob.glob(f"{PQ}/train_cand/*.parquet")):
        country = os.path.basename(fn).split("_")[0]
        C = pl.read_parquet(fn, columns=["rid", "s1", "p_rank"]).filter(pl.col("p_rank") == 0).join(truth, on="rid", how="left")
        C = C.with_columns(pl.when(pl.col("true_s1").is_not_null()).then(idnum("true_s1")).otherwise(idnum("rid")).mod(NF).alias("fold"))
        C = C.filter(pl.col("fold") >= 6).with_columns((pl.col("s1") == pl.col("true_s1")).fill_null(False).alias("y"))
        parts.append(diff_tokens(C.select("rid", "s1", "y"), "train", country).select("y", "extra", "miss"))
    D = pl.concat(parts); p0 = D["y"].mean(); prior = np.log(p0 / (1 - p0))
    tabs = []
    for side in ["extra", "miss"]:
        t = D.select("y", side).explode(side).drop_nulls(side).group_by(side).agg(pl.col("y").sum().alias("pos"), pl.len().alias("n"))
        t = t.with_columns((((pl.col("pos") + ALPHA * p0) / (pl.col("n") - pl.col("pos") + ALPHA * (1 - p0))).log() - prior).alias(f"lo_{side}"))
        tabs.append(t.rename({side: "tok"}).select("tok", f"lo_{side}", pl.col("n").alias(f"n_{side}")))
    T = tabs[0].join(tabs[1], on="tok", how="full", coalesce=True)
    T.write_parquet(f"{PQ}/te_table.parquet")
    print("table", T.height, "p0", round(p0, 4))
    for side in ["extra", "miss"]:
        print(side, "most non-match-like:", T.filter(pl.col(f"n_{side}") > 200).sort(f"lo_{side}").head(25).select("tok", f"lo_{side}").rows())
T = pl.read_parquet(f"{PQ}/te_table.parquet")
SUF = os.environ.get("FEATSUF", "")
for d in [f"feat{SUF}_train", f"feat{SUF}_val"]:
    os.makedirs(f"{PQ}/feat_te/{d}", exist_ok=True)
    for fn in sorted(glob.glob(f"{PQ}/{d}/*.parquet")):
        out = f"{PQ}/feat_te/{d}/{os.path.basename(fn)}"
        if os.path.exists(out): continue
        country = os.path.basename(fn).split("_")[0]
        te_features(pl.read_parquet(fn, columns=["rid", "s1"]), "train", country, T).write_parquet(out)
    print(d, "done", flush=True)
