"""Competition metric: macro F0.5 over Source-1 entities (singletons included)."""
import numpy as np
import polars as pl


def assign(pred, t, col="prob"):
    """pred: (rid, s1, prob). Each S2/S3 record goes to its single best S1 if prob >= t."""
    best = pred.sort(col, descending=True).unique("rid", keep="first")
    return best.filter(pl.col(col) >= t).select("rid", "s1")


def macro_f05(assigned, gt, s1_ids):
    """assigned: (rid, s1) predicted links; gt: (s1, rid) true links; s1_ids: Series of S1 entities evaluated."""
    base = pl.DataFrame({"s1": s1_ids})
    a = assigned.join(base, on="s1", how="semi")
    g = gt.join(base, on="s1", how="semi")
    tp = a.join(g, on=["s1", "rid"], how="semi").group_by("s1").len().rename({"len": "tp"})
    npred = a.group_by("s1").len().rename({"len": "np"})
    ntrue = g.group_by("s1").len().rename({"len": "nt"})
    d = base.join(tp, on="s1", how="left").join(npred, on="s1", how="left").join(ntrue, on="s1", how="left").fill_null(0)
    p = d["tp"] / d["np"].clip(1); r = d["tp"] / d["nt"].clip(1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0)
    f = pl.select(pl.when((d["np"] == 0) & (d["nt"] == 0)).then(1.0).otherwise(f)).to_series()
    out = d.with_columns(f.alias("f"))
    return float(out["f"].mean()), out


def sweep(pred, gt, s1_ids, ts=np.arange(0.05, 0.96, 0.05), col="prob"):
    res = []
    for t in ts:
        f, _ = macro_f05(assign(pred, t, col), gt, s1_ids)
        res.append((round(float(t), 3), round(f, 5)))
    return res


def macro_f05_testlike(assigned, gt, s1_ids, dup=2):
    """Macro F0.5 where every link made by a query WITHOUT a true match is counted `dup` times.
    Train has ~26% such decoy records, test ~40%; dup=2 gives ~41% -> a test-like estimate."""
    decoy = assigned.join(gt.select("rid"), on="rid", how="anti")
    extra = pl.concat([decoy.with_columns((pl.col("rid") + f"#dup{k}").alias("rid")) for k in range(1, dup)]) if dup > 1 else decoy.head(0)
    return macro_f05(pl.concat([assigned, extra]), gt, s1_ids)
