"""Cluster-level decision rules on top of per-pair probabilities (label-free; use the test set's own structure).

A record's "cluster" = records whose best S1 is the same and whose address numbers are exactly the same.
  kill  : drop links in a cluster whose numbers differ from the S1's when any member was confidently rejected
          (prob < lo) -> test decoy businesses come as multi-record clusters with one shared changed number
  boost : add a borderline link (prob >= add_min) when its cluster's numbers equal the S1's and a member was
          confidently accepted (prob >= add_hi)
  ambig : for the listed countries, skip a link when the runner-up S1 is also plausible (p2 >= amb) —
          France has 10x more such ties than US/India/validation
Only rules that fire far more on test than on validation are used (that excess is where test-only errors live)."""
import glob, os
import polars as pl
from common import PQ

nums = lambda df: df.with_columns(pl.col("ab").str.extract_all(r"\b\d+\b").list.sort().list.join(" ").alias("num")).select("entity_id", "num")
RULESETS = {
    "cluster_mod":  dict(lo=0.05, add_hi=0.95, add_min=0.3, amb=0.3, amb_countries=("France",)),
    "cluster_bold": dict(lo=0.2,  add_hi=0.9,  add_min=0.2, amb=0.1, amb_countries=("France",)),
}


def best_with_runnerup(P):
    return (P.sort("prob", descending=True).group_by("rid", maintain_order=True)
             .agg(pl.col("s1").first(), pl.col("prob").first(), pl.col("prob").get(1, null_on_oob=True).fill_null(0).alias("p2")))


def test_frame(tag, split="test"):
    """Best link + runner-up prob + number sets for every record of the split (per country, streamed)."""
    out = []
    files = glob.glob(f"{PQ}/pred_{split}_{tag}/*.parquet")
    for c in sorted({os.path.basename(f).split("_")[0] for f in files}):
        B = pl.concat([best_with_runnerup(pl.read_parquet(f, columns=["rid", "s1", "prob"])) for f in files if os.path.basename(f).startswith(c + "_")])
        A = pl.concat([nums(pl.read_parquet(f"{PQ}/{split}_v2/{s}_{c}.parquet", columns=["entity_id", "ab"])) for s in ["s2", "s3"]
                       if os.path.exists(f"{PQ}/{split}_v2/{s}_{c}.parquet")])
        S = nums(pl.read_parquet(f"{PQ}/{split}_v2/s1_{c}.parquet", columns=["entity_id", "ab"]))
        out.append(B.join(A.rename({"entity_id": "rid"}), on="rid", how="left").join(S.rename({"entity_id": "s1", "num": "s_num"}), on="s1", how="left")
                    .with_columns(pl.lit(c).alias("country")))
    return pl.concat(out)


def decide(B, thr_expr, lo, add_hi, add_min, amb, amb_countries):
    B = B.with_columns(pl.col("num").fill_null(""), pl.col("s_num").fill_null(""))
    B = B.with_columns(((pl.col("num") != "") & (pl.col("num") == pl.col("s_num"))).alias("agree"), (pl.col("num") != "").alias("hasnum"))
    k = ["s1", "num"]
    B = B.with_columns(pl.col("prob").min().over(k).alias("cmin"), pl.col("prob").max().over(k).alias("cmax"), pl.len().over(k).alias("csize"))
    base = pl.col("prob") >= thr_expr
    kill = pl.col("hasnum") & ~pl.col("agree") & (pl.col("csize") >= 2) & (pl.col("cmin") < lo)
    ambig = pl.col("country").is_in(list(amb_countries)) & (pl.col("p2") >= amb)
    boost = pl.col("agree") & (pl.col("csize") >= 2) & (pl.col("cmax") >= add_hi) & (pl.col("prob") >= add_min) & ~base & ~ambig
    sel = (base & ~kill & ~ambig) | boost
    stats = {n: B.filter(base & e).height for n, e in [("kill", kill), ("ambig", ambig)]} | {"boost": B.filter(boost).height}
    return B.filter(sel).select("rid", "s1"), stats
