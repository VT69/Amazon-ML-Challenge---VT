"""Group-consistency features (label-free, identical for train / test / unseen countries).

For every S1, its "group" = the S2/S3 records (whole split, incl. synthetic ones in train) whose top-1 retrieval
candidate (p_rank == 0) is that S1. For a pair (query q, candidate s1):
  g_n          other records in s1's group
  g_num_twin   others in the group with exactly q's number set        (rival-cluster evidence when != s1's numbers)
  g_num_s1     others in the group whose numbers equal s1's numbers   (support for the S1's own version)
  g_nc_twin    others with exactly q's name core
  g_nc_s1      others whose name core equals s1's name core
  q_num_eq_s1  q's number set == s1's number set (null if either side has none)
  g_rival      q disagrees with s1's numbers but has a twin that agrees with q (a self-consistent rival cluster)
Test decoys are multi-record look-alike businesses; these features let the model compare clusters, not just pairs."""
import os, glob
import polars as pl
from common import PQ

R_SRCS = ["s2", "s3", "s2syn", "s3syn"]
nums = pl.col("ab").fill_null("").str.extract_all(r"\b\d+\b").list.sort().list.join(" ").alias("num")


def group_table(split, country):
    top = pl.concat([pl.read_parquet(f, columns=["rid", "s1", "p_rank"]).filter(pl.col("p_rank") == 0).select("rid", "s1")
                     for f in sorted(glob.glob(f"{PQ}/{split}_cand/{country}_*.parquet"))])
    A = pl.concat([pl.read_parquet(f"{PQ}/{split}_v2/{s}_{country}.parquet", columns=["entity_id", "nc", "ab"]).select("entity_id", "nc", nums)
                   for s in R_SRCS if os.path.exists(f"{PQ}/{split}_v2/{s}_{country}.parquet")])
    S = pl.read_parquet(f"{PQ}/{split}_v2/s1_{country}.parquet", columns=["entity_id", "nc", "ab"]).select("entity_id", "nc", nums)
    top = top.join(A.rename({"entity_id": "rid"}), on="rid", how="left")
    return dict(
        A=A, S=S,
        top=top.select("rid", pl.col("s1").alias("top_s1")),
        n=top.group_by("s1").len().rename({"len": "cnt_s"}),
        sn=top.group_by("s1", "num").len().rename({"len": "cnt_sn"}),
        sc=top.group_by("s1", "nc").len().rename({"len": "cnt_sc"}),
    )


def add_group(F, G):
    """F: pair frame with rid, s1 -> adds g_* columns."""
    P = F.select("rid", "s1").with_row_index("_i")
    P = (P.join(G["A"].rename({"entity_id": "rid", "nc": "q_nc", "num": "q_num"}), on="rid", how="left")
          .join(G["S"].rename({"entity_id": "s1", "nc": "s_nc", "num": "s_num"}), on="s1", how="left")
          .join(G["top"], on="rid", how="left")
          .join(G["n"], on="s1", how="left")
          .join(G["sn"].rename({"num": "q_num", "cnt_sn": "cnt_qnum"}), on=["s1", "q_num"], how="left")
          .join(G["sn"].rename({"num": "s_num", "cnt_sn": "cnt_snum"}), on=["s1", "s_num"], how="left")
          .join(G["sc"].rename({"nc": "q_nc", "cnt_sc": "cnt_qnc"}), on=["s1", "q_nc"], how="left")
          .join(G["sc"].rename({"nc": "s_nc", "cnt_sc": "cnt_snc"}), on=["s1", "s_nc"], how="left"))
    ing = (pl.col("top_s1") == pl.col("s1")).fill_null(False).cast(pl.Int32)
    qn_ok = pl.col("q_num").fill_null("") != ""
    sn_ok = pl.col("s_num").fill_null("") != ""
    eqn = pl.col("q_num") == pl.col("s_num")
    P = P.with_columns(
        (pl.col("cnt_s").fill_null(0) - ing).alias("g_n"),
        pl.when(qn_ok).then(pl.col("cnt_qnum").fill_null(0) - ing).otherwise(None).alias("g_num_twin"),
        pl.when(sn_ok).then(pl.col("cnt_snum").fill_null(0) - (ing * eqn.fill_null(False).cast(pl.Int32))).otherwise(None).alias("g_num_s1"),
        (pl.col("cnt_qnc").fill_null(0) - ing).alias("g_nc_twin"),
        (pl.col("cnt_snc").fill_null(0) - (ing * (pl.col("q_nc") == pl.col("s_nc")).fill_null(False).cast(pl.Int32))).alias("g_nc_s1"),
        pl.when(qn_ok & sn_ok).then(eqn).otherwise(None).cast(pl.Float32).alias("q_num_eq_s1"),
    ).with_columns(
        pl.when(pl.col("q_num_eq_s1") == 0).then((pl.col("g_num_twin") > 0).cast(pl.Float32)).otherwise(0.0).alias("g_rival"),
        (pl.col("g_num_twin") / (pl.col("g_n") + 1)).cast(pl.Float32).alias("g_num_twin_frac"),
        (pl.col("g_num_s1") / (pl.col("g_n") + 1)).cast(pl.Float32).alias("g_num_s1_frac"),
    )
    cols = ["g_n", "g_num_twin", "g_num_s1", "g_nc_twin", "g_nc_s1", "q_num_eq_s1", "g_rival", "g_num_twin_frac", "g_num_s1_frac"]
    P = P.sort("_i").select([pl.col(c).cast(pl.Float32) for c in cols])
    return F.with_columns(P)
