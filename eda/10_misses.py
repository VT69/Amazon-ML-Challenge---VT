"""Inspect retrieval misses of the best retriever (NA_w|20000|4)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
rk = pl.read_parquet(f"{PQ}/retr_ranks.parquet").filter(pl.col("method") == "NA_w|20000|4", pl.col("s1").is_not_null())
raw = pl.concat([load("train", s).select("entity_id", "business_name", "business_address") for s in ["s1", "s2", "s3"]])
K = pl.scan_parquet(f"{PQ}/train_keys.parquet").select("entity_id", "nc", "ab").collect()
m = rk.join(raw.rename({"business_name": "qn", "business_address": "qa"}), left_on="qid", right_on="entity_id") \
      .join(raw.rename({"business_name": "sn", "business_address": "sa"}), left_on="s1", right_on="entity_id") \
      .join(K.rename({"nc": "q_nc", "ab": "q_ab"}), left_on="qid", right_on="entity_id") \
      .join(K.rename({"nc": "s_nc", "ab": "s_ab"}), left_on="s1", right_on="entity_id")
m = m.with_columns((pl.col("rank") >= 50).alias("miss"), pl.col("qn").str.contains(r"[^\x00-\x7F]").alias("q_nonascii"),
                   pl.col("qa").is_null().alias("q_noaddr"), pl.col("qid").str.slice(0, 2).alias("src"))
print(m.group_by("country", "src").agg(pl.col("miss").mean().round(4), pl.len()).sort("country", "src"))
print(m.group_by("country", "q_nonascii", "q_noaddr").agg(pl.col("miss").mean().round(4), pl.len(), pl.col("miss").sum().alias("n_miss")).sort("country", "q_nonascii", "q_noaddr"))
for c in ["India", "US"]:
    x = m.filter(pl.col("miss"), pl.col("country") == c).sample(25, seed=1)
    print(f"\n==== {c} misses")
    for r in x.iter_rows(named=True):
        print(f"Q  {r['qn']!s:45.45} | {r['qa']!s:110.110}\n S1 {r['sn']!s:45.45} | {r['sa']!s:110.110}\n   q_ab={r['q_ab']!s:80.80}\n   s_ab={r['s_ab']!s:80.80}")
