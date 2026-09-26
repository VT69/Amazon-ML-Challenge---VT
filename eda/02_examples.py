"""Dump raw example clusters (S1 + all matched S2/S3) for eyeballing."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
s1 = load("train", "s1"); pairs = gt_pairs()
r = pl.concat([load("train", "s2"), load("train", "s3")])
cnt = pl.read_parquet(f"{PQ}/train_cnt.parquet")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
country = sys.argv[2] if len(sys.argv) > 2 else None
k = int(sys.argv[3]) if len(sys.argv) > 3 else 12
pool = cnt.filter(pl.col("n") > 0) if country is None else cnt.filter((pl.col("n") > 0) & (pl.col("country") == country))
ids = pool.sample(k, seed=seed)["s1"]
sub = pairs.filter(pl.col("s1").is_in(ids.implode())).join(r.rename({"entity_id": "rid"}), on="rid")
for i in ids:
    a = s1.filter(pl.col("entity_id") == i).row(0)
    print(f"\n### {a[0]} | {a[1]} | {a[2]} | {a[3]}")
    for row in sub.filter(pl.col("s1") == i).sort("rid").iter_rows():
        print(f"   {row[1]:<14}| {row[3]} | {row[4]}")
