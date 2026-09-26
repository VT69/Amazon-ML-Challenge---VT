"""Peek at France records in test (no labels). Find plausible clusters via rare name tokens."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
fr = {s: load("test", f"s{s}").filter(pl.col("country") == "France") for s in [1, 2, 3]}
for s in [1, 2, 3]:
    print(f"\n== France S{s} random 15")
    for r in fr[s].sample(15, seed=3).iter_rows(): print("  ", r[1], "|", r[2])
# crude cluster peek: take S1 France, find S2/S3 France records sharing the 2 rarest name tokens (lowercased)
s1 = fr[1].sample(20, seed=5)
tok = lambda df: df.with_columns(pl.col("business_name").str.to_lowercase().str.extract_all(r"[a-zà-ÿ]{4,}").alias("t"))
r = tok(pl.concat([fr[2], fr[3]])).explode("t")
df = r.group_by("t").len()
for row in tok(s1).iter_rows(named=True):
    ts = [t for t in (row["t"] or [])]
    fq = {t: (df.filter(pl.col("t") == t)["len"].to_list() or [0])[0] for t in ts}
    ts = sorted(ts, key=lambda t: fq[t])[:2]
    if not ts: continue
    cand = r.filter(pl.col("t").is_in(ts)).group_by("entity_id").agg(pl.col("t").n_unique().alias("k"), pl.first("business_name"), pl.first("business_address")).filter(pl.col("k") == len(ts))
    # also require first address number overlap
    num = (row["business_address"].split(" ")[0])
    cand = cand.filter(pl.col("business_address").fill_null("").str.contains(num, literal=True) | pl.col("business_address").is_null())
    print(f"\n### {row['business_name']} | {row['business_address']}   rare={ts}")
    for c in cand.head(8).iter_rows(): print("     ", c[0], "|", c[2], "|", c[3])
