"""Normalise every record (v2) for a split; one (source, country) chunk at a time -> eda/data/{split}_v2/{src}_{country}.parquet"""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm2 import addr_v2, name_v2, name_core_v2, skeleton
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None     # restrict to some countries
out = f"{PQ}/{split}_v2"; os.makedirs(out, exist_ok=True)
for s in ["s1", "s2", "s3"]:
    lf = pl.scan_parquet(f"{PQ}/{split}_{s}.parquet")
    for country in lf.select("country").unique().collect()["country"].to_list():
        if ONLY and country not in ONLY: continue
        t0 = time.time()
        d = lf.filter(pl.col("country") == country).collect()
        d = d.with_columns(name_v2("business_name").alias("nb"), addr_v2("business_address", country).alias("ab"))
        d = d.with_columns(name_core_v2(pl.col("nb")).alias("nc"))
        d = d.with_columns(pl.Series("sk", skeleton(d["nc"].to_list())))
        d.select("entity_id", "country", "nb", "nc", "ab", "sk").write_parquet(f"{out}/{s}_{country}.parquet")
        print(split, s, country, d.height, round(time.time() - t0), "s", flush=True)
