"""Number-relation features for each test query's BEST link only (~10M pairs, not all ~220M candidates).
Enables decision rules such as 'drop links whose house number was changed by a little' without re-scoring.
python 21_link_numfeats.py <split> <tag>  ->  data/linkfeat_<split>_<tag>/<chunk>.parquet"""
import sys, os, glob, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from feats import _parse_nums, _num_feats
from retr import to_buf
import numpy as np
split, tag = sys.argv[1], sys.argv[2]
out = f"{PQ}/linkfeat_{split}_{tag}"; os.makedirs(out, exist_ok=True)
nums = lambda df: df.select("entity_id", pl.col("ab").str.extract_all(r"\b\d+\b").list.join(" ").alias("num"))
files = sorted(glob.glob(f"{PQ}/pred_{split}_{tag}/*.parquet"))
for country in sorted({os.path.basename(f).split("_")[0] for f in files}):
    S = nums(pl.read_parquet(f"{PQ}/{split}_v2/s1_{country}.parquet", columns=["entity_id", "ab"]))
    Srow = S.with_row_index("sr").select("entity_id", "sr"); Sn = _parse_nums(*to_buf(S["num"].fill_null("").to_list()))
    R = pl.concat([nums(pl.read_parquet(f"{PQ}/{split}_v2/{s}_{country}.parquet", columns=["entity_id", "ab"]))
                   for s in ["s2", "s3"] if os.path.exists(f"{PQ}/{split}_v2/{s}_{country}.parquet")])
    Rrow = R.with_row_index("qr").select("entity_id", "qr"); Rn = _parse_nums(*to_buf(R["num"].fill_null("").to_list()))
    for f in [f for f in files if os.path.basename(f).startswith(country + "_")]:
        t0 = time.time()
        B = pl.read_parquet(f, columns=["rid", "s1", "prob"]).sort("prob", descending=True).unique("rid", keep="first")
        B = B.join(Rrow.rename({"entity_id": "rid"}), on="rid").join(Srow.rename({"entity_id": "s1"}), on="s1")
        nf = _num_feats(*Rn, *Sn, B["qr"].to_numpy().astype(np.int64), B["sr"].to_numpy().astype(np.int64))
        cols = ["nm_trunc", "nm_near", "nm_changed", "nm_added", "nm_miss", "nm_mindiff", "nm_first_eq", "nm_max_eq"]
        B = B.drop("qr", "sr").with_columns([pl.Series(c, nf[:, j]) for j, c in enumerate(cols)])
        B.write_parquet(f"{out}/{os.path.basename(f)}")
        print(os.path.basename(f), B.height, f"{time.time() - t0:.0f}s", flush=True)
print("done")
