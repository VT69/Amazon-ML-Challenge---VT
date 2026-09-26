import os, sys, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from common import *
for split in ["train", "test"]:
    for s in [1, 2, 3]:
        p = os.path.join(RAW, split, f"{split}_source{s}.tsv")
        df = read_tsv(p)
        n_lines = sum(1 for _ in open(p, "rb")) - 1
        print(split, s, df.shape, "lines:", n_lines, "cols:", df.columns)
        assert df.height == n_lines
        df.write_parquet(os.path.join(PQ, f"{split}_s{s}.parquet"))
gt = read_tsv(os.path.join(RAW, "train", "train_ground_truth.tsv"))
print("gt", gt.shape, gt.columns)
gt.write_parquet(os.path.join(PQ, "train_gt.parquet"))
pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
pairs = pairs.with_columns(pl.col("rid").str.slice(0, 2).alias("src"))
print("pairs", pairs.shape, pairs["src"].value_counts())
pairs.write_parquet(os.path.join(PQ, "train_pairs.parquet"))
