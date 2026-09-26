"""Shared helpers for the EDA scripts. Not the final pipeline."""
import os, re, unicodedata
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "student_resource", "dataset")
PQ = os.path.join(ROOT, "eda", "data")
FIGS = os.path.join(ROOT, "eda", "figs")
os.environ.setdefault("TMP", os.path.join(ROOT, ".tmp"))

def read_tsv(path):
    # quote_char=None: names contain quotes; tabs are the only delimiter
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=False)

def load(split, which):
    return pl.read_parquet(os.path.join(PQ, f"{split}_{which}.parquet"))

def gt_pairs():
    return pl.read_parquet(os.path.join(PQ, "train_pairs.parquet"))
