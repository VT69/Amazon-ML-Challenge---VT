"""Shared paths and loaders.

Environment variables (all optional):
  ER_DATA  folder containing train/ and test/ with the official .tsv files   (default: ../dataset)
  ER_WORK  working folder for intermediate parquet files, models, predictions (default: ../work)
  ER_OUT   folder for the final matching_results.tsv / candidate_pairs.tsv   (default: ../output)
"""
import os, re, unicodedata
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("ER_DATA", os.path.join(ROOT, "dataset"))
PQ = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))
OUT = os.environ.get("ER_OUT", os.path.join(ROOT, "output"))
os.makedirs(PQ, exist_ok=True)
os.environ.setdefault("TMP", os.path.join(PQ, "tmp"))


def read_tsv(path):
    # quote_char=None: names contain quotes; tabs are the only delimiter
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=False)


def load(split, which):
    return pl.read_parquet(os.path.join(PQ, f"{split}_{which}.parquet"))


def gt_pairs():
    return pl.read_parquet(os.path.join(PQ, "train_pairs.parquet"))
