"""Step 4/5/8: similarity distributions on true pairs vs random & same-block negatives."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm import *
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

rng = np.random.default_rng(0)
s1 = load("train", "s1"); r = pl.concat([load("train", "s2"), load("train", "s3")])
pairs = gt_pairs().sample(200_000, seed=1)
def prep(df, p):
    return df.select(pl.col("entity_id").alias(p + "id"), pl.col("business_name").alias(p + "name"),
                     pl.col("business_address").alias(p + "addr"), pl.col("country").alias(p + "c"))
pos = (pairs.join(prep(s1, "a_"), left_on="s1", right_on="a_id").join(prep(r, "b_"), left_on="rid", right_on="b_id")
       .with_columns(pl.lit(1).alias("y")))
# random negatives, same country
neg = []
for c in ["US", "India"]:
    a = s1.filter(pl.col("country") == c).sample(50_000, seed=2)
    b = r.filter(pl.col("country") == c).sample(50_000, seed=3)
    neg.append(pl.concat([prep(a, "a_").with_columns(s1=pl.col("a_id")), prep(b, "b_").with_columns(rid=pl.col("b_id"))], how="horizontal"))
neg = pl.concat(neg).with_columns(pl.col("rid").str.slice(0, 2).alias("src"), pl.lit(0).alias("y"))
# drop accidental positives
neg = neg.join(gt_pairs().select("s1", "rid", pl.lit(True).alias("hit")), on=["s1", "rid"], how="left").filter(pl.col("hit").is_null()).drop("hit")
cols = ["s1", "rid", "src", "a_name", "a_addr", "a_c", "b_name", "b_addr", "b_c", "y"]
df = pl.concat([pos.select(cols), neg.select(cols)])
df = df.with_columns(basic(pl.col("a_name")).alias("an"), basic(pl.col("b_name")).alias("bn"),
                     basic(pl.col("a_addr")).alias("aa0"), basic(pl.col("b_addr")).alias("ba0"))
df = df.with_columns(name_core(pl.col("an")).alias("anc"), name_core(pl.col("bn")).alias("bnc"),
                     addr_norm(pl.col("aa0")).alias("aa"), addr_norm(pl.col("ba0")).alias("ba"))
df.write_parquet(f"{PQ}/eda_pairs.parquet")

def jacc(x, y):
    X, Y = set(x.split()), set(y.split())
    return len(X & Y) / len(X | Y) if X | Y else 0.0
def ngram(s, n=3):
    s = f" {s} "; return {s[i:i + n] for i in range(len(s) - n + 1)}
def ngj(x, y):
    X, Y = ngram(x), ngram(y); return len(X & Y) / len(X | Y) if X | Y else 0.0
feats = {k: [] for k in ["raw_eq", "fold_eq", "norm_eq", "core_eq", "tokset_eq", "n_jacc", "n_core_jacc", "n_3g", "n_lev", "n_jw",
                         "n_tsr", "n_pr", "a_jacc", "a_3g", "a_tsr", "a_num_eq", "a_first_num_eq", "b_addr_missing"]}
import re
num_re = re.compile(r"\d+")
for row in df.select("a_name", "b_name", "an", "bn", "anc", "bnc", "aa", "ba", "b_addr").iter_rows():
    a_name, b_name, an, bn, anc, bnc, aa, ba, braw = row
    feats["raw_eq"].append(a_name == b_name); feats["fold_eq"].append(a_name.lower() == b_name.lower())
    feats["norm_eq"].append(an == bn); feats["core_eq"].append(anc == bnc and anc != "")
    feats["tokset_eq"].append(set(an.split()) == set(bn.split()))
    feats["n_jacc"].append(jacc(an, bn)); feats["n_core_jacc"].append(jacc(anc, bnc)); feats["n_3g"].append(ngj(an, bn))
    feats["n_lev"].append(Levenshtein.normalized_similarity(an, bn)); feats["n_jw"].append(JaroWinkler.similarity(an, bn))
    feats["n_tsr"].append(fuzz.token_set_ratio(an, bn) / 100); feats["n_pr"].append(fuzz.partial_ratio(an, bn) / 100)
    feats["a_jacc"].append(jacc(aa, ba)); feats["a_3g"].append(ngj(aa, ba)); feats["a_tsr"].append(fuzz.token_set_ratio(aa, ba) / 100)
    na, nb = set(num_re.findall(aa)), set(num_re.findall(ba))
    feats["a_num_eq"].append(len(na & nb) / len(na | nb) if na | nb else np.nan)
    fa, fb = num_re.findall(aa), num_re.findall(ba)
    feats["a_first_num_eq"].append((fa[0] == fb[0]) if fa and fb else np.nan)
    feats["b_addr_missing"].append(braw is None)
df = df.with_columns(**{k: pl.Series(v, strict=False) for k, v in feats.items()})
df.write_parquet(f"{PQ}/eda_pairs.parquet")
