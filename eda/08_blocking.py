"""Step 9: key-based blocking simulation on full train pools."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
K = pl.read_parquet(f"{PQ}/train_keys.parquet")
pairs = gt_pairs().select("s1", "rid", "src")
ntdf = pl.read_parquet(f"{PQ}/train_ntok_df.parquet"); atdf = pl.read_parquet(f"{PQ}/train_atok_df.parquet")
DOM = r"\b(com|www|net|org|in|co)\b"
K = K.with_columns(pl.col("nc").str.replace_all(DOM, "").str.replace_all(r"[^a-z0-9]", "").alias("n_nospace"))

def rare_tokens(col, dfT, k, min_df=2, alpha_only=False):
    t = K.select("entity_id", "country", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 2)
    if alpha_only: t = t.filter(pl.col("t").str.contains(r"^[a-z]+$"))
    t = t.join(dfT, on=["country", "t"]).filter(pl.col("df") >= min_df)
    t = t.sort("df", "t").group_by("entity_id", maintain_order=True).head(k)
    return t.select("entity_id", "country", "t")

blockers = {}
def single(name, col):
    blockers[name] = K.select("entity_id", "country", pl.col(col).alias("key")).filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
single("n_exact", "nb"); single("n_core", "nc"); single("n_sorted", "n_sorted"); single("n_nospace", "n_nospace")
single("a_exact", "ab"); single("a_numstreet", "a_numstreet"); single("a_num", "a_num")
blockers["n_rare1"] = rare_tokens("nc", ntdf, 1).rename({"t": "key"})
blockers["n_rare2"] = rare_tokens("nc", ntdf, 2).rename({"t": "key"})
blockers["a_rare1"] = rare_tokens("ab", atdf, 1, alpha_only=True).rename({"t": "key"})
blockers["a_rare2"] = rare_tokens("ab", atdf, 2, alpha_only=True).rename({"t": "key"})
nr1 = blockers["n_rare1"]; ar1 = blockers["a_rare1"]; ar2 = blockers["a_rare2"]; nr2 = blockers["n_rare2"]
num = blockers["a_num"]
blockers["num+n_rare2"] = num.join(nr2, on=["entity_id", "country"]).select("entity_id", "country", (pl.col("key") + "|" + pl.col("key_right")).alias("key"))
blockers["num+a_rare1"] = num.join(ar1, on=["entity_id", "country"]).select("entity_id", "country", (pl.col("key") + "|" + pl.col("key_right")).alias("key"))
blockers["n_rare2+a_rare2"] = nr2.join(ar2, on=["entity_id", "country"]).select("entity_id", "country", (pl.col("key") + "|" + pl.col("key_right")).alias("key"))

src_of = pl.col("entity_id").str.slice(0, 2)
rng_s1 = K.filter(pl.col("src") == "S1").sample(20_000, seed=7).select("entity_id")
n_r_country = K.filter(pl.col("src") != "S1").group_by("country").len().rename({"len": "nR"})
n_s1 = K.filter(pl.col("src") == "S1").height
res, hits = [], pairs.select("s1", "rid")
for name, b in blockers.items():
    t0 = time.time()
    b = b.with_columns(pl.col("key").cast(pl.String))
    bs1 = b.filter(src_of == "S1"); br = b.filter(src_of != "S1")
    # recall
    ph = (pairs.join(bs1.select(pl.col("entity_id").alias("s1"), "key"), on="s1")
          .join(br.select(pl.col("entity_id").alias("rid"), "key"), on=["rid", "key"], how="semi")
          .select("s1", "rid").unique().with_columns(pl.lit(True).alias(name)))
    hits = hits.join(ph, on=["s1", "rid"], how="left").with_columns(pl.col(name).fill_null(False))
    # block-size based candidate upper bound per S1 (sum of block sizes)
    bsz = br.group_by("country", "key").len().rename({"len": "bs"})
    cb = bs1.join(bsz, on=["country", "key"], how="left").fill_null(0).group_by("entity_id").agg(pl.col("bs").sum())
    cb = K.filter(pl.col("src") == "S1").select("entity_id").join(cb, on="entity_id", how="left").fill_null(0)["bs"]
    # exact distinct candidates on a 20k sample
    samp = bs1.join(rng_s1, on="entity_id", how="semi").join(br.rename({"entity_id": "rid"}), on=["country", "key"]).select("entity_id", "rid").unique()
    ex = rng_s1.join(samp.group_by("entity_id").len(), on="entity_id", how="left").fill_null(0)["len"]
    total_pairs = (K.filter(pl.col("src") == "S1").group_by("country").len().join(n_r_country, on="country").select((pl.col("len") * pl.col("nR")).sum()).item())
    res.append(dict(blocker=name, recall=hits[name].mean(), recall_S2=hits.filter(pl.col("rid").str.starts_with("S2"))[name].mean(),
                    recall_S3=hits.filter(pl.col("rid").str.starts_with("S3"))[name].mean(),
                    cand_mean=ex.mean(), cand_med=ex.median(), cand_p95=ex.quantile(.95), cand_max=ex.max(),
                    ub_mean=cb.mean(), ub_p95=cb.quantile(.95), ub_max=cb.max(),
                    reduction=1 - cb.sum() / total_pairs, secs=round(time.time() - t0)))
    print(res[-1], flush=True)
    samp.write_parquet(f"{PQ}/blk_samp_{name.replace('+','_')}.parquet")
hits.write_parquet(f"{PQ}/blk_hits.parquet")
R = pl.DataFrame(res); R.write_csv("eda/out/08_blocking.csv"); print(R)
