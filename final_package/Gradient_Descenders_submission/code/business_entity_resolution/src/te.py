"""Out-of-fold name-token log-odds features (table built by 16_te.py from training folds 6..39)."""
import polars as pl
from common import PQ

_names = {}


def name_table(split, country):
    key = (split, country)
    if key not in _names:
        import os
        _names[key] = pl.concat([pl.read_parquet(f"{PQ}/{split}_v2/{s}_{country}.parquet", columns=["entity_id", "nc"])
                                 for s in ["s1", "s2", "s3", "s2syn", "s3syn"] if os.path.exists(f"{PQ}/{split}_v2/{s}_{country}.parquet")])
    return _names[key]


def diff_tokens(P, split, country):
    """P: (rid, s1, ...) -> adds list columns extra (query-only name tokens) and miss (S1-only name tokens)."""
    N = name_table(split, country)
    P = P.join(N.rename({"entity_id": "rid", "nc": "qn"}), on="rid", how="left").join(N.rename({"entity_id": "s1", "nc": "sn"}), on="s1", how="left")
    qt = pl.col("qn").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    st = pl.col("sn").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    return P.with_columns(qt.list.set_difference(st).alias("extra"), st.list.set_difference(qt).alias("miss")).drop("qn", "sn")


def te_features(P, split, country, T):
    """P: (rid, s1). Returns (rid, s1, te_extra_min/sum/known, te_miss_min/sum/known). Unknown tokens -> neutral."""
    P = diff_tokens(P.select("rid", "s1").with_row_index("i"), split, country)
    agg = []
    for side in ["extra", "miss"]:
        e = P.select("i", side).explode(side).drop_nulls(side).join(T.select(pl.col("tok").alias(side), f"lo_{side}"), on=side, how="left")
        agg.append(e.group_by("i").agg(pl.col(f"lo_{side}").min().alias(f"te_{side}_min"), pl.col(f"lo_{side}").sum().alias(f"te_{side}_sum"),
                                       pl.col(f"lo_{side}").is_not_null().sum().cast(pl.Int16).alias(f"te_{side}_known")))
    return P.select("i", "rid", "s1").join(agg[0], on="i", how="left").join(agg[1], on="i", how="left").drop("i")
