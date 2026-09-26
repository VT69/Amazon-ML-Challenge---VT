"""Step 1-2: dataset overview + ground-truth cardinality."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(200)

rows = []
for split in ["train", "test"]:
    for s in [1, 2, 3]:
        df = load(split, f"s{s}")
        n = df.height
        r = dict(split=split, src=f"S{s}", rows=n,
                 id_unique=df["entity_id"].n_unique() == n,
                 name_null=df["business_name"].is_null().mean(),
                 addr_null=df["business_address"].is_null().mean(),
                 country_null=df["country"].is_null().mean(),
                 name_uniq=df["business_name"].n_unique() / n,
                 addr_uniq=df["business_address"].n_unique() / n,
                 name_addr_dup=1 - df.select(["business_name", "business_address"]).n_unique() / n,
                 name_len_med=df["business_name"].str.len_chars().median(),
                 addr_len_med=df["business_address"].str.len_chars().median(),
                 name_tok_mean=df["business_name"].str.split(" ").list.len().mean(),
                 addr_tok_mean=df["business_address"].str.split(" ").list.len().mean())
        rows.append(r)
        print(split, s, df["country"].value_counts(sort=True).with_columns((pl.col("count") / n).round(4).alias("frac")).rows())
ov = pl.DataFrame(rows)
print(ov)

# ---------- ground truth ----------
gt = load("train", "gt"); pairs = gt_pairs()
s1 = load("train", "s1")
print("gt rows", gt.height, "s1 rows", s1.height, "gt ids == s1 ids:",
      set(gt["source1_entity_id"].to_list()) == set(s1["entity_id"].to_list()))
cnt = (s1.select(pl.col("entity_id").alias("s1"), "country")
       .join(pairs.group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3")), on="s1", how="left")
       .fill_null(0).with_columns((pl.col("n2") + pl.col("n3")).alias("n")))
cnt.write_parquet(f"{PQ}/train_cnt.parquet")
N = cnt.height
print("singleton %", (cnt["n"] == 0).mean(), " only S2", ((cnt["n2"] > 0) & (cnt["n3"] == 0)).mean(),
      " only S3", ((cnt["n3"] > 0) & (cnt["n2"] == 0)).mean(), " both", ((cnt["n3"] > 0) & (cnt["n2"] > 0)).mean())
print("max n", cnt["n"].max(), "max n2", cnt["n2"].max(), "max n3", cnt["n3"].max(), "mean n (non-singleton)", cnt.filter(pl.col("n") > 0)["n"].mean())
for c in ["n", "n2", "n3"]:
    print(c, cnt[c].value_counts().sort(c).with_columns((pl.col("count") / N).round(4).alias("frac")).rows())
print(cnt.group_by("country").agg(pl.len(), (pl.col("n") == 0).mean().alias("singleton"), pl.col("n").mean().alias("mean_n"),
      pl.col("n2").mean(), pl.col("n3").mean()))
print("n2 x n3 crosstab (cap 5)")
print(cnt.with_columns(pl.col("n2").clip(0, 5), pl.col("n3").clip(0, 5)).group_by("n2", "n3").len().sort("n2", "n3")
      .pivot(on="n3", index="n2", values="len"))

# is each S2/S3 record matched to at most one S1? coverage of S2/S3
rid_counts = pairs.group_by("rid").len()
print("S2/S3 ids appearing >1 times in GT:", (rid_counts["len"] > 1).sum())
for s in [2, 3]:
    d = load("train", f"s{s}")
    matched = d["entity_id"].is_in(pairs["rid"].implode())
    print(f"S{s}: {matched.mean():.4f} of records are matched to some S1; unmatched = {(~matched).sum()}")
    print("   by country:", d.with_columns(matched.alias("m")).group_by("country").agg(pl.len(), pl.col("m").mean()).rows())
# ids in GT that do not exist in sources
all_ids = pl.concat([load("train", "s2")["entity_id"], load("train", "s3")["entity_id"]])
print("GT ids missing from sources:", (~pairs["rid"].is_in(all_ids.implode())).sum())
# country consistency between S1 and matched records
c23 = pl.concat([load("train", f"s{s}").select("entity_id", "country") for s in [2, 3]]).rename({"entity_id": "rid", "country": "c_r"})
pc = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1")), on="s1").join(c23, on="rid")
print("country agreement in true pairs:", (pc["c1"] == pc["c_r"]).mean())
print(pc.filter(pl.col("c1") != pl.col("c_r")).group_by("c1", "c_r").len())
