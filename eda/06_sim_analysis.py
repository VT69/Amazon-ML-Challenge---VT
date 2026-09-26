import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from collections import Counter
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70)
df = pl.read_parquet(f"{PQ}/eda_pairs.parquet")
num = ["raw_eq", "fold_eq", "norm_eq", "core_eq", "tokset_eq", "n_jacc", "n_core_jacc", "n_3g", "n_lev", "n_jw", "n_tsr", "n_pr", "a_jacc", "a_3g", "a_tsr", "a_num_eq", "a_first_num_eq", "b_addr_missing"]
print("== mean of similarity by group (y, src, country)")
print(df.group_by("y", "src", "a_c").agg(pl.len(), *[pl.col(c).cast(pl.Float64).mean().round(3) for c in num]).sort("y", "src", "a_c"))
print("== quantiles (pos vs neg)")
for c in ["n_jacc", "n_3g", "n_tsr", "n_jw", "a_jacc", "a_3g", "a_tsr"]:
    q = df.group_by("y").agg(*[pl.col(c).quantile(x).alias(f"q{int(x*100)}") for x in [.05, .1, .25, .5, .75, .9, .95]]).sort("y")
    print(c, q.rows())
# name relation categories on positives
p = df.filter(pl.col("y") == 1)
B = pl.col("b_name")
cat = (pl.when(pl.col("raw_eq")).then(pl.lit("01 raw exact"))
       .when(pl.col("fold_eq")).then(pl.lit("02 case-only"))
       .when(pl.col("norm_eq")).then(pl.lit("03 punct/space/diacritic"))
       .when(pl.col("tokset_eq")).then(pl.lit("04 reorder"))
       .when(pl.col("core_eq")).then(pl.lit("05 legal-suffix diff"))
       .when(B.str.contains(r"[\x{0900}-\x{0DFF}]") & ~B.str.contains(r"[A-Za-z]")).then(pl.lit("06 full native script"))
       .when(B.str.contains(r"[\x{0900}-\x{0DFF}]")).then(pl.lit("07 partial native script"))
       .when(B.str.contains(r"(?i)\.com|\.in\b|\.net|\.org|www\.")).then(pl.lit("08 domain"))
       .when(B.str.contains(r"(?i)f/k/a|\bn[ée]e\b|\bdba\b|d/b/a|a/k/a|formerly")).then(pl.lit("09 alias kw"))
       .when(pl.col("n_tsr") >= 0.85).then(pl.lit("10 fuzzy tsr>=.85"))
       .when(pl.col("n_tsr") >= 0.6).then(pl.lit("11 fuzzy .6-.85"))
       .otherwise(pl.lit("12 unrelated <.6")))
p = p.with_columns(cat.alias("ncat"))
print("== name relation categories (true pairs)")
print(p.group_by("ncat", "src").len().pivot(on="src", index="ncat", values="len").sort("ncat")
      .with_columns((pl.col("S2") / pl.col("S2").sum()).round(4).alias("S2%"), (pl.col("S3") / pl.col("S3").sum()).round(4).alias("S3%")))
print(p.group_by("ncat", "a_c").len().pivot(on="a_c", index="ncat", values="len").sort("ncat")
      .with_columns((pl.col("US") / pl.col("US").sum()).round(4).alias("US%"), (pl.col("India") / pl.col("India").sum()).round(4).alias("IN%")))
print("== examples of '12 unrelated' and '11 fuzzy'")
for c in ["12 unrelated <.6", "11 fuzzy .6-.85", "09 alias kw", "08 domain"]:
    print(c); [print("   ", r) for r in p.filter(pl.col("ncat") == c).sample(12, seed=0).select("a_name", "b_name", "a_tsr").rows()]
# address among unrelated-name pairs
print("== address sim for positives whose name is unrelated (<.6):", p.filter(pl.col("ncat") == "12 unrelated <.6").select(pl.col("a_tsr").mean(), pl.col("b_addr_missing").mean(), pl.col("a_first_num_eq").mean()).rows())
print("== both-weak positives (n_tsr<.6 and (addr missing or a_tsr<.6)) frac:", p.select(((pl.col("n_tsr") < .6) & (pl.col("b_addr_missing") | (pl.col("a_tsr") < .6))).mean()).item())
# token diffs
drop, add = Counter(), Counter()
for an, bn in p.filter(~pl.col("ncat").is_in(["06 full native script", "08 domain", "12 unrelated <.6"])).select("an", "bn").iter_rows():
    A, Bt = an.split(), bn.split()
    for t in set(A) - set(Bt): drop[t] += 1
    for t in set(Bt) - set(A): add[t] += 1
print("== top tokens DROPPED from S1 name in match:", drop.most_common(40))
print("== top tokens ADDED in S2/S3 name:", add.most_common(60))
# address missing-ness & first number
print("== address: positives b_addr_missing by src:", p.group_by("src").agg(pl.col("b_addr_missing").mean()).rows())
print("== first-number equality among positives with both numbers:", p.group_by("src", "a_c").agg(pl.col("a_first_num_eq").mean()).rows())
print("== first-number equality among negatives:", df.filter(pl.col("y") == 0).group_by("a_c").agg(pl.col("a_first_num_eq").mean()).rows())
