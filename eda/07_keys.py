"""Build normalised strings + blocking keys for all records of a split. Cached to parquet."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm import *
split = sys.argv[1] if len(sys.argv) > 1 else "train"
frames = []
for s in [1, 2, 3]:
    d = load(split, f"s{s}").with_columns(pl.lit(f"S{s}").alias("src"))
    d = d.with_columns(basic(pl.col("business_name")).alias("nb"), basic(pl.col("business_address")).alias("ab0"),
                       pl.col("business_address").is_null().alias("a_missing"))
    d = d.with_columns(name_core(pl.col("nb")).alias("nc"), addr_norm(pl.col("ab0")).alias("ab")).drop("ab0")
    frames.append(d.select("entity_id", "src", "country", "nb", "nc", "ab", "a_missing"))
    print(s, flush=True)
k = pl.concat(frames)
# derived keys
k = k.with_columns(
    pl.col("nc").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.sort().list.join(" ").alias("n_sorted"),
    pl.col("nc").str.replace_all(" ", "").alias("n_nospace"),
    pl.col("ab").str.extract(r"\b(\d+)\b", 1).alias("a_num"),
    pl.col("ab").str.extract(r"\b(\d+) ([a-z]{3,})", 0).alias("a_numstreet"),
)
k.write_parquet(f"{PQ}/{split}_keys.parquet")
# token document frequencies (name core tokens and address tokens), global over all sources, per country
for col, name in [("nc", "ntok"), ("ab", "atok")]:
    t = (k.select("country", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
         .filter(pl.col("t").str.len_chars() >= 2).group_by("country", "t").len().rename({"len": "df"}))
    t.write_parquet(f"{PQ}/{split}_{name}_df.parquet"); print(name, t.height, flush=True)
