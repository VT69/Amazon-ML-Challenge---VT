"""Learn a transliteration dictionary (unidecoded native-script token -> English token) from TRAINING pairs only.
True pairs whose query name contains non-ASCII script and has the same token count as the S1 name are aligned by
position. Keep mappings seen >= MIN_N times with share >= MIN_SHARE. Output: eda/data/translit.parquet (tok, eng, n, share)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm2 import basic
MIN_N, MIN_SHARE = 3, 0.6
gt = gt_pairs().select("s1", "rid")
s1 = load("train", "s1").filter(pl.col("country") == "India").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("sn"))
R = pl.concat([load("train", s).filter(pl.col("country") == "India").select(pl.col("entity_id").alias("rid"), pl.col("business_name").alias("qn"))
               for s in ["s2", "s3"]]).filter(pl.col("qn").str.contains(r"[^\x00-\x7F]"))
P = gt.join(R, on="rid").join(s1, on="s1")
print("native-script true pairs:", P.height)
P = P.with_columns(basic(pl.col("qn")).str.split(" ").alias("qt"), basic(pl.col("sn")).str.split(" ").alias("st"))
P = P.filter(pl.col("qt").list.len() == pl.col("st").list.len())
print("same token count:", P.height)
A = P.select(pl.col("qt").alias("tok"), pl.col("st").alias("eng")).explode("tok", "eng").filter(pl.col("tok") != pl.col("eng"))
c = A.group_by("tok", "eng").len().rename({"len": "n"})
c = c.with_columns((pl.col("n") / pl.col("n").sum().over("tok")).alias("share")).sort("n", descending=True).unique("tok", keep="first")
D = c.filter((pl.col("n") >= MIN_N) & (pl.col("share") >= MIN_SHARE)).sort("n", descending=True)
D.write_parquet(f"{PQ}/translit.parquet")
print("dictionary size:", D.height)
print(D.head(40).rows())
# coverage on TEST India native-script names (tokens of transliterated names found in dictionary or already English-like)
T = pl.concat([load("test", s).filter(pl.col("country") == "India").select("business_name") for s in ["s2", "s3"]])
T = T.filter(pl.col("business_name").str.contains(r"[^\x00-\x7F]")).sample(50_000, seed=0)
tt = T.select(basic(pl.col("business_name")).str.split(" ").alias("t")).explode("t")
print("test native tokens covered by dictionary:", round(tt["t"].is_in(D["tok"].implode()).mean(), 4))


# ---- layer 2 (unsupervised): uncovered native tokens -> most frequent English S1-vocabulary word with the same skeleton
from norm2 import skel_tok
V = pl.concat([load(sp, "s1").filter(pl.col("country") == "India").select(basic(pl.col("business_name")).alias("nb")) for sp in ["train", "test"]])
V = V.select(pl.col("nb").str.split(" ").alias("w")).explode("w").filter(pl.col("w").str.contains(r"^[a-z]{3,}$"))
V = V.group_by("w").len().filter(pl.col("len") >= 5)
ASCII = pl.concat([load(sp, s).filter(pl.col("country") == "India", ~pl.col("business_name").str.contains(r"[^\x00-\x7F]"))
                   .select(basic(pl.col("business_name")).str.split(" ").alias("w")) for sp in ["train", "test"] for s in ["s1", "s2", "s3"]])
ASCII = ASCII.explode("w").group_by("w").len().filter(pl.col("len") >= 3).select("w")   # words people write in latin script
V = V.with_columns(pl.col("w").map_elements(skel_tok, return_dtype=pl.String).alias("sk")).sort("len", descending=True).unique("sk", keep="first")
N = pl.concat([load(sp, s).filter(pl.col("country") == "India").select("business_name") for sp in ["train", "test"] for s in ["s2", "s3"]])
N = N.filter(pl.col("business_name").str.contains(r"[^\x00-\x7F]")).select(basic(pl.col("business_name")).str.split(" ").alias("t")).explode("t")
N = N.group_by("t").len().filter(pl.col("len") >= 3, pl.col("t").str.contains(r"^[a-z]{3,}$"))
N = N.join(D.select(pl.col("tok").alias("t")), on="t", how="anti").join(ASCII.select(pl.col("w").alias("t")), on="t", how="anti")
N = N.with_columns(pl.col("t").map_elements(skel_tok, return_dtype=pl.String).alias("sk")).filter(pl.col("sk").str.len_chars() >= 3).join(V.select("sk", pl.col("w").alias("eng")), on="sk")
print("layer-2 skeleton mappings:", N.height, N.sort("len", descending=True).head(30).select("t", "eng").rows())
M = pl.concat([D.select("tok", "eng"), N.select(pl.col("t").alias("tok"), "eng")]).unique("tok", keep="first")
M.write_parquet(f"{PQ}/translit_full.parquet"); print("full map:", M.height)
