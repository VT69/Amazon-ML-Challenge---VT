"""Step 3/6: per-source, per-country pattern profile (train + test)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)

N = pl.col("business_name"); A = pl.col("business_address").fill_null("")
LEGAL = r"(?i)\b(llc|l\.l\.c\.?|inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|pvt\.?|private|llp|lp|l\.p\.|plc|p\.?c\.?|sarl|sas|s\.a\.s\.?|sa|sci|eurl|gmbh|group|holdings?)\b"
flags = {
    "name_upper": N.str.contains(r"^[^a-z]*[A-Z][^a-z]*$"),
    "name_lower": N.str.contains(r"^[^A-Z]*[a-z][^A-Z]*$"),
    "name_nonascii": N.str.contains(r"[^\x00-\x7F]"),
    "name_devanagari": N.str.contains(r"[\x{0900}-\x{097F}]"),
    "name_other_indic": N.str.contains(r"[\x{0980}-\x{0DFF}]"),
    "name_latin_diacritic": N.str.contains(r"[\x{00C0}-\x{024F}]"),
    "name_mixed_script": N.str.contains(r"[\x{0900}-\x{0DFF}]") & N.str.contains(r"[A-Za-z]"),
    "name_domain": N.str.contains(r"(?i)(\.com|\.in|\.net|\.org|\.fr|\.co\b|www\.)"),
    "name_alias_kw": N.str.contains(r"(?i)(f/k/a|\bn[ée]e\b|\bdba\b|d/b/a|\baka\b|a/k/a|formerly|\bt/a\b|trading as|\bex\b)"),
    "name_id_tag": N.str.contains(r"(?i)\(id:|#\d|\bid\s*\d"),
    "name_prefix_junk": N.str.contains(r"^[^\w\x{0900}-\x{0DFF}\x{00C0}-\x{024F}]"),
    "name_paren": N.str.contains(r"[()]"),
    "name_digit": N.str.contains(r"\d"),
    "name_amp": N.str.contains(r"&"),
    "name_and": N.str.contains(r"(?i)\band\b|\bet\b"),
    "name_legal": N.str.contains(LEGAL),
    "name_dblspace": N.str.contains(r"  "),
    "name_repeat_tok": N.str.to_lowercase().str.split(" ").list.eval((pl.element() == pl.element().shift(1)) & (pl.element() != "")).list.any(),
    "addr_null": pl.col("business_address").is_null(),
    "addr_None_lit": A.str.contains(r"(?i)^none$|^nan$|^null$|^n/?a$"),
    "addr_NULL_tok": A.str.contains(r"<NULL>|\bNone\b|\bnan\b"),
    "addr_upper": A.str.contains(r"^[^a-z]*[A-Z][^a-z]*$"),
    "addr_nonascii": A.str.contains(r"[^\x00-\x7F]"),
    "addr_indic": A.str.contains(r"[\x{0900}-\x{0DFF}]"),
    "addr_zip5": A.str.contains(r"\b\d{5}(-\d{4})?\b"),
    "addr_pin6": A.str.contains(r"\b\d{3}\s?\d{3}\b"),
    "addr_starts_digit": A.str.contains(r"^\d"),
    "addr_lead_zero": A.str.contains(r"(^|[ ,])0\d+"),
    "addr_hno_prefix": A.str.contains(r"(?i)\b(h\.?\s?no|hn|door no|block|plot no|flat no|no\.?)\b"),
    "addr_near": A.str.contains(r"(?i)\b(near|opp\.?|opposite|behind|beside|next to|nr\.?)\b"),
    "addr_unit": A.str.contains(r"(?i)\b(unit|suite|ste|apt|#|pmb|lot|bldg|floor|flr)\b"),
    "addr_fraction": A.str.contains(r"\d 1/2\b"),
    "addr_trailing_dash_num": A.str.contains(r"\d-\s"),
    "addr_us_state2_last": A.str.contains(r", [A-Z]{2}$"),
    "addr_ends_state2": A.str.contains(r"\b[A-Z]{2}$"),
}
out = []
for split in ["train", "test"]:
    for s in [1, 2, 3]:
        df = load(split, f"s{s}")
        g = (df.with_columns(**{k: v for k, v in flags.items()},
                             ncomma=A.str.count_matches(","), nlen=N.str.len_chars(), alen=A.str.len_chars(),
                             ntok=N.str.split(" ").list.len())
             .group_by("country").agg(pl.len().alias("n"), *[pl.col(k).mean().round(4) for k in flags],
                                      pl.col("ncomma").mean().round(2), pl.col("nlen").mean().round(1),
                                      pl.col("alen").mean().round(1), pl.col("ntok").mean().round(2))
             .with_columns(pl.lit(split).alias("split"), pl.lit(f"S{s}").alias("src")))
        out.append(g)
res = pl.concat(out).sort("country", "split", "src")
res.write_csv("eda/out/03_profile.csv")
t = res.drop("n").unpivot(index=["country", "split", "src"]).with_columns((pl.col("split").str.slice(0, 2) + "_" + pl.col("country").str.slice(0, 2) + "_" + pl.col("src")).alias("k"))
print(t.pivot(on="k", index="variable", values="value"))
