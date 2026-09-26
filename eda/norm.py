"""EDA-level normalisers (polars expressions). Deliberately simple; not the final pipeline."""
import polars as pl
from unidecode import unidecode

LEGAL = ["llc", "l l c", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private",
         "llp", "lp", "l p", "plc", "pc", "p c", "sarl", "s a r l", "sas", "s a s", "sasu", "eurl", "sci", "sa", "ei", "the"]
ABBR = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "blvd": "boulevard",
        "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
        "sq": "square", "mkt": "market", "ngr": "nagar", "r": "rue", "bd": "boulevard", "all": "allee", "crs": "cours",
        "chem": "chemin", "rte": "route", "imp": "impasse", "fl": "floor", "flr": "floor", "apt": "apartment", "ste": "suite",
        "n": "north", "s": "south", "e": "east", "w": "west"}

def ascii_fold(e: pl.Expr) -> pl.Expr:
    # unidecode only where needed (non-ascii rows), keeps it fast
    return pl.when(e.str.contains(r"[^\x00-\x7F]")).then(e.map_elements(unidecode, return_dtype=pl.String)).otherwise(e)

def basic(e: pl.Expr) -> pl.Expr:
    return (ascii_fold(e.fill_null("")).str.to_lowercase().str.replace_all("&", " and ")
            .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars())

def name_core(e_basic: pl.Expr) -> pl.Expr:
    pat = r"\b(" + "|".join(LEGAL + ["and"]) + r")\b"
    return e_basic.str.replace_all(pat, " ").str.replace_all(r"\s+", " ").str.strip_chars()

def addr_norm(e_basic: pl.Expr) -> pl.Expr:
    e = e_basic.str.replace_all(r"\b0+(\d)", "$1")   # leading zeros
    for k, v in ABBR.items():
        e = e.str.replace_all(rf"\b{k}\b", v)
    return e.str.replace_all(r"\bnull\b|\bnone\b", " ").str.replace_all(r"\s+", " ").str.strip_chars()
