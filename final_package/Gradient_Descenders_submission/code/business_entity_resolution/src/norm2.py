"""Normaliser v2 (polars expressions + small python helpers).

Changes vs norm.py:
  * state/region canonicalisation on whole comma-separated address components (so 'CT' != 'court'),
    covering English names, 2-letter codes and native-script forms  -> single token 'zs<code>'
  * country-aware street abbreviations (France: st -> saint, r -> rue; US/India: st -> street)
  * mojibake 'Â' removed; leetspeak fixed inside name tokens that mix letters and digits (channe1 -> channel)
  * name consonant skeleton (praaivett / private -> prvt) for transliterated native-script names and typos
Nothing here is specific to a closed set of countries: unknown countries just get the generic path.
"""
import re
import polars as pl
from unidecode import unidecode

STATES = {
    # India: english, variants, native script, code
    "mh": ["maharashtra", "महाराष्ट्र"], "dl": ["delhi", "दिल्ली", "nct of delhi"], "up": ["uttar pradesh", "उत्तर प्रदेश"],
    "ka": ["karnataka", "ಕರ್ನಾಟಕ"], "tn": ["tamil nadu", "tamilnadu", "தமிழ்நாடு"], "gj": ["gujarat", "ગુજરાત"],
    "wb": ["west bengal", "পশ্চিমবঙ্গ"], "tg": ["telangana", "తెలంగాణ"], "hr": ["haryana", "हरियाणा"],
    "rj": ["rajasthan", "राजस्थान"], "kl": ["kerala", "keralam", "കേരളം"], "br": ["bihar", "बिहार"],
    "mp": ["madhya pradesh", "मध्य प्रदेश"], "ap": ["andhra pradesh", "ఆంధ్రప్రదేశ్", "a p"], "pb": ["punjab", "ਪੰਜਾਬ"],
    "od": ["odisha", "orissa", "ଓଡ଼ିଶା"],
    # US
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas"], "ca": ["california"], "co": ["colorado"],
    "ct": ["connecticut"], "de": ["delaware"], "dc": ["district of columbia"], "fl": ["florida"], "ga": ["georgia"],
    "hi": ["hawaii"], "id": ["idaho"], "il": ["illinois"], "in": ["indiana"], "ia": ["iowa"], "ks": ["kansas"],
    "ky": ["kentucky"], "la": ["louisiana"], "me": ["maine"], "md": ["maryland"], "ma": ["massachusetts"],
    "mi": ["michigan"], "mn": ["minnesota"], "ms": ["mississippi"], "mo": ["missouri"], "mt": ["montana"],
    "ne": ["nebraska"], "nv": ["nevada"], "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"],
    "ny": ["new york"], "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"], "ok": ["oklahoma"],
    "or": ["oregon"], "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"], "sd": ["south dakota"],
    "tx": ["texas"], "ut": ["utah"], "vt": ["vermont"], "va": ["virginia"], "wa": ["washington"], "wv": ["west virginia"],
    "wi": ["wisconsin"], "wy": ["wyoming"],
}
# France (test only): regions and the departements the sources substitute for them -> region code
FR_REGIONS = {"hdf": ["hauts de france", "nord", "pas de calais"], "naq": ["nouvelle aquitaine", "gironde"],
              "pdl": ["pays de la loire", "loire atlantique"]}
CITY_SYN = {"bombay": "mumbai", "calcutta": "kolkata", "bengaluru": "bangalore", "madras": "chennai",
            "gurugram": "gurgaon", "ahmadabad": "ahmedabad"}


def _key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", unidecode(s).lower())).strip()


def _comp_map(country):
    m = {}
    if country == "France":
        for code, names in FR_REGIONS.items():
            for n in names: m[_key(n)] = "zs" + code
        return m
    for code, names in STATES.items():
        # codes are only trusted as a whole component; 'in'/'or'/'me' never collide there
        m[code] = "zs" + code
        for n in names:
            m[n] = "zs" + code          # raw form (native script)
            m[_key(n)] = "zs" + code    # ascii-folded form
    return m


ABBR_COMMON = {"rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "blvd": "boulevard",
               "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square",
               "fl": "floor", "flr": "floor", "apt": "apartment", "ste": "suite", "bldg": "building", "no": "no",
               "n": "north", "s": "south", "e": "east", "w": "west", "ct": "court", "pl": "place",
               "mkt": "market", "ngr": "nagar", "opp": "opposite", "nr": "near", "bhd": "behind"}
ABBR_FR = {"r": "rue", "bd": "boulevard", "bld": "boulevard", "av": "avenue", "ave": "avenue", "all": "allee",
           "crs": "cours", "chem": "chemin", "rte": "route", "imp": "impasse", "pl": "place", "st": "saint",
           "ste": "sainte", "apt": "appartement", "bat": "batiment", "res": "residence", "qu": "quai", "n": "no"}


def fold(e: pl.Expr) -> pl.Expr:
    e = e.fill_null("").str.replace_all("Â", " ")
    return pl.when(e.str.contains(r"[^\x00-\x7F]")).then(e.map_elements(unidecode, return_dtype=pl.String)).otherwise(e)


def basic(e: pl.Expr) -> pl.Expr:
    return (fold(e).str.to_lowercase().str.replace_all("&", " and ").str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def addr_v2(col: str, country: str) -> pl.Expr:
    """Raw address column -> normalised address string for one country."""
    cmap = _comp_map(country)
    comps = (pl.col(col).fill_null("").str.replace_all("Â", " ").str.split(",")
             .list.eval(pl.element().str.strip_chars().str.to_lowercase()))
    # exact component replacement (raw native forms first, then ascii key)
    comps = comps.list.eval(pl.element().replace(cmap))
    e = basic(comps.list.join(" , ")).str.replace_all(r"\b0+(\d)", "$1")
    ab = ABBR_FR if country == "France" else dict(ABBR_COMMON, st="street")
    # rebuild word-by-word with dictionary expansion (vectorised via list.eval + replace)
    wmap = {k: v for k, v in cmap.items() if k.isascii() and k.isalpha() and len(k) > 3}   # full one-word names only
    words = e.str.split(" ").list.eval(pl.element().replace(ab).replace(CITY_SYN).replace(wmap))
    e = words.list.join(" ")
    return (e.str.replace_all(r"\b(null|none|n a|na)\b", " ").str.replace_all(r"\s+", " ").str.strip_chars())


LEGAL = ["llc", "l l c", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private",
         "llp", "lp", "l p", "plc", "pc", "p c", "pllc", "sarl", "s a r l", "sas", "s a s", "sasu", "eurl", "sci", "sa", "ei",
         "the", "and", "dba", "d b a", "aka", "fka", "f k a", "formerly", "known", "as", "doing", "business", "trading",
         "limittedd", "praaivett", "praiveett", "lnc", "pvtltd", "com", "www", "net", "org"]
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a"})


_TRMAP = None


def translit_map():
    """unidecoded native-script token -> English token, learned from training pairs (17_translit.py)."""
    global _TRMAP
    if _TRMAP is None:
        import os
        from common import PQ
        fn = os.path.join(PQ, "translit_full.parquet")
        _TRMAP = dict(pl.read_parquet(fn).select("tok", "eng").iter_rows()) if os.path.exists(fn) else {}
    return _TRMAP


def name_v2(col: str) -> pl.Expr:
    """Raw name -> basic normalised name with leetspeak fixed in mixed alnum tokens; for names written in a
    non-Latin script, tokens are mapped through the learned transliteration dictionary."""
    e = basic(pl.col(col))
    fix = e.str.split(" ").list.eval(
        pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9]"))
        .then(pl.element().str.replace_all("0", "o").str.replace_all("1", "l").str.replace_all("3", "e")
              .str.replace_all("5", "s").str.replace_all("4", "a").str.replace_all("7", "t"))
        .otherwise(pl.element()))
    tm = translit_map()
    if not tm:
        return fix.list.join(" ")
    mapped = fix.list.eval(pl.element().replace(tm))
    return (pl.when(pl.col(col).str.contains(r"[^\x00-\x7F]")).then(mapped.list.join(" "))
            .otherwise(fix.list.join(" ")))


def name_core_v2(e_name: pl.Expr) -> pl.Expr:
    pat = r"\b(" + "|".join(LEGAL) + r")\b"
    return e_name.str.replace_all(pat, " ").str.replace_all(r"\s+", " ").str.strip_chars()


_SK = [("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("w", "v"), ("x", "ks")]
_cache = {}


def skel_tok(t):
    r = _cache.get(t)
    if r is None:
        s = t
        for a, b in _SK: s = s.replace(a, b)
        out = s[0]
        for ch in s[1:]:
            if ch in "aeiouyh" or ch == out[-1]: continue
            out += ch
        r = _cache[t] = out
    return r


def skeleton(names):
    """iterable of normalised names -> 'zk<skel> ...' tokens (alphabetic tokens of len>=3 only)."""
    return [" ".join("zk" + skel_tok(t) for t in (n or "").split() if len(t) >= 3 and t.isalpha()) for n in names]
