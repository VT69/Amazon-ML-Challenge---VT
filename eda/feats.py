"""Pair features for (query = S2/S3 record, candidate = S1 record).

Design for 12 GB RAM: entity strings live once per side; pairs reference them by integer row index. Token-overlap
features use integer feature ids (retr.featurize) and a numba merge-join; string similarities use rapidfuzz.cpdist.
No country feature is used (France is unseen in training) - only country-agnostic similarity signals.
"""
import os
import numpy as np
import numba as nb
import polars as pl
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler
from retr import to_buf, featurize, NFEAT
from common import PQ

ALIAS = r"(?i)\b(dba|d/b/a|d\.b\.a|formerly|f/k/a|fka|aka|a/k/a|known as|doing business as)\b"


# ---------------------------------------------------------------- sparse overlap
@nb.njit(parallel=True, cache=True)
def _overlap(qp, qi, dp, di, pq, pd, idf):
    n = len(pq)
    out = np.zeros((n, 6), dtype=np.float32)   # cover_q, cover_d, cosine, n_shared, n_q, n_d
    for k in nb.prange(n):
        a0, a1 = qp[pq[k]], qp[pq[k] + 1]; b0, b1 = dp[pd[k]], dp[pd[k] + 1]
        wq = 0.0; wd = 0.0; ws = 0.0; ns = 0
        for j in range(a0, a1): wq += idf[qi[j]] ** 2
        for j in range(b0, b1): wd += idf[di[j]] ** 2
        i, j = a0, b0
        while i < a1 and j < b1:
            if qi[i] == di[j]:
                ws += idf[qi[i]] ** 2; ns += 1; i += 1; j += 1
            elif qi[i] < di[j]: i += 1
            else: j += 1
        out[k, 0] = ws / wq if wq > 0 else np.nan
        out[k, 1] = ws / wd if wd > 0 else np.nan
        out[k, 2] = ws / np.sqrt(wq * wd) if wq > 0 and wd > 0 else np.nan
        out[k, 3] = ns; out[k, 4] = a1 - a0; out[k, 5] = b1 - b0
    return out


# ---------------------------------------------------------------- number relations
@nb.njit(cache=True)
def _parse_nums(buf, off):
    """space separated digit tokens -> CSR of int64 values (last 15 digits) and digit lengths."""
    n = len(off) - 1
    vals = np.empty(len(buf) // 2 + n + 1, dtype=np.int64); lens = np.empty(len(vals), dtype=np.int8)
    ptr = np.zeros(n + 1, dtype=np.int64); k = 0
    for r in range(n):
        v = 0; L = 0
        for i in range(off[r], off[r + 1] + 1):
            c = buf[i] if i < off[r + 1] else 32
            if 48 <= c <= 57:
                if L < 15: v = v * 10 + (c - 48); L += 1
            elif L > 0:
                vals[k] = v; lens[k] = L; k += 1; v = 0; L = 0
        ptr[r + 1] = k
    return ptr, vals[:k].copy(), lens[:k].copy()


@nb.njit(cache=True)
def _is_fix(a, la, b, lb):
    """decimal a is a prefix or suffix of decimal b (dropped leading/trailing digits)."""
    if la >= lb: return False
    return b % 10 ** la == a or b // 10 ** (lb - la) == a


@nb.njit(parallel=True, cache=True)
def _num_feats(qp, qv, ql, sp, sv, sl, pq, pd):
    """per pair: query numbers absent from S1, classified against the S1 numbers not matched exactly:
    truncation (prefix/suffix either way) / near-change (|d|<=20) / other; S1 numbers absent from query;
    min |diff| of changed numbers; first/max number equality (nan when a side has no numbers)."""
    n = len(pq); out = np.full((n, 8), np.nan, dtype=np.float32)
    for k in nb.prange(n):
        a0, a1 = qp[pq[k]], qp[pq[k] + 1]; b0, b1 = sp[pd[k]], sp[pd[k] + 1]
        if a1 == a0 and b1 == b0: continue
        smask = 0; qmask = 0
        for i in range(a0, min(a1, a0 + 62)):
            for j in range(b0, min(b1, b0 + 62)):
                if qv[i] == sv[j]:
                    smask |= 1 << (j - b0); qmask |= 1 << (i - a0)
        tr = 0; ne = 0; ot = 0; ad = 0; md = 1e9
        nfree = 0
        for j in range(b0, min(b1, b0 + 62)):
            if not smask & (1 << (j - b0)): nfree += 1
        for i in range(a0, min(a1, a0 + 62)):
            if qmask & (1 << (i - a0)): continue
            if nfree == 0:
                ad += 1; continue
            kind = 2
            for j in range(b0, min(b1, b0 + 62)):
                if smask & (1 << (j - b0)): continue
                if _is_fix(qv[i], ql[i], sv[j], sl[j]) or _is_fix(sv[j], sl[j], qv[i], ql[i]):
                    kind = 0; break
            if kind == 2:
                for j in range(b0, min(b1, b0 + 62)):
                    if smask & (1 << (j - b0)): continue
                    d = abs(qv[i] - sv[j])
                    if d < md: md = d
                    if d <= 20: kind = 1
            if kind == 0: tr += 1
            elif kind == 1: ne += 1
            else: ot += 1
        mi = 0
        for j in range(b0, min(b1, b0 + 62)):
            if not smask & (1 << (j - b0)): mi += 1
        out[k, 0] = tr; out[k, 1] = ne; out[k, 2] = ot; out[k, 3] = ad; out[k, 4] = mi
        out[k, 5] = md if md < 1e9 else np.nan
        if a1 > a0 and b1 > b0:
            out[k, 6] = 1.0 if qv[a0] == sv[b0] else 0.0
            qm = qv[a0]; sm = sv[b0]
            for i in range(a0, a1): qm = max(qm, qv[i])
            for j in range(b0, b1): sm = max(sm, sv[j])
            out[k, 7] = 1.0 if qm == sm else 0.0
    return out

def csr(strings, mode):
    b, o = to_buf(strings)
    return featurize(b, o, mode)


def idf_from(ptr, ids, nfeat):
    n = len(ptr) - 1
    df = np.bincount(ids, minlength=nfeat).astype(np.float32)
    return np.log((n + 1.0) / (df + 1.0)).astype(np.float32)


# ---------------------------------------------------------------- entity side tables
SPACES = {  # name -> (column, feature mode)
    "nw": ("nc", 2), "sk": ("sk", 2), "n3": ("nc", 1), "aw": ("ab", 2), "num": ("num", 2),
}


def load_side(split, src, country, ids=None):
    """v2 strings (+ raw-name flags) for one source/country, optionally restricted to entity ids."""
    d = pl.scan_parquet(f"{PQ}/{split}_v2/{src}_{country}.parquet")
    r = pl.scan_parquet(f"{PQ}/{split}_{src}.parquet").select("entity_id", "business_name")
    if ids is not None:
        d = d.join(ids.lazy(), on="entity_id", how="semi")
        r = r.join(ids.lazy(), on="entity_id", how="semi")
    else:
        r = r.join(d.select("entity_id"), on="entity_id", how="semi")
    r = r.select("entity_id",
                 pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("nonascii"),
                 pl.col("business_name").str.contains(r"(?i)(\.com|www\.|\.in\b|\.net|\.org|\.fr\b)").alias("domain"),
                 pl.col("business_name").str.contains(ALIAS).alias("alias"),
                 pl.col("business_name").str.len_chars().alias("raw_len"))
    d = d.join(r, on="entity_id", how="left").collect()
    return d.with_columns(
        pl.col("ab").str.extract_all(r"\b\d+\b").list.join(" ").alias("num"),
        pl.col("ab").str.extract(r"\b(zs[a-z]+)\b", 1).alias("state"),
        pl.col("nc").str.replace_all(" ", "").alias("ncns"))


class Side:
    def __init__(self, df):
        self.df = df
        self.row = pl.DataFrame({"entity_id": df["entity_id"], "_row": np.arange(df.height, dtype=np.int32)})
        self.sp = {k: csr(df[c].fill_null("").to_list(), m) for k, (c, m) in SPACES.items()}
        self.np = {c: df[c].fill_null("").to_numpy() for c in ("nc", "ncns")}
        self.nums = _parse_nums(*to_buf(df["num"].fill_null("").to_list()))


# ---------------------------------------------------------------- features
def pair_features(C, Q: Side, S: Side, idf):
    """C: candidates (rid, s1, p_score, p_rank, f_score, f_rank). Returns C with feature columns."""
    C = (C.join(Q.row.rename({"entity_id": "rid", "_row": "qr"}), on="rid")
          .join(S.row.rename({"entity_id": "s1", "_row": "sr"}), on="s1"))
    pq, pd = C["qr"].to_numpy(), C["sr"].to_numpy()
    feats = {}
    for k in SPACES:
        qp, qi = Q.sp[k]; dp, di = S.sp[k]
        o = _overlap(qp, qi, dp, di, pq, pd, idf[k])
        feats[f"{k}_cq"], feats[f"{k}_cd"], feats[f"{k}_cos"], feats[f"{k}_ns"] = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        feats[f"{k}_extra"] = o[:, 4] - o[:, 3]; feats[f"{k}_miss"] = o[:, 5] - o[:, 3]
        if k in ("nw", "aw", "num"): feats[f"{k}_nq"] = o[:, 4]
    nf = _num_feats(*Q.nums, *S.nums, pq, pd)
    for j, name in enumerate(["nm_trunc", "nm_near", "nm_changed", "nm_added", "nm_miss", "nm_mindiff", "nm_first_eq", "nm_max_eq"]):
        feats[name] = nf[:, j]
    g = lambda side, col, idx: side.np[col][idx].tolist()
    qn, sn = g(Q, "nc", pq), g(S, "nc", pd)
    cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)
    feats["n_tsr"] = cp(qn, sn, fuzz.token_set_ratio)
    feats["n_ratio"] = cp(qn, sn, fuzz.ratio); feats["n_pr"] = cp(qn, sn, fuzz.partial_ratio)
    feats["n_jw"] = cp(qn, sn, JaroWinkler.normalized_similarity)
    qns, sns = g(Q, "ncns", pq), g(S, "ncns", pd)
    feats["nns_pr"] = cp(qns, sns, fuzz.partial_ratio); feats["nns_ratio"] = cp(qns, sns, fuzz.ratio)
    # address fuzzy scorers dropped: 50us/pair; order-invariant idf token overlaps (aw_*, num_*) carry the signal
    del qn, sn, qns, sns
    C = C.with_columns([pl.Series(k, v) for k, v in feats.items()])
    qcol = lambda c, name: Q.df[c].gather(pq).alias(name)
    scol = lambda c, name: S.df[c].gather(pd).alias(name)
    eq = lambda c, name: (Q.df[c].fill_null("").gather(pq) == S.df[c].fill_null("").gather(pd)).alias(name)
    C = C.with_columns(qcol("nonascii", "q_nonascii"), qcol("domain", "q_domain"), qcol("alias", "q_alias"),
                       qcol("raw_len", "q_rawlen"), scol("raw_len", "s_rawlen"),
                       qcol("state", "q_state"), scol("state", "s_state"),
                       (Q.df["ab"].str.len_chars() == 0).gather(pq).alias("q_noaddr"),
                       scol("nc_dup", "s_nc_dup"), scol("ab_dup", "s_ab_dup"),
                       eq("nc", "n_eq"), eq("sk", "sk_eq"), eq("ncns", "nns_eq"))
    C = C.with_columns(
        pl.when(pl.col("q_state").is_null() | pl.col("s_state").is_null()).then(None)
          .otherwise(pl.col("q_state") == pl.col("s_state")).cast(pl.Float32).alias("state_eq"),
        pl.col("rid").str.starts_with("S3").alias("is_s3"),
        (pl.col("f_rank") < 99).alias("by_fallback"), (pl.col("p_rank") < 99).alias("by_primary"),
    ).drop("q_state", "s_state", "qr", "sr")
    # within-query context: how this candidate compares with the query's other candidates
    ctx = ["p_score", "n_tsr", "nw_cos", "aw_cq", "num_cq", "sk_cos", "n3_cos"]
    C = C.with_columns([(pl.col(c) - pl.col(c).max().over("rid")).alias(f"{c}_gapmax") for c in ctx] +
                       [pl.col(c).rank("min", descending=True).over("rid").cast(pl.Int16).alias(f"{c}_rk") for c in ctx] +
                       [pl.len().over("rid").cast(pl.Int16).alias("n_cand")])
    return C


def s1_side(split, country):
    """Full S1 side for a country with duplicate counts (name/address ambiguity in the reference source)."""
    d = load_side(split, "s1", country)
    d = d.with_columns(pl.len().over("nc").cast(pl.Int32).alias("nc_dup"), pl.len().over("ab").cast(pl.Int32).alias("ab_dup"))
    S = Side(d)
    idf = {k: idf_from(*S.sp[k], NFEAT[m]) for k, (c, m) in SPACES.items()}
    return S, idf


def r_side(split, country, rids):
    parts = []
    for src in ["s2", "s3", "s2syn", "s3syn"]:
        fn = f"{PQ}/{split}_v2/{src}_{country}.parquet"
        if os.path.exists(fn):
            parts.append(load_side(split, src, country, ids=rids.select("entity_id")))
    d = pl.concat(parts).with_columns(pl.lit(0, pl.Int32).alias("nc_dup"), pl.lit(0, pl.Int32).alias("ab_dup"))
    return Side(d)


FEATURES = None  # filled by training script from the frame columns (everything except ids/label)
