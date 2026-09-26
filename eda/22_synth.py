"""Synthetic test-style decoys for TRAINING / VALIDATION folds (0..5) only — built from training data alone.

Test contains look-alike businesses that are *multi-record* (several S2/S3 records agreeing with each other) and differ
from the real S1 only by a small house-number change (and sometimes a legal-form swap / one name word). Train has
none of these (its decoys are lone records that add words like "Enterprises"). For ~55% of S1s in folds 0..5 with true
matches, copy 1-3 of the S1's true records and apply ONE shared change -> a synthetic sibling cluster (label: no match).

Outputs (same formats as real data, so every downstream step just works):
  data/train_v2/{s2syn,s3syn}_{country}.parquet   normalised strings
  data/train_{s2syn,s3syn}.parquet                raw-like rows (for raw-name flags)
  data/train_cand/{country}_syn_{k}.parquet       candidates from the REAL retriever (same config as 13_candidates)
Synthetic ids: S2-/S3- + (9e9 + counter*40 + fold) -> numeric, never collide, and fold == parent S1's fold."""
import sys, os, re, random, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm2 import name_core_v2, skeleton
from retr import Index
import numpy as np
random.seed(7); np.random.seed(7)
FRAC, FOLDS, NF = 0.55, range(0, 6), 40
KP, KF, TAU, CHUNK = 20, 10, 0.6, 250_000
P_CFG, F_CFG = dict(max_df=5000, min_keep=5), dict(max_df=20000, min_keep=8)
LEGAL = {"US": ["llc", "inc", "corp", "pc", "pllc", "ltd", "co", "lp", "llp"],
         "India": ["private limited", "limited", "llp", "pvt ltd", "ltd"]}
LEGAL_RE = r"\b(llc|l l c|inc|incorporated|corp|corporation|co|company|ltd|limited|pvt|private|llp|lp|plc|pc|pllc)\b"
gt = gt_pairs().select("s1", "rid")
idnum = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)


def new_number(t):
    v = int(t)
    if random.random() < 0.7 or len(t) == 1:
        for _ in range(10):
            w = v + random.choice([-1, 1]) * random.randint(1, 20)
            if w > 0 and w != v: return str(w)
        return str(v + 1)
    i = random.randrange(len(t)); d = random.choice([c for c in "0123456789" if c != t[i] and not (i == 0 and c == "0")])
    return t[:i] + d + t[i + 1:]


counter = 0
for country in ["India", "US"]:
    t0 = time.time()
    S1 = pl.read_parquet(f"{PQ}/train_v2/s1_{country}.parquet")
    S1 = S1.with_columns(idnum("entity_id").mod(NF).alias("fold")).filter(pl.col("fold").is_in(list(FOLDS)))
    S1 = S1.with_columns(pl.col("ab").str.extract_all(r"\b\d+\b").alias("nums")).filter(pl.col("nums").list.len() > 0)
    g = gt.join(S1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    R = pl.concat([pl.read_parquet(f"{PQ}/train_v2/{s}_{country}.parquet").join(g.select(pl.col("rid").alias("entity_id")), on="entity_id", how="semi")
                   .with_columns(pl.lit(s).alias("src")) for s in ["s2", "s3"]])
    raw = pl.concat([pl.read_parquet(f"{PQ}/train_{s}.parquet", columns=["entity_id", "business_name"]).join(R.select("entity_id"), on="entity_id", how="semi") for s in ["s2", "s3"]])
    R = R.join(raw, on="entity_id", how="left").join(g.rename({"rid": "entity_id"}), on="entity_id")
    groups = {k[0]: v for k, v in R.group_by("s1")}
    vocab = [w for w in S1["nc"].str.split(" ").explode().drop_nulls().to_list() if len(w) >= 4]
    rows = []
    for s in S1.iter_rows(named=True):
        if s["entity_id"] not in groups or random.random() > FRAC: continue
        recs = groups[s["entity_id"]].to_dicts()
        k = min(len(recs), random.choices([1, 2, 3], [0.3, 0.4, 0.3])[0])
        recs = random.sample(recs, k)
        cand_nums = sorted({t for t in s["nums"] if any(re.search(rf"\b{t}\b", r["ab"] or "") for r in recs)}, key=len, reverse=True)
        if not cand_nums: continue
        t = cand_nums[0]; t2 = new_number(t)
        u = random.random(); kind = "num" if u < 0.6 else ("num+legal" if u < 0.85 else "num+word")
        swap_legal = random.choice(LEGAL[country]); new_word = random.choice(vocab)
        for r in recs:
            if not re.search(rf"\b{t}\b", r["ab"] or ""): continue
            ab = re.sub(rf"\b{t}\b", t2, r["ab"], count=1)
            nb = r["nb"]
            if kind == "num+legal":
                nb = re.sub(r"\s+", " ", re.sub(LEGAL_RE, " ", nb)).strip() + " " + swap_legal
            elif kind == "num+word":
                toks = r["nc"].split()
                if toks: toks[random.randrange(len(toks))] = new_word
                nb = " ".join(toks)
            counter += 1
            eid = f"{'S2' if r['src'] == 's2' else 'S3'}-{9_000_000_000 + counter * NF + s['fold']}"
            rows.append(dict(entity_id=eid, country=country, nb=nb, ab=ab, src=r["src"] + "syn", parent_s1=s["entity_id"], kind=kind,
                             business_name=r["business_name"] if kind == "num" else nb, business_address=ab))
    D = pl.DataFrame(rows)
    D = D.with_columns(name_core_v2(pl.col("nb")).alias("nc"))
    D = D.with_columns(pl.Series("sk", skeleton(D["nc"].to_list())))
    for src in ["s2syn", "s3syn"]:
        x = D.filter(pl.col("src") == src)
        x.select("entity_id", "country", "nb", "nc", "ab", "sk").write_parquet(f"{PQ}/train_v2/{src}_{country}.parquet")
        rawf = f"{PQ}/train_{src}.parquet"
        xr = x.select("entity_id", "business_name", "business_address", "country")
        if os.path.exists(rawf): xr = pl.concat([pl.read_parquet(rawf).filter(pl.col("country") != country), xr])
        xr.write_parquet(rawf)
    print(country, "synthetic records:", D.height, "clusters:", D["parent_s1"].n_unique(), D.group_by("kind").len().rows(), f"{time.time() - t0:.0f}s", flush=True)
    # ---- real retrieval for the synthetic queries (same indexes / config as 13_candidates.py)
    s1 = pl.read_parquet(f"{PQ}/train_v2/s1_{country}.parquet"); s1_ids = s1["entity_id"].to_numpy()
    ptxt = lambda d: d.select(pl.concat_str(["nc", "sk", "ab"], separator=" ")).to_series().to_list()
    ixp = Index(ptxt(s1), 2); ixf = Index(s1["nc"].to_list(), 1); del s1
    for ci, c0 in enumerate(range(0, D.height, CHUNK)):
        q = D.slice(c0, CHUNK); nq = q.height
        dP, sP, _ = ixp.query(ptxt(q), K=KP, **P_CFG)
        trig = np.flatnonzero((q["ab"].str.len_chars() == 0).to_numpy() | (sP[:, 0] < TAU))
        dF, sF, _ = ixf.query(q["nc"].gather(trig).to_list(), K=KF, **F_CFG)
        P = pl.DataFrame({"qi": np.repeat(np.arange(nq, dtype=np.int32), KP), "si": dP.ravel(), "p_score": sP.ravel(),
                          "p_rank": np.tile(np.arange(KP, dtype=np.int8), nq)}).filter(pl.col("si") >= 0)
        F = pl.DataFrame({"qi": np.repeat(trig.astype(np.int32), KF), "si": dF.ravel(), "f_score": sF.ravel(),
                          "f_rank": np.tile(np.arange(KF, dtype=np.int8), len(trig))}).filter(pl.col("si") >= 0)
        C = P.join(F, on=["qi", "si"], how="full", coalesce=True).with_columns(
            pl.col("p_score").fill_null(0.0), pl.col("f_score").fill_null(0.0),
            pl.col("p_rank").fill_null(99).cast(pl.Int8), pl.col("f_rank").fill_null(99).cast(pl.Int8))
        C = C.with_columns(pl.Series("rid", q["entity_id"].to_numpy()[C["qi"].to_numpy()]),
                           pl.Series("s1", s1_ids[C["si"].to_numpy()])).drop("qi", "si")
        C.write_parquet(f"{PQ}/train_cand/{country}_syn_{ci:03d}.parquet")
        top1_parent = C.filter(pl.col("p_rank") == 0).join(D.select(pl.col("entity_id").alias("rid"), "parent_s1"), on="rid")
        print(f"  {country} syn chunk {ci}: q={nq} cand/q={C.height / nq:.1f}  top-1 == parent S1: {(top1_parent['s1'] == top1_parent['parent_s1']).mean():.3f}", flush=True)
print("synthetic done", counter)
