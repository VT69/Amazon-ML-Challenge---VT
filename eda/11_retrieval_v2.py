"""Step 11: retrieval with normaliser v2 + skeleton tokens + name-char fallback. Same query sample as step 10.
Memory ~2 GB per country; ~10 min total."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from norm2 import addr_v2, name_v2, name_core_v2, skeleton
from retr import Index
import numpy as np, psutil
N_POS, N_NEG, KMAX = 20_000, 10_000, 50
proc = psutil.Process(); mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
pairs = gt_pairs().select("s1", "rid")
raw = {s: pl.scan_parquet(f"{PQ}/train_{s}.parquet") for s in ["s1", "s2", "s3"]}


def prep(df, country):
    df = df.with_columns(name_v2("business_name").alias("nb"), addr_v2("business_address", country).alias("ab"))
    df = df.with_columns(name_core_v2(pl.col("nb")).alias("nc"))
    return df.with_columns(pl.Series("sk", skeleton(df["nc"].to_list())))


rows, out = [], []
for country in ["US", "India"]:
    t0 = time.time()
    s1 = prep(raw["s1"].filter(pl.col("country") == country).collect(), country)
    R = pl.concat([raw[s].filter(pl.col("country") == country).select("entity_id") .collect() for s in ["s2", "s3"]])
    R = R.join(pairs, left_on="entity_id", right_on="rid", how="left")
    qid = pl.concat([R.filter(pl.col("s1").is_not_null()).sample(N_POS, seed=11), R.filter(pl.col("s1").is_null()).sample(N_NEG, seed=12)])
    n_R = R.height; del R
    q = pl.concat([raw[s].filter(pl.col("country") == country).collect() for s in ["s2", "s3"]]).join(qid, on="entity_id").sort("entity_id")
    q = prep(q, country)
    print(country, "prep", round(time.time() - t0), "s", mem(), flush=True)
    s1_pos = dict(zip(s1["entity_id"].to_list(), range(s1.height)))
    truth = np.array([s1_pos.get(x, -2) if x is not None else -2 for x in q["s1"].to_list()]); is_pos = truth >= 0
    no_addr = (q["ab"].str.len_chars() == 0).to_numpy()
    txt = lambda d, parts: d.select(pl.concat_str([pl.col(p) for p in parts], separator=" ")).to_series().to_list()
    res = {}
    for m, parts, mode, cfgs in [("NA2", ["nc", "ab"], 2, [(20000, 4), (5000, 4)]),
                                 ("NAS2", ["nc", "sk", "ab"], 2, [(20000, 5), (5000, 5)]),
                                 ("N_c3ns", ["nc"], 1, [(20000, 8)]),
                                 ("NS", ["nc", "sk"], 2, [(None, 0)])]:
        ix = Index(txt(s1, parts), mode); ix.query(["warm up"], K=KMAX)
        for max_df, mk in cfgs:
            t1 = time.time(); d, s, work = ix.query(txt(q, parts), K=KMAX, max_df=max_df, min_keep=mk); tq = time.time() - t1
            hit = d == truth[:, None]; rank = np.where(hit.any(1), hit.argmax(1), KMAX)
            key = f"{m}|{max_df}|{mk}"; res[key] = (d, s, rank)
            r = dict(country=country, method=key, ms_per_q=round(1000 * tq / q.height, 3), full_R_h=round(tq / q.height * n_R / 3600, 2))
            for k in [1, 5, 10, 20, 50]: r[f"R@{k}"] = round(float((rank[is_pos] < k).mean()), 4)
            r["R@50_noaddr"] = round(float((rank[is_pos & no_addr] < 50).mean()), 4)
            rows.append(r); print(r, mem(), flush=True)
        del ix
    # union / adaptive fallback simulation: primary NAS2|20000|5 top-Kp  U  fallback top-Kf when triggered
    dP, sP, rP = res["NAS2|20000|5"]
    for fb in ["N_c3ns|20000|8", "NS|None|0"]:
        dF, sF, rF = res[fb]
        for Kp, Kf in [(10, 10), (20, 10), (20, 20), (50, 20)]:
            for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 2.0]:          # trigger when no address or top1 < tau (2.0 = always)
                trig = no_addr | (sP[:, 0] < tau)
                hitu = (rP < Kp) | (trig & (rF < Kf))
                cand = np.array([len(set(dP[i, :Kp]) | (set(dF[i, :Kf]) if trig[i] else set())) - (1 if -1 in set(dP[i, :Kp]) | (set(dF[i, :Kf]) if trig[i] else set()) else 0) for i in range(len(trig))])
                rows.append(dict(country=country, method=f"U[{Kp}]+{fb}[{Kf}] tau={tau}", trig_frac=round(float(trig.mean()), 3),
                                 R_union=round(float(hitu[is_pos].mean()), 4), cand_mean=round(float(cand.mean()), 1)))
                print(rows[-1], flush=True)
    out.append(q.select("entity_id", "s1", "nc", "ab").with_columns(pl.Series("rank_nas2", rP), pl.Series("top1_nas2", sP[:, 0]), pl.lit(country).alias("country")))
    del s1, q
pl.DataFrame([r for r in rows if "R@1" in r]).write_csv("out/11_retrieval_v2.csv")
pl.DataFrame([r for r in rows if "R_union" in r]).write_csv("out/11_union.csv")
pl.concat(out).write_parquet(f"{PQ}/retr_v2_ranks.parquet")
