"""Step 10: reverse-direction retrieval (S2/S3 query -> top-K S1) on a query SAMPLE against the FULL S1 index.
Memory: one country's S1 strings + index (~0.5 GB). Time: measured per 1k queries and extrapolated."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from retr import Index
import numpy as np, psutil
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
N_POS, N_NEG, KMAX = 20_000, 10_000, 50
proc = psutil.Process()
mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
KEYS = pl.scan_parquet(f"{PQ}/train_keys.parquet")
pairs = gt_pairs().select("s1", "rid")
cols = ["entity_id", "nc", "ab"]
METHODS = {  # name -> (string builder, feature mode)
    "A_w":    (lambda d: d["ab"], 2),
    "NA_w":   (lambda d: (d["nc"] + " " + d["ab"]), 2),
    "A_c3":   (lambda d: d["ab"], 0),
    "N_c3":   (lambda d: d["nc"], 0),
}
rows, ranks_all = [], []
for country in ["US", "India"]:
    s1 = KEYS.filter((pl.col("country") == country) & (pl.col("src") == "S1")).select(cols).collect()
    R = KEYS.filter((pl.col("country") == country) & (pl.col("src") != "S1")).select(cols + ["src"])
    R = R.join(pairs.lazy(), left_on="entity_id", right_on="rid", how="left").collect()
    q = pl.concat([R.filter(pl.col("s1").is_not_null()).sample(N_POS, seed=11),
                   R.filter(pl.col("s1").is_null()).sample(N_NEG, seed=12)])
    n_R = R.height; del R
    s1_pos = dict(zip(s1["entity_id"].to_list(), range(s1.height)))
    truth = np.array([s1_pos.get(x, -2) if x is not None else -2 for x in q["s1"].to_list()])
    is_pos = truth >= 0
    print(country, "S1", s1.height, "R", n_R, "queries", q.height, mem(), flush=True)
    for m, (f, mode) in METHODS.items():
        t0 = time.time(); ix = Index(f(s1).fill_null("").to_list(), mode); tb = time.time() - t0
        ix.query(["warm up"], K=KMAX)  # jit warm-up
        cfgs = {"A_w": [(None, 0), (20000, 3), (5000, 3)], "NA_w": [(None, 0), (20000, 4), (5000, 4)],
                "A_c3": [(5000, 8), (20000, 8)], "N_c3": [(5000, 8), (20000, 8)]}[m]
        for max_df, mk in cfgs:
            t0 = time.time(); d, s, work = ix.query(f(q).fill_null("").to_list(), K=KMAX, max_df=max_df, min_keep=mk); tq = time.time() - t0
            hit = (d == truth[:, None])
            rank = np.where(hit.any(1), hit.argmax(1), KMAX)       # KMAX = miss
            r = dict(country=country, method=m, max_df=max_df or 0, min_keep=mk, nnz_M=round(ix.nnz / 1e6, 1), build_s=round(tb, 1),
                     ms_per_q=round(1000 * tq / q.height, 3), full_R_hours=round(tq / q.height * n_R / 3600, 2),
                     work_mean_k=round(work.mean() / 1e3, 1))
            for k in [1, 5, 10, 20, 50]:
                r[f"R@{k}"] = round(float((rank[is_pos] < k).mean()), 4)
            r["top1s_pos"] = round(float(np.median(s[is_pos, 0])), 3); r["top1s_neg"] = round(float(np.median(s[~is_pos, 0])), 3)
            rows.append(r); print(r, mem(), flush=True)
            if True:
                ranks_all.append(pl.DataFrame({"country": country, "method": f"{m}|{max_df}|{mk}", "qid": q["entity_id"], "s1": q["s1"],
                                               "rank": rank.astype(np.int16), "top1": s[:, 0], "top10": s[:, 9]}))
                np.save(f"{PQ}/retr_{country}_{m}_{max_df}_{mk}.npy", d[:, :20])
        del ix
    del s1
res = pl.DataFrame(rows); res.write_csv("out/09_retrieval.csv")
pl.concat(ranks_all).write_parquet(f"{PQ}/retr_ranks.parquet")
print(res)
