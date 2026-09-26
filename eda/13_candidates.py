"""Candidate generation for a whole split (reverse direction: every S2/S3 record queries the S1 index of its country).
  primary : word index over name-core + name-skeleton + address, top-KP (max_df 5000, min_keep 5)
  fallback: char-trigram (no-space) index over name-core, top-KF, only when query has no address or primary top1 < TAU
Output: eda/data/{split}_cand/{country}_{src}_{chunk}.parquet with columns
  rid (str), s1 (str), p_score, p_rank, f_score, f_rank   (rank 99 / score 0 = not retrieved by that retriever)
Memory: one country's S1 indexes (~1 GB) + one 250k-query chunk. Time ~0.7 ms/query on 2 cores."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from retr import Index
import numpy as np, psutil
split = sys.argv[1] if len(sys.argv) > 1 else "train"
KP, KF, TAU, CHUNK = 20, 10, 0.6, 250_000
P_CFG, F_CFG = dict(max_df=5000, min_keep=5), dict(max_df=20000, min_keep=8)
src_dir, out = f"{PQ}/{split}_v2", f"{PQ}/{split}_cand"; os.makedirs(out, exist_ok=True)
proc = psutil.Process(); mem = lambda: f"{proc.memory_info().rss / 2**30:.2f}GB"
countries = sorted({f.split("_", 1)[1][:-8] for f in os.listdir(src_dir) if f.startswith("s1_")})
ptxt = lambda d: d.select(pl.concat_str(["nc", "sk", "ab"], separator=" ")).to_series().to_list()
for country in countries:
    t0 = time.time()
    s1 = pl.read_parquet(f"{src_dir}/s1_{country}.parquet")
    s1_ids = s1["entity_id"].to_numpy()
    ixp = Index(ptxt(s1), 2); ixf = Index(s1["nc"].to_list(), 1); del s1
    ixp.query(["warm up"], K=KP); ixf.query(["warm up"], K=KF)
    print(country, "index built", round(time.time() - t0), "s", mem(), flush=True)
    for src in ["s2", "s3"]:
        R = pl.read_parquet(f"{src_dir}/{src}_{country}.parquet")
        for ci, c0 in enumerate(range(0, R.height, CHUNK)):
            fn = f"{out}/{country}_{src}_{ci:03d}.parquet"
            if os.path.exists(fn): continue
            t1 = time.time(); q = R.slice(c0, CHUNK); nq = q.height
            dP, sP, _ = ixp.query(ptxt(q), K=KP, **P_CFG)
            trig = np.flatnonzero((q["ab"].str.len_chars() == 0).to_numpy() | (sP[:, 0] < TAU))
            dF, sF, _ = ixf.query(q["nc"].gather(trig).to_list(), K=KF, **F_CFG)
            # long format, then outer-merge the two retrievers on (query row, s1 index)
            qi = np.repeat(np.arange(nq, dtype=np.int32), KP); rk = np.tile(np.arange(KP, dtype=np.int8), nq)
            P = pl.DataFrame({"qi": qi, "si": dP.ravel(), "p_score": sP.ravel(), "p_rank": rk}).filter(pl.col("si") >= 0)
            F = pl.DataFrame({"qi": np.repeat(trig.astype(np.int32), KF), "si": dF.ravel(), "f_score": sF.ravel(),
                              "f_rank": np.tile(np.arange(KF, dtype=np.int8), len(trig))}).filter(pl.col("si") >= 0)
            C = P.join(F, on=["qi", "si"], how="full", coalesce=True).with_columns(
                pl.col("p_score").fill_null(0.0), pl.col("f_score").fill_null(0.0),
                pl.col("p_rank").fill_null(99).cast(pl.Int8), pl.col("f_rank").fill_null(99).cast(pl.Int8))
            C = C.with_columns(pl.Series("rid", q["entity_id"].to_numpy()[C["qi"].to_numpy()]),
                               pl.Series("s1", s1_ids[C["si"].to_numpy()])).drop("qi", "si")
            C.write_parquet(fn)
            print(f"{country} {src} chunk {ci} q={nq} trig={len(trig) / nq:.2f} cand/q={C.height / nq:.1f} "
                  f"{(time.time() - t1) / nq * 1000:.2f} ms/q {mem()}", flush=True)
        del R
    del ixp, ixf
print("done")
