"""Final step: write output/matching_results.tsv and output/candidate_pairs.tsv from the scored test candidates.

  python 09_make_output.py [--tag all_v1] [--t 0.8] [--t-country France=0.9]

Decision rule: every S2/S3 record is linked to its single highest-probability S1 candidate if prob >= threshold
(each S2/S3 record matches at most one S1). Submitted setting (version v019, model all_v3, public LB 0.96423):
threshold 0.8 for US and India, 0.9 for France.
candidate_pairs.tsv is the full candidate set the model scored (every S2/S3 candidate of each S1), written in
S1-hash buckets so it never needs more than ~2 GB RAM."""
import sys, os, glob, argparse, subprocess; sys.path.insert(0, os.path.dirname(__file__))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="all_v3"); ap.add_argument("--t", type=float, default=0.8)
ap.add_argument("--t-country", default="France=0.9", help="per-country thresholds overriding --t")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
s1 = pl.read_parquet(f"{PQ}/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})

# ---- matching_results.tsv (queries are partitioned across chunk files -> per-query argmax chunk by chunk)
best = pl.concat([pl.read_parquet(f, columns=["rid", "s1", "prob"]).sort("prob", descending=True).unique("rid", keep="first")
                  .with_columns(pl.lit(os.path.basename(f).split("_")[0]).alias("country"))
                  for f in sorted(glob.glob(f"{PQ}/pred_test_{a.tag}/*.parquet"))])
tc = {k: float(v) for k, v in (kv.split("=") for kv in a.t_country.split(",") if kv)}
thr = pl.col("country").replace_strict(tc, default=a.t, return_dtype=pl.Float64) if tc else pl.lit(a.t)
links = best.filter(pl.col("prob") >= thr).select("rid", "s1")
grp = links.sort("rid").group_by("s1").agg(pl.col("rid").str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
M = s1.join(grp, on="source1_entity_id", how="left")
M.write_csv(os.path.join(OUT, "matching_results.tsv"), separator="\t", null_value="", quote_style="never")
print("matching_results.tsv:", M.height, "S1 rows,", links.height, "links,", int(M["matched_entity_ids"].is_not_null().sum()), "S1 with matches", flush=True)

# ---- candidate_pairs.tsv
NB = 8
files = sorted(glob.glob(f"{PQ}/test_cand/*.parquet"))
cc = os.path.join(OUT, "candidate_pairs.tsv")
with open(cc + ".tmp", "w", encoding="utf-8", newline="") as out:
    out.write("source1_entity_id\tcandidate_entity_ids\n")
    for b in range(NB):
        inb = pl.col("s1").hash(seed=0) % NB == b
        G = (pl.scan_parquet(files).select("rid", "s1").filter(inb).unique()
             .group_by("s1").agg(pl.col("rid").sort().str.join(",").alias("c")).collect())
        B = s1.filter(pl.col("source1_entity_id").hash(seed=0) % NB == b).rename({"source1_entity_id": "s1"}).join(G, on="s1", how="left")
        for sid, c in B.iter_rows():
            out.write(f"{sid}\t{c or ''}\n")
        print("candidate bucket", b, B.height, flush=True)
os.replace(cc + ".tmp", cc)

# ---- checks: streamed rule check of the candidate file (+ matches are subsets of candidates)
c = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "10_check_candidates.py"), cc,
                    os.path.join(OUT, "matching_results.tsv")], capture_output=True, text=True)
print(c.stdout, c.stderr)
print("Official format check (matching file): python <student_resource>/utils/validate_submission.py "
      f"--matching {os.path.join(OUT, 'matching_results.tsv')} --test-dir {os.path.join(RAW, 'test')} --check-ids")
