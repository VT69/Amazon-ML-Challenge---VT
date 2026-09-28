"""Cut a new, numbered submission version from stored test predictions. Never overwrites earlier versions.

  python 19_make_submission.py --tag <model_tag> --t 0.03 [--note "what changed"] [--val-tag <tag>]

Writes submissions/vNNN_<tag>_t<thr>/ with matching_results.tsv, candidate_pairs.tsv (hard link to a cached copy:
the candidate set is shared by every version built on the same candidates), manifest.json, validator output,
and appends a row to submissions/INDEX.md (with an empty column for the leaderboard score)."""
import sys, os, glob, json, argparse, subprocess, datetime, shutil; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from evalf import assign, macro_f05, macro_f05_testlike

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--t", type=float, required=True)
ap.add_argument("--note", default=""); ap.add_argument("--t-country", default="", help='e.g. "France=0.9,India=0.8"'); ap.add_argument("--rule", default="", help="cluster rule set from rules.RULESETS");
ap.add_argument("--empty-country", default="", help="DIAGNOSTIC: predict no matches for this country's S1s"); ap.add_argument("--val-tag", default=None); ap.add_argument("--cand", default="c1")
a = ap.parse_args()
SUBS = os.path.join(ROOT, "submissions"); CACHE = os.path.join(SUBS, "_cache"); os.makedirs(CACHE, exist_ok=True)
TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")

# ---- version folder
nums = [int(os.path.basename(d)[1:4]) for d in glob.glob(os.path.join(SUBS, "v[0-9][0-9][0-9]_*"))]
ver = f"v{(max(nums) + 1 if nums else 1):03d}"
name = f"{ver}_{a.tag}_t{a.t:g}" + "".join(f"_{kv.replace('=', '')}" for kv in a.t_country.split(",") if kv) + (f"_EMPTY{a.empty_country}" if a.empty_country else "") + (f"_{a.rule}" if a.rule else "")
vd = os.path.join(SUBS, name); os.makedirs(vd)

# ---- matching_results.tsv: every test S1 exactly once, each S2/S3 record linked to its best S1 if prob >= t
s1 = pl.read_parquet(f"{PQ}/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
# queries are partitioned across chunk files, so the per-query argmax can be taken chunk by chunk (keeps RAM small)
best = pl.concat([pl.read_parquet(f, columns=["rid", "s1", "prob"]).sort("prob", descending=True).unique("rid", keep="first")
                  .with_columns(pl.lit(os.path.basename(f).split("_")[0]).alias("country"))
                  for f in sorted(glob.glob(f"{PQ}/pred_test_{a.tag}/*.parquet"))])
tc = {k: float(v) for k, v in (kv.split("=") for kv in a.t_country.split(",") if kv)}
thr = pl.col("country").replace_strict(tc, default=a.t, return_dtype=pl.Float64) if tc else pl.lit(a.t)
links = best.filter(pl.col("prob") >= thr)
if a.rule:
    import rules
    B = rules.test_frame(a.tag)
    links, rule_stats = rules.decide(B, thr, **rules.RULESETS[a.rule]); del B
    print("rule", a.rule, rule_stats, flush=True)
    links = links.join(best.select("rid", "country"), on="rid")
if a.empty_country:
    links = links.filter(pl.col("country") != a.empty_country)
links = links.select("rid", "s1")
grp = links.sort("rid").group_by("s1").agg(pl.col("rid").str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
M = s1.join(grp, on="source1_entity_id", how="left")
M.write_csv(os.path.join(vd, "matching_results.tsv"), separator="\t", null_value="", quote_style="never")

# ---- candidate_pairs.tsv (cached per candidate-set version, hard-linked into each version)
cc = os.path.join(CACHE, f"candidate_pairs_{a.cand}.tsv")
if not os.path.exists(cc):
    # streamed in S1 hash buckets: never holds all ~220M candidate pairs in memory
    NB = 8
    files = sorted(glob.glob(f"{PQ}/test_cand/*.parquet"))
    with open(cc + ".tmp", "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for b in range(NB):
            inb = pl.col("s1").hash(seed=0) % NB == b
            G = (pl.scan_parquet(files).select("rid", "s1").filter(inb).unique()
                 .group_by("s1").agg(pl.col("rid").sort().str.join(",").alias("c")).collect())
            ids = s1.filter(pl.col("source1_entity_id").hash(seed=0) % NB == b).rename({"source1_entity_id": "s1"})
            B = ids.join(G, on="s1", how="left")
            for sid, c in B.iter_rows():
                out.write(f"{sid}\t{c or ''}\n")
            print("candidate bucket", b, B.height, flush=True)
            del G, B
    os.replace(cc + ".tmp", cc)
try:
    os.link(cc, os.path.join(vd, "candidate_pairs.tsv"))
except OSError:
    shutil.copyfile(cc, os.path.join(vd, "candidate_pairs.tsv"))

# ---- validation: official script on the matching file (+ID existence), streamed checker on the big candidate file
v = subprocess.run([sys.executable, os.path.join(ROOT, "student_resource", "utils", "validate_submission.py"),
                    "--matching", os.path.join(vd, "matching_results.tsv"), "--candidate", os.path.join(vd, "_skip_"),
                    "--test-dir", TEST_DIR, "--check-ids"], capture_output=True, text=True)
# every version is cut from predictions over the same cached candidate set, so its matches are candidates by
# construction; the (slow) streamed candidate check runs once per candidate set and is cached.
chk = os.path.join(CACHE, f"candidate_check_{a.cand}.txt")
if os.path.exists(chk) and "PASS" in open(chk, encoding="utf-8").read():
    c = subprocess.CompletedProcess([], 0, stdout=f"cached: {open(chk, encoding='utf-8').read()}", stderr="")
else:
    c = subprocess.run([sys.executable, os.path.join(ROOT, "eda", "20_check_candidates.py"), os.path.join(vd, "candidate_pairs.tsv"),
                        os.path.join(vd, "matching_results.tsv")], capture_output=True, text=True)
    open(chk, "w", encoding="utf-8").write(c.stdout + c.stderr)
open(os.path.join(vd, "validator.txt"), "w", encoding="utf-8").write(v.stdout + v.stderr + "\n--- candidate check ---\n" + c.stdout + c.stderr)
status = "PASS" if v.returncode == 0 and c.returncode == 0 else "FAIL"

# ---- local validation score of the same model/threshold (fold-0 S1s of train)
val = {}
vt = a.val_tag or a.tag
if os.path.exists(f"{PQ}/val_pred_{vt}.parquet"):
    vp = pl.read_parquet(f"{PQ}/val_pred_{vt}.parquet"); gt = gt_pairs().select("s1", "rid")
    ts1 = pl.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
    ts1 = ts1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64).mod(40) == 0)
    have = vp.join(ts1.rename({"entity_id": "s1"}), on="s1").select("country").unique()["country"].to_list()
    for c in have:
        val[c] = round(macro_f05(assign(vp, a.t), gt, ts1.filter(pl.col("country") == c)["entity_id"])[0], 5)
    val["testlike"] = round(macro_f05_testlike(assign(vp, a.t), gt, ts1.filter(pl.col("country").is_in(have))["entity_id"])[0], 5)

stats = dict(n_s1=M.height, n_s1_with_match=int(M["matched_entity_ids"].is_not_null().sum()), n_links=links.height,
             n_queries=best.height)
git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
man = dict(version=ver, name=name, created=datetime.datetime.now().isoformat(timespec="seconds"), model=f"lgb_{a.tag}",
           threshold=a.t, threshold_by_country=a.t_country, rule=a.rule, candidates=a.cand, note=a.note, validator=status, local_val_f05=val, stats=stats, git_head=git)
json.dump(man, open(os.path.join(vd, "manifest.json"), "w"), indent=2)

idx = os.path.join(SUBS, "INDEX.md")
if not os.path.exists(idx):
    open(idx, "w").write("# Submission versions\n\nUpload `matching_results.tsv` from a folder; write the leaderboard score in the last column.\n\n"
                         "| version | folder | model | threshold | local val F0.5 | S1 with matches | links | validator | note | leaderboard |\n"
                         "|---|---|---|---|---|---|---|---|---|---|\n")
vs = ", ".join(f"{k} {v}" for k, v in val.items()) or "-"
open(idx, "a").write(f"| {ver} | {name} | lgb_{a.tag} | {a.t:g}{(' ' + a.t_country) if a.t_country else ''} | {vs} | {stats['n_s1_with_match']} | {stats['n_links']} | {status} | {a.note} | |\n")
print(json.dumps(man, indent=2)); print(v.stdout[-600:])
