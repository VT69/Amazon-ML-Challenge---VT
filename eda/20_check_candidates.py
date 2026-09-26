"""Streaming validator for a large candidate_pairs.tsv (the official validator holds every list in memory and
runs out of RAM on ~3 GB). Applies the same rules line by line:
header, one row per test S1 (no missing / duplicate / unknown rows), no repeated ID within a list, only S2-/S3-
prefixed IDs, IDs exist in test (optional), and every matched ID of the given matching files appears among the
candidates of the same S1.   python 20_check_candidates.py <candidate_pairs.tsv> [matching_results.tsv ...]"""
import sys, csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "student_resource", "dataset", "test")
cand_path, match_paths = sys.argv[1], sys.argv[2:]


def ids_of(fn):
    with open(os.path.join(TEST, fn), encoding="utf-8", newline="") as f:
        next(f)
        return {line.split("\t", 1)[0] for line in f if line.strip()}


s1 = ids_of("test_source1.tsv")
valid = ids_of("test_source2.tsv") | ids_of("test_source3.tsv")
matched = {}                                   # s1 -> matched ids (only non-empty rows)
for mp in match_paths:
    with open(mp, encoding="utf-8", newline="") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if b:
                matched.setdefault(a, set()).update(b.split(","))

errors, seen, n_rows, n_empty, n_ids = [], set(), 0, 0, 0
missing_match = 0
with open(cand_path, encoding="utf-8", newline="") as f:
    header = f.readline().rstrip("\n")
    if header != "source1_entity_id\tcandidate_entity_ids":
        errors.append(f"bad header: {header!r}")
    for ln, line in enumerate(f, start=2):
        a, tab, b = line.rstrip("\n").partition("\t")
        if not tab:
            errors.append(f"line {ln}: no tab"); continue
        n_rows += 1
        if a in seen: errors.append(f"line {ln}: duplicate S1 row {a}")
        seen.add(a)
        if a not in s1: errors.append(f"line {ln}: unknown S1 {a}")
        ids = b.split(",") if b else []
        if not ids:
            n_empty += 1
        n_ids += len(ids)
        st = set(ids)
        if len(st) != len(ids): errors.append(f"line {ln}: repeated id in list for {a}")
        bad = [i for i in st if not i.startswith(("S2-", "S3-")) or i not in valid]
        if bad: errors.append(f"line {ln}: invalid ids for {a}: {bad[:3]}")
        if a in matched and not matched[a] <= st:
            missing_match += 1
        if len(errors) > 20: break
missing_rows = len(s1 - seen)
if missing_rows: errors.append(f"{missing_rows} test S1 entities missing from candidate file")
if missing_match: errors.append(f"{missing_match} S1 rows have matched ids that are not among their candidates")
print(f"rows={n_rows} empty={n_empty} candidate_ids={n_ids} avg_per_S1={n_ids / max(n_rows, 1):.1f}")
print("\n".join(errors) if errors else "PASS: candidate_pairs.tsv follows every rule; all matches are subsets of candidates")
sys.exit(1 if errors else 0)
