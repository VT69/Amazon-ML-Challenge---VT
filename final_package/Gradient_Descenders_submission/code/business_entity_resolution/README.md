# Business Entity Resolution – Team Gradient Descenders

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` (submitted version **v019**, public
leaderboard F0.5 = **0.96423**, our best) from the official training and test data.

Designed for a modest laptop (12 GB RAM, 2-core Intel i3). Every step streams or chunks its data, and
no step compares all S1 records with all S2/S3 records.

## Setup

```bash
pip install -r requirements.txt
```

Folders (override with environment variables):

| Variable | Default | Content |
|---|---|---|
| `ER_DATA` | `./dataset` | the official `train/` and `test/` folders with the `.tsv` files |
| `ER_WORK` | `./work` | intermediate parquet files, model, predictions (~15 GB at peak) |
| `ER_OUT` | `./output` | final `matching_results.tsv` and `candidate_pairs.tsv` |

## Full pipeline (data → blocking → matching → output)

Run from this folder, in order. The times are measured on the 2-core laptop above.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python src/01_convert.py` | official TSVs → parquet; ground truth → (S1, S2/S3) pairs | 5 min |
| 2 | `python src/02_translit.py` | learns the native-script → English transliteration map from **training** pairs only (+ an unsupervised skeleton layer) | 15 min |
| 3 | `python src/03_normalize.py train` then `python src/03_normalize.py test` | normaliser v2 for every record (names, addresses, states, skeletons, transliteration) | 2 × 20 min |
| 4 | `python src/04_candidates.py train` then `python src/04_candidates.py test` | blocking: numba inverted-index top-K retrieval, S2/S3 record → S1 of the same country | 3.5 h + 2.5 h |
| 5 | `python src/05_features.py` | pair features for the validation fold (0) and a 25% query sample of training folds 1–5 | 1 h |
| 6 | `python src/06_token_logodds.py` | out-of-fold name-token log-odds table (folds 6–39) + per-pair features | 20 min |
| 7a | `FAST=1 python src/07_train.py all_v1` | fast LightGBM (63 leaves, lr 0.15); used only as a pre-filter in 8b | 1 h |
| 7b | `FAST=0 python src/07_train.py all_v3` | **submitted model**: full-quality LightGBM (127 leaves, lr 0.07, 450 trees); prints the validation macro F0.5 sweep | 30 min |
| 8a | `python src/08_predict.py test all_v1` | scores every test candidate pair with all_v1 (streamed in sub-batches) | 2.5 h |
| 8b | `PREFILTER_TAG=all_v1 PREFILTER_LIST=artifacts/v019_prefiltered_chunks.txt python src/08_predict.py test all_v3` | scores the test pairs with all_v3. For the 22 chunks in the list, the model only runs on pairs where all_v1 gave ≥ 0.001 (the rest get 0; they cannot reach the 0.8 threshold). Features are always computed on the full candidate list. Exactly as submitted | 2.5 h |
| 9 | `python src/09_make_output.py` | decision rule (best S1 per record, threshold 0.8; France 0.9; model all_v3) → both output files + rule checks | 20 min |

On Windows PowerShell, set the variables first (e.g. `$env:FAST="1"`) instead of using the `VAR=value` prefixes.

### Shortcut: reuse the trained artefacts
`artifacts/` contains the exact files used for the submission: the models `lgb_all_v3.txt` (submitted) and
`lgb_all_v1.txt` (pre-filter), the transliteration maps, the token log-odds table and the list of pre-filtered
chunks. To reproduce the output without retraining, run step 1, then copy `translit.parquet`,
`translit_full.parquet` and `te_table.parquet` into `ER_WORK`, and both `lgb_*.txt` into `ER_WORK/models/`.
Then run steps 3, 4 (test only), 8a, 8b and 9.

### Validation
- Step 7 reports macro F0.5 per Source-1 entity (singletons included, exactly as the competition defines it)
  on S1-disjoint validation fold 0.
- `src/10_check_candidates.py` checks `candidate_pairs.tsv` against every submission rule, line by line. The
  official validator runs out of memory on the 2.9 GB file, so this check is streamed.
- For `matching_results.tsv`, also run the official validator:
  `python <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv --test-dir <data>/test --check-ids`

## Source layout (`src/`)

| File | Role |
|---|---|
| `common.py` | paths and loaders |
| `norm2.py` | normaliser v2: folding, leetspeak, legal suffixes, skeletons, state canonicalisation, country-aware abbreviations, transliteration |
| `retr.py` | numba inverted index: integer features (char trigrams, hashed words), IDF cosine, rarest-first top-K with df cap |
| `feats.py` | pair features: IDF token overlaps, RapidFuzz scores, number relations, within-query context |
| `te.py` | out-of-fold name-token log-odds features |
| `evalf.py` | competition metric (macro F0.5 per S1) and decision rule |
| `group.py` | group-consistency features (used only by an experimental model, not by the submitted one) |
| `01`…`10` | pipeline steps (see table above) |
