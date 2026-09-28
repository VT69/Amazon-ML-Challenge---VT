# Entity resolution – working notes

Hardware target: laptop, 12 GB RAM, 2-core i3-7020U. Every step streams / chunks; nothing is O(N×M).

## Key data facts (train)
- S1 is deduplicated; **every S2/S3 record matches at most one S1** (0 duplicates in GT) → solve in reverse:
  each S2/S3 record retrieves top-K S1 candidates and is assigned to ≤ 1 S1.
- ~26% of S2/S3 records match nothing. They are **deliberate "sibling" hard negatives**: a copy of a real S1
  with a number changed by a little (145→134, 1-7-172→1-7-170) and/or an extra business word
  (enterprises, public, exports, industries, holdings, overseas, infratech, ventures, group).
  True matches instead *drop/truncate* numbers (A/904→A/04) and add noise words (shri, smt, center, services).
- Country agreement in true pairs = 100% → block within country. Test adds France (unseen): features are
  country-agnostic, normaliser handles French abbreviations (st→saint, r→rue, bd→boulevard) and
  région/département equivalence.
- State written differently per source (US S3 spells names, S1/S2 codes; India S1 names, S2 native script,
  S3 codes) → canonical `zs<code>` token matched on whole address components.
- India: ~17% of S2/S3 names are in native scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati…).

## Pipeline
1. `12_norm_all.py` – normaliser v2 (`norm2.py`): ascii fold, leetspeak fix, legal-suffix removal, consonant
   skeleton tokens, state canonicalisation, country-aware abbreviations, **learned transliteration map**
   (`17_translit.py`: 1,130 token alignments from training native-script pairs + 87 skeleton matches to the
   S1 vocabulary).
2. `13_candidates.py` – `retr.py` numba inverted index over S1 per country:
   - primary: IDF-cosine over hashed words of name-core + name-skeleton + address, top-20
     (rarest-first, skip df>5000 after the 5 rarest features)
   - fallback: char-trigram (no spaces) name index, top-10, only for queries with no address or top-1 < 0.6
   - ~22 candidates/query, ~1.3 ms/query, ~1.3 GB RAM.
   Sample recall (v2, before transliteration): US 0.992, India 0.974.
3. `14_build.py` / `feats.py` – ~85 pair features: IDF token overlaps (name words, skeleton, name trigrams,
   address words, numbers), rapidfuzz name scores, **number relations** (exact / truncated / near-changed /
   changed / added / missing, min numeric diff), extra/missing token counts, retrieval ranks/scores and
   within-query context (gap to best, rank). No country feature.
4. `16_te.py` – out-of-fold log-odds of name words present on only one side (folds 6..39).
5. `15_train.py` – LightGBM; each S2/S3 record → argmax S1 if prob ≥ t; t tuned for macro F0.5 per S1
   (`evalf.py` reproduces the competition metric incl. singletons).

## Validation protocol
fold = numeric id % 40 of the query's true S1 (or of the query id if unmatched). Val = fold 0 (all queries,
all candidates); train = 25% of queries in folds 1..5; token log-odds from folds 6..39.

## Results log
| date | model | scope | macro F0.5 | notes |
|---|---|---|---|---|
| 09-25 | india_v1 (fast LGB, no TE, no translit) | India val | 0.9824 @ t=0.03 | losses: 2311 FN not retrieved, 802 FN wrong S1, 221 FP |
| 09-26 | all_v1 (fast LGB, TE, translit) | US+India val | 0.9893 @ t=0.05 | US 0.9904, India 0.9877 (India up from 0.9824) |
| 09-26 | all_v1 t=0.05 / 0.2 / 0.5 | public LB | 0.904 / 0.936 / 0.955 | local val said the opposite order -> test distribution differs |

## Test-set shift (found 09-26)
- Test has 5.75 S2/S3 records per S1 vs 4.68 in train (US 5.76, India 5.82, France 5.53) -> ~40% decoys vs 26%.
- Test decoys come as multi-record fake businesses (sibling clusters, "twin" records with the same changed
  number), whereas train siblings are lone records. Most get p≈0; the harder ones (identical name, only a
  house number changed by a little and/or legal form swapped Inc↔PLLC, Co↔PC, Sci↔SAS) land at 0.05-0.5.
- Grey-zone links (0.05<=p<0.5): 0.05 per S1 on val (91-94% correct) vs 0.20-0.59 per S1 on test.
- Counting val decoys twice ("test-like" score) barely moves the local score: the model already rejects
  train-style siblings; the test-only kind is what hurts. Local validation cannot rank thresholds for test;
  the leaderboard is the only reliable signal for decision thresholds.
| 09-26 | all_v1 t=0.7 / 0.8 / 0.9 | public LB | 0.961 / 0.963 / 0.961 | threshold curve peaks at t=0.8; thresholds alone top out ~0.963 |
| 09-26 | all_v1 t=0.8, France 0.9 / 0.7 | public LB | 0.963 / 0.962 | France threshold is not the lever; LB rank-50 cutoff = 0.985 |
| 09-26 | all_v2 (synthetic sibling clusters + group feats, full LGB) | test-like val (fold 0 + 41.9k synthetic look-alikes) | 0.9827 @ t=0.4 (v1 on same set: 0.9329 @ 0.4, 0.9442 @ 0.8) | synthetic look-alikes linked @0.8: v1 11,209 -> v2 274 |
| 09-26 | all_v2 t=0.3/0.4/0.5/0.7/0.8/0.9 | public LB | 0.939/0.945/0.949/0.956/0.958/0.958 | WORSE than v1 (0.963): synthetic decoys + group feats do not match the real test; test-like val misled |
| 09-27 | all_v3 (full-quality LGB, v1 features) t=0.8 France 0.9 | public LB | 0.96423 | FINAL/BEST. v017 (cluster rules) 0.96214 |
