# ML Challenge 2026: Business Entity Resolution — Team Gradient Descenders

**Team Name:** Gradient Descenders  
**Rank:** **1020** out of 90,000

**Team Members:** Vaibhav Tiwari, Mohit Upadhyay, Kavish Bishnoi, Harshit Gupta  
**Submission Date:** September 25 – 27, 2026

---

## 1. Executive Summary
Every S2/S3 record matches at most one S1 entity. So we solve the problem in reverse: each S2/S3 record
retrieves its top candidates from an inverted index over the S1 records of its own country, a LightGBM pair
classifier scores them, and the record is linked to its single best S1 if the probability clears a threshold.

The key techniques are:
- a normaliser that unifies states, abbreviations and native-script Indian names (the transliteration dictionary is learned from the training pairs)
- a custom numba top-K retriever that fits a 12 GB, 2-core laptop
- house-number relation features that separate true matches (numbers dropped or truncated) from the planted look-alike businesses (numbers changed slightly)

The submitted version (v019) scores **F0.5 = 0.96423** on the public leaderboard (our best of 17 submissions).

---

## Repository layout

| Path | What it is |
|---|---|
| `final_package/Gradient_Descenders_submission/` | the final submission package (without the multi-GB `output/` files): `code/business_entity_resolution/` (runnable pipeline `src/`, `README.md` with exact reproduction steps, pinned `requirements.txt`, trained models in `artifacts/`) and this write-up as `Documentation_template.md` |
| `eda/` | exploration and development scripts (`00`…`22_*.py`), shared modules (`norm2.py`, `retr.py`, `feats.py`, `te.py`, `group.py`, `rules.py`, `evalf.py`), run logs in `eda/out/`, working notes in `eda/NOTES.md`, team summary in `eda/TEAM_SUMMARY.md` |
| `submissions/` *(local only, git-ignored)* | all 19 submission versions (17 uploaded) with manifests, validator output and `INDEX.md` (public leaderboard score of each) |
| `student_resource/` | official problem statement and submission validator (dataset git-ignored) |

Large data (parquet caches, candidate sets, predictions, the 2.9 GB `candidate_pairs.tsv` and the 1.18 GB submission zip) is
kept out of git; everything can be regenerated with the pipeline in `final_package/.../code/business_entity_resolution/`.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploratory analysis of the 12.5M training records:

- **At most one S1 per S2/S3 record.** No S2/S3 id appears twice in the ground truth. S1 entities have 3.46
  matches on average (up to 5 from S2 and 6 from S3), and 5.6% of S1 are singletons. About 26% of S2/S3
  records match nothing. True pairs always share the country.
- **Unmatched records are deliberate "sibling" look-alikes.** Each is a copy of a real S1 business (91% of
  the targeted S1s have true matches) with small edits:
  - a house number changed by a little (145→134, 1-7-172→1-7-170, 1409→1408)
  - and/or an added business word (enterprises, public, exports, industries, holdings, overseas, infratech, ventures, group)

  True matches instead *drop or truncate* digits (A/904→A/04) and add noise words (shri, smt, center, services).

  | Top candidate pair | True matches | Siblings |
  |---|---|---|
  | query numbers are a subset of the S1's | 67% | 7% |
  | a number changed by ≤ 20 | 3% | 50% |
  | query name has a word the S1 lacks | 50% | 93% |
- **Name noise:**
  - abbreviations and legal-suffix changes (22% of true pairs)
  - punctuation and diacritics (14%)
  - word reordering (7%)
  - domain names like `mccutcheonsdental.com` (5%)
  - "d/b/a" / "formerly" aliases (3%)
  - leetspeak (`Channe1`, `H0LDINGS`)
  - native-script names in India (about 17% of Indian S2/S3 names, in Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Odia and Gurmukhi)
- **Address noise:**
  - component reordering and truncation
  - "null" / "N/A" placeholders
  - mojibake (`Â`)
  - each source writes the state differently: US S1/S2 use codes and S3 full names; India S1 uses English names, S2 native script, S3 codes (MH, TG, KA, …)
  - about 3.4% of S2/S3 addresses are missing
- **France (test only):** 3 regions, about 18 cities; sources swap région and département and write
  `St`/`Saint`, `R`/`Rue`, `BD`/`Boulevard`.

### 2.2 Solution Strategy
**Approach Type:** Blocking (inverted-index top-K retrieval) + gradient-boosted pair classifier + per-record argmax decision  
**Core Innovation:**
1. **Reverse formulation.** An index over S1 only (small), queried by every S2/S3 record. This gives linear time,
   constant memory, and each record naturally links to at most one S1.
2. **Learned transliteration.** A dictionary aligned from native-script ↔ English training pairs, plus an
   unsupervised consonant-skeleton match to the S1 vocabulary.
3. **Number-relation features** that distinguish digit truncation (typical of true matches) from small digit
   changes (the signature of planted look-alikes).
4. **Country-agnostic features only.** There is no country feature, so France, which is absent from training,
   is not out-of-distribution.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** retrieval over integer features rather than exact keys:
  - *Primary index:* FNV-hashed words of the normalised name core, its consonant skeleton tokens and the
    normalised address, scored by IDF cosine. The top 20 S1 candidates are kept. Features are processed
    rarest-first, and features with document frequency > 5,000 are skipped once the 5 rarest have been used.
  - *Fallback index:* exact character trigrams of the space-free name core, top 10, used only when the record
    has no address or the primary top-1 score is < 0.6 (about 20–30% of records).
  - Always restricted to the record's country (true pairs never cross countries).
- **Candidate pairs generated:** 220,686,712 on the test set (22.1 per S2/S3 record, 127.4 per S1). The full
  within-country cross product is about 6.7 × 10¹², so the reduction ratio is 99.997%.
- **How we ensured true matches were not lost:**
  - The retrieval design was chosen by measuring recall on samples against the *full* S1 index. Name-only
    retrieval peaked at 88% recall@50; name+address words gave 99%.
  - The misses were analysed and fixed one by one:
    - state canonicalisation on whole address components (so `CT` ≠ `court`)
    - consonant skeletons for typos and transliteration noise
    - the name-trigram fallback for missing addresses and concatenated domain names
    - the learned transliteration map, which raised India's recall from 97.0% to 97.9%
  - Final blocking recall on validation fold 0: **US 99.19%, India 97.87%, overall 98.67%** of true pairs retrieved.
  - Cost: about 1.3 ms per record and about 1.3 GB of RAM.

---

## 4. Matching Model

**Features used (87 in total):**
- **Name features:**
  - IDF-weighted overlap of name words, consonant skeletons and name trigrams: query coverage, S1 coverage,
    cosine, shared count, extra and missing token counts
  - RapidFuzz `token_set_ratio`, `ratio`, `partial_ratio`, Jaro-Winkler on the name core
  - `ratio` / `partial_ratio` on the space-free name (for domain names)
  - exact name core / skeleton / space-free equality
  - out-of-fold **token log-odds** for words present in only one of the two names (min, sum, count known).
    This is learned on folds 6–39, disjoint from the training and validation folds.
  - flags: raw name in native script, contains a domain, contains an alias keyword; raw name lengths
  - S1 name ambiguity: how many S1 records share the name core
- **Address features:**
  - IDF-weighted overlap of address words: coverage both ways, cosine, shared, extra and missing
  - state token equality, missing-address flag
  - S1 address ambiguity
- **House-number relations** (numba):
  - query numbers that are exact / **truncated** (prefix or suffix) / **changed by ≤ 20** / changed more / added
  - S1 numbers missing from the query
  - minimum numeric difference of changed numbers
  - first- and largest-number equality
- **Other:**
  - retrieval scores and ranks from both indexes, and which index found the candidate
  - within-record context: each candidate's gap to the best candidate and its rank among the record's candidates, for 7 key scores
  - number of candidates
  - source (S2 vs S3)

**Model type:** LightGBM binary classifier (MIT licence). Submitted model `all_v3`: 127 leaves, learning rate 0.07,
max_bin 127, feature/bagging fraction 0.8/0.5, L2 1.0, early stopping on validation logloss (best iteration 450,
validation logloss 0.00405 vs 0.00522 for the earlier fast model `all_v1` with 63 leaves / lr 0.15, which is kept only
as a test-time pre-filter: for 22 of 42 test chunks, all_v3 is evaluated only on pairs where all_v1 ≥ 0.001; such
pairs cannot reach the 0.8 threshold). Trained on 7.25M candidate pairs: a 25% query
sample of S1 folds 1–5, all of each sampled record's candidates. No external data or pretrained models.  
**Threshold selection method:**
- Each S2/S3 record is linked to its highest-probability S1 candidate if the probability ≥ t.
- On local validation (macro F0.5 per S1, singletons included, S1-disjoint fold 0), t = 0.05 was best (0.989).
- On the public leaderboard, scores rose with t (see §5): the test set holds about 50% more look-alike records
  per business than training.
- The submitted setting is **t = 0.8 for US and India, 0.9 for France**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - local validation 0.989 (US 0.990, India 0.988) at t = 0.05
  - 0.9807 at the submitted t = 0.8 for the submitted model all_v3 (0.9800 for all_v1)
  - **public leaderboard 0.96423** (v019, best); 0.963 for the fast model (v005 / v007)

| Version | Model | Threshold | Local val F0.5 | Public LB |
|---|---|---|---|---|
| v001 | LGB v1 | 0.05 | 0.989 | 0.904 |
| v002 | LGB v1 | 0.2 | 0.989 | 0.936 |
| v003 | LGB v1 | 0.5 | 0.986 | 0.955 |
| v004 | LGB v1 | 0.7 | 0.982 | 0.961 |
| v005 | LGB v1 | 0.8 | 0.980 | 0.963 |
| v006 | LGB v1 | 0.9 | 0.973 | 0.961 |
| v007 | LGB v1 | 0.8 (France 0.9) | 0.980 | 0.963 |
| v008 | LGB v1 | 0.8 (France 0.7) | 0.980 | 0.962 |
| v009–v014 | LGB v2 (synthetic look-alikes + group features) | 0.3–0.9 | 0.978–0.983 (test-like val) | 0.939–0.958 |
| v017 | LGB v1 + cluster-level decision rules | 0.8 (France 0.9) | 0.980 | 0.962 |
| **v019** | **LGB v3 (full quality, v1 features)** | **0.8 (France 0.9)** | **0.981** | **0.96423** |

- **Common false positives (wrong merges):**
  - planted look-alike businesses that copy the S1 name exactly and change only the house number slightly
    (9399 vs 9398 Kellogg Creek Dr) or swap the legal form (Inc ↔ PLLC, SCI ↔ SAS)
  - different businesses at the same address (e.g. "Nantes Culturelle" vs "Nantes Ecole")

  In test, look-alikes come as multi-record fake businesses (5.75 S2/S3 records per S1 vs 4.68 in train), and
  low-probability links (0.05–0.5) were 4–12× more frequent than on validation. This is why stricter thresholds
  won on the leaderboard.
- **Common false negatives (missed matches):**
  - Validation India at t = 0.03 (before the transliteration fix): 2,311 true pairs not retrieved, 802
    assigned to a different look-alike S1, 145 below threshold.
  - Retrieval misses were mostly native-script names with truncated addresses (fixed in part by the
    transliteration map), records with no address whose name is shared by many S1s, and heavily
    truncated addresses combined with a changed name.

---

## 6. Conclusion
A reverse-direction retrieve-and-classify pipeline reaches 0.989 macro F0.5 on S1-disjoint validation and
0.964 on the public test within 12 GB of RAM:
- IDF top-K blocking with 98.7% recall
- 87 similarity and number-relation features
- a LightGBM matcher with a per-record argmax decision

The main lesson is that the test set contains a harder, more numerous kind of look-alike record than training.
Local validation therefore favoured low thresholds while the leaderboard rewarded high ones. We learned that
leaderboard feedback on the decision threshold is essential, and that synthetic look-alikes must mirror the
test distribution closely: our synthetic-sibling model (v2) improved test-like validation but not the leaderboard.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (full instructions in its `README.md`, pinned versions in `requirements.txt`):

- `src/01_convert.py` → `src/02_translit.py` → `src/03_normalize.py` (train, test) → `src/04_candidates.py` (train, test)
  → `src/05_features.py` → `src/06_token_logodds.py` → `src/07_train.py all_v1` (FAST=1) and `src/07_train.py all_v3`
  (FAST=0) → `src/08_predict.py test all_v1` → `src/08_predict.py test all_v3` (pre-filtered by all_v1 on the chunks
  listed in `artifacts/v019_prefiltered_chunks.txt`) → `src/09_make_output.py`. This last step writes
  `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Modules: `norm2.py` (normalisation), `retr.py` (numba inverted-index retrieval), `feats.py` (pair features),
  `te.py` (token log-odds), `evalf.py` (competition metric), `10_check_candidates.py` (streamed rule checks).
- `artifacts/` holds the submitted model (`lgb_all_v3.txt`), the pre-filter model (`lgb_all_v1.txt`), the
  transliteration maps, the token log-odds table and the pre-filtered chunk list, so the output can be reproduced
  exactly without retraining.
- End-to-end runtime on a 12 GB / 2-core laptop is about 12 hours, dominated by candidate retrieval and test scoring.

### B. Additional Results
Blocking alternatives measured on 30k-query samples against the full US S1 index (recall of true pairs):

| Retriever | R@1 | R@10 | R@50 | ms/query |
|---|---|---|---|---|
| Name trigrams (uncapped) | 0.54 | 0.78 | 0.88 | 11.0 |
| Address trigrams (uncapped) | 0.81 | 0.92 | 0.94 | 18.7 |
| Name + address words (df cap 5000) | 0.93 | 0.98 | 0.99 | 0.25 |
| + skeleton tokens, state canonicalisation (v2) | 0.95 | 0.98 | 0.99 | 0.45 |

Most important features (gain): candidate retrieval rank, minimum changed-number difference, gap to the
record's best retrieval score, out-of-fold extra-token log-odds, address-coverage gap, shared address tokens.
