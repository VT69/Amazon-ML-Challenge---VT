**Approach so far (Amazon ML Challenge 2026 – business entity resolution)**

We solve it "in reverse": every S2/S3 record matches at most one S1, so each S2/S3 record retrieves its top candidates from an S1 index of its own country and is linked to its single best S1 if the model's probability clears a threshold. Everything runs on a 12 GB / dual-core laptop, with no pairwise all-vs-all comparisons.

**Normalisation:**
- ASCII folding, leetspeak repair (Channe1 → channel), legal-suffix removal
- consonant "skeleton" tokens (praaivett / private → prvt)
- state canonicalisation across forms: NY / New York, MH / Maharashtra / महाराष्ट्र
- country-aware address abbreviations, incl. French st → saint, r → rue
- a transliteration dictionary for native-script Indian names (Devanagari, Telugu, Tamil, …), learned from the training pairs: 1,217 word mappings, e.g. मॉडर्न स्काई → modern sky

**Blocking:** a custom numba inverted index (IDF cosine) over name + skeleton + address words, top-20 per record. A character-trigram name index adds up to 10 more when a record has no address or a weak top hit. That gives about 22 candidates per record at about 1 ms per record, with about 97–99% of true pairs retrieved.

**Features (~95):**
- IDF-weighted token overlaps: name words, skeletons, name trigrams, address words, numbers
- RapidFuzz name similarities
- house-number relations: exact / truncated / slightly changed / added / missing, and the minimum numeric difference
- extra and missing word counts
- out-of-fold log-odds of words present on only one side (catches look-alike siblings that add "Enterprises / Exports / Holdings…")
- retrieval scores and ranks, plus how each candidate compares with the record's other candidates
- no country feature, so France (test-only) isn't out-of-distribution

**Model:** LightGBM binary classifier. The decision is argmax S1 per record plus a threshold. Validation uses S1-disjoint folds and the exact competition metric (macro F0.5 per S1, singletons included).

**Results:**

| Version | Threshold | Local val F0.5 | Public LB |
|---|---|---|---|
| v1 | 0.05 | 0.989 | 0.904 |
| v2 | 0.2 | 0.989 | 0.936 |
| v3 | 0.5 | 0.986 | 0.955 |
| v4 | 0.7 | 0.982 | 0.961 |
| v5 | 0.8 | 0.980 | **0.963 (best)** |
| v6 | 0.9 | 0.973 | 0.961 |

**Key insight:** test contains about 50% more decoy records per business than train. They're multi-record look-alike businesses that copy the name and change only the house number by a little, whereas train decoys are lone records. That's why stricter thresholds win on the leaderboard while local validation prefers loose ones.

**Next (v7+):** France-specific thresholds; synthetic test-style sibling clusters in training; group-consistency features (does a record agree with the S1's own version, or with a rival cluster?); full-quality retraining.
