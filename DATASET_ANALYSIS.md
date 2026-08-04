# ASL Citizen — Dataset Analysis

## What the dataset is
ASL Citizen is a crowdsourced isolated-sign dataset released by Microsoft Research, containing roughly 84,000 video clips covering about 2,700 distinct ASL signs performed by 52 signers. Each clip shows one signer performing one sign in front of an everyday RGB camera, framed as a dictionary-lookup task rather than continuous sentence translation. It was collected under IRB approval with consent from each contributor. This analysis was run on a stratified 8.5% sample (7,129 clips) drawn proportionally by split and gloss to keep the sample's class balance representative of the full dataset.

## Key statistics

| Metric | Value |
|---|---|
| Total clips (sample) | 7,129 |
| Unique signs (gloss vocabulary) | 2,731 |
| Unique signers | 30 (in sample) / 52 (full dataset) |
| Train / Test / Val split sizes | 4,309 / 2,795 / 25 |
| Median clip duration | 2.67s |
| Dominant frame rate | ~30 fps (93.3% of clips) |
| Distinct resolutions observed | 2 (640×480, 960×540) |
| OOV rate (val/test vs train vocab) | 0.00% |
| Signer overlap across splits | 0 signers shared |

**Figure 1 — Duration distribution:** right-skewed, most clips fall between 1.5–4 seconds with a peak near 2–3s, and a long tail extending to ~9s.

**Figure 2 — FPS and resolution:** frame rate is dominated by a single cluster at ~30fps, with a smaller secondary cluster at ~24–25fps and a small group near 15fps. Resolution is essentially binary — nearly every clip is one of exactly two fixed camera resolutions, with no continuous variation.

**Figure 3 — Clips per signer:** highly uneven. The top signer (P33) contributes roughly 950 clips in the sample while the bottom several signers contribute fewer than 20 each — a long-tailed, imbalanced signer distribution.

## Data quality issues found

- **Unreadable/missing/corrupt files:** 0 out of 7,129 (0.0%)
- **Zero or negative duration clips:** 0
- **Extreme duration outliers (>99.5th percentile):** 36 clips
- **Vocabulary OOV rate (val/test vs train):** 0.00% — every gloss in val and test also appears in train
- **Signer overlap across splits:** 0 — train, val, and test share no signers in common. This is the check Vardhan flagged as critical, and the result is clean: the official splits are genuinely signer-independent, so accuracy numbers won't be inflated by the model having seen a signer's individual style during training.
- **Class frequency:** with 2,731 signs across 7,129 clips, the most frequent glosses appear only 4–5 times each — the dataset is extremely long-tailed at the per-class level, even before considering the sample reduction.

## Implications for our experiments

- **Frame sampling will need to be generous, not sparse.** At ~30fps and a median duration of 2.67s, a typical clip is roughly 80 frames long.
- **The long duration tail (out to ~9s) means a fixed-length input window will either truncate slow signers or pad heavily for fast ones.** Padding/truncation strategy should be a config option we test, not a silent default.
- **Resolution is not a real source of variation** (only two fixed values), so any resizing/augmentation pipeline can standardize input size cheaply without worrying about extreme aspect-ratio edge cases.
- **Signer imbalance (P33's ~950 clips vs. others' <20) means naive accuracy will be dominated by how well the model handles a handful of prolific signers.** We should report per-signer accuracy breakdowns, not just an aggregate number, to catch this.
- **Val set is tiny (25 clips in sample, corresponding to a small slice of the real val split).** Any hyperparameter decisions tuned against val accuracy will be noisy; we may want to either enlarge val sampling specifically, or rely more on cross-validation-style checks than a single held-out val number.
- **Extreme class sparsity (most signs seen 4–5 times) makes the few-shot prompting condition (k in-context examples) genuinely hard** — for many signs there may not be enough examples in train to construct a k>3 few-shot prompt without reusing test-adjacent data. This should shape how we pick k.
- **Signer-independent splits (0 overlap) are a genuine strength of this dataset for our purposes** — we can trust that test accuracy reflects generalization to unseen signers, not memorized signing style.
