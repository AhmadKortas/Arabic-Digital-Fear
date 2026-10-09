# Digital Fear in Arabic Cybersecurity Discourse — Engagement & Agreement Analysis

Analysis code for the RQ2 and RQ3 sections of our study of *digital fear* in
Arabic cybersecurity discourse on Facebook. One script reproduces the fear-level
statistics, the engagement analysis, and the annotator-agreement evaluation,
along with the three figures used in the paper.

## What the script does

**RQ2 — prevalence and reception of digital fear**
- Reports how the annotated sample splits across the three fear levels
  (No Fear / Moderate Fear / High Fear).
- Compares engagement (likes, comments, shares, total interactions) across
  fear levels using medians and means, and tests the differences with a
  Kruskal–Wallis test (suited to skewed count data).
- Fits a negative binomial regression for shares and total interactions that
  **controls for content type, page category, and page size (followers)**, so
  the share effect isn't confused with format or reach. Results are reported as
  incidence-rate ratios relative to No Fear.

**RQ3 — reliability of FANAR as an annotator**
- Compares FANAR's labels against a human annotator on the validation set.
- Reports accuracy, Cohen's kappa (plain, linear-weighted, quadratic-weighted),
  per-class precision/recall/F1, and the size of each disagreement.
- Draws the confusion matrix.

## Figures produced (saved to `./figs`)
- `fig_fear_distribution.png` — fear-level split of the annotated sample
- `fig_engagement_by_fear.png` — mean engagement by fear level
- `fig_confusion_matrix.png` — FANAR vs. human labels

## Data

Two CSV files are expected (set the paths at the top of the script):
- the FANAR-annotated sample — post text, `llm_label`, and CrowdTangle metadata
  (`Total Interactions`, `Likes`, `Comments`, `Shares`, `Followers at Posting`,
  `Type`, `Page Category`)
- the human validation sample — `post_id`, `llm_label`, `Manual label`

The data are not included in this repository.

## Running it

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib
python digital_fear_engagement_and_agreement.py
```

All results print to the console; the figures are written to `./figs`.

## Note on CrowdTangle counts

CrowdTangle exports large numbers with thousands separators (e.g. `49,695`).
The script strips these before converting to numeric, so the most-engaged posts
are not accidentally dropped — worth keeping in mind if you adapt the code for
other CrowdTangle columns.
