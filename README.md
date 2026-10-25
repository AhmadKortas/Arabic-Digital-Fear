# Digital Fear in Arabic Cybersecurity Discourse — Analysis Code

Analysis code for our study of *digital fear* in Arabic cybersecurity discourse
on Facebook. The scripts reproduce the descriptive and engagement results (RQ1),
the fear-level and engagement analysis (RQ2), and the annotator-agreement
evaluation (RQ3), along with the figures used in the paper.

## RQ1 — Characterizing the discourse (`rq1_analysis.py`)

This script covers the whole RQ1 section. It loads the two datasets — the broad
Arabic cybersecurity set (DS1) and the filtered cyber threat intelligence set
(DS2) — and reports, for each one, the descriptive and engagement statistics that
appear in the paper: number of posts, unique pages, date range, median message
length, share of posts containing a URL, mean and median engagement (likes,
comments, shares, total interactions), the content-type and page-category
breakdowns, and reaction-based audience sentiment. It then writes six figures to
a `figs/` folder:

- temporal distribution of posts (2014–2024)
- content-type distribution across the two datasets
- average engagement by content type
- reaction-based audience sentiment
- posting activity by day of week
- geographic distribution by administering country (top 10 per dataset)

Each figure is saved as both PDF (for the paper) and PNG.

A small but important detail: CrowdTangle exports large counts with thousands
separators (e.g. `49,695`). The loader strips these before converting to numeric,
otherwise the most-viral posts turn into missing values and the engagement
averages come out too low. This is why total interactions reconcile exactly with
the paper's Table 3.

## RQ2 and RQ3 — Fear prevalence, engagement, and annotator agreement

`digital_fear_engagement_and_agreement.py` reproduces the fear-level
distribution, the engagement-by-fear-level comparison with the Kruskal–Wallis
test, the negative binomial model that adjusts for content type, page category
and page size, and the FANAR-vs-human agreement metrics (accuracy, Cohen's kappa,
per-class scores, confusion matrix). See the comments in that file for details.

## Data

The scripts expect the dataset CSVs in the same folder (paths are set at the top
of each script):

- the broad Arabic cybersecurity dataset (DS1)
- the filtered cyber threat intelligence dataset (DS2)
- for RQ2/RQ3, the FANAR-annotated sample and the human validation sample

The data are not included in this repository.

## Running it

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib
python rq1_analysis.py
python digital_fear_engagement_and_agreement.py
```

All statistics print to the console; figures are written to `figs/`.
