"""
Digital fear in Arabic cybersecurity discourse — engagement & annotator agreement.

This script reproduces the quantitative results and figures for two parts of the
paper:

  RQ2  How prevalent is digital fear, and do fear-level posts engage differently?
       - label distribution over the FANAR-annotated sample
       - engagement by fear level (medians, means, Kruskal-Wallis)
       - negative binomial regression that controls for content type, page
         category and page size, so the share effect isn't just a format artifact

  RQ3  How reliable is FANAR as an annotator?
       - FANAR vs. human agreement on the 1k validation set
       - accuracy, Cohen's kappa (plain + weighted), per-class P/R/F1
       - confusion matrix

Two input files are expected (paths set below):
  - the FANAR-annotated sample  (post text, llm_label, CrowdTangle metadata)
  - the human validation sample (post_id, llm_label, Manual label)

Figures are written to ./figs as PNG. Everything prints to stdout as it runs.

Author: Ahmad
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import kruskal
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, classification_report)
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ----------------------------------------------------------------------
# Paths and settings — edit these two to point at your own copies
# ----------------------------------------------------------------------
FANAR_FILE = "annotated_digitalfear_fanar.csv"
HUMAN_FILE = "manual_annotation_digitalfear_1k.csv"
FIG_DIR    = "figs"

# Fixed label order used everywhere (keeps tables and plots consistent)
LEVELS = ["No Fear", "Moderate Fear", "High Fear"]

# A calm three-colour palette: green = no fear, orange = moderate, red = high
COLORS = ["#55A868", "#DD8452", "#C44E52"]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.autolayout": True,
})


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def to_number(series):
    """Turn a text column into numbers.

    CrowdTangle writes large counts with thousands separators (e.g. "49,695"),
    so we strip the commas before converting. Without this step those commas
    silently become NaN and the biggest, most-viral posts drop out of every
    average — which is exactly the kind of bug that makes 'Likes' look larger
    than 'Total Interactions'.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(),
                         errors="coerce")


def load_fanar():
    """Load the FANAR-annotated sample and parse the numeric columns we need."""
    df = pd.read_csv(FANAR_FILE, dtype=str, keep_default_na=False)
    for col in ["Total Interactions", "Likes", "Comments", "Shares",
                "Followers at Posting"]:
        df[col] = to_number(df[col])
    return df


def load_human():
    """Load the human validation set, keeping only rows both sides labelled.

    One row in the sheet has an empty manual label; we drop it rather than
    guess, so the agreement numbers are computed on genuinely paired decisions.
    """
    df = pd.read_csv(HUMAN_FILE, dtype=str, keep_default_na=False)
    paired = df["llm_label"].isin(LEVELS) & df["Manual label"].isin(LEVELS)
    return df[paired].copy()


# ======================================================================
# RQ2 — prevalence and engagement
# ======================================================================
def fear_distribution(df):
    """Print how the three fear levels are split across the annotated sample."""
    print("\n--- Fear level distribution ---")
    n = len(df)
    for level in LEVELS:
        count = (df["llm_label"] == level).sum()
        print(f"  {level:<15} {count:>5}  ({100 * count / n:.1f}%)")
    print(f"  {'Total':<15} {n:>5}")


def engagement_by_level(df):
    """Compare engagement across fear levels and test whether it really differs.

    We look at both the median (the typical post) and the mean (which the rare
    viral posts pull upward), then run a Kruskal-Wallis test — the rank-based
    alternative to ANOVA, which suits these heavily skewed counts far better
    than anything assuming a normal distribution.
    """
    print("\n--- Engagement by fear level ---")
    metrics = ["Total Interactions", "Likes", "Comments", "Shares"]

    header = " " * 22 + "".join(f"{lvl:>16}" for lvl in LEVELS)
    print(header)
    for metric in metrics:
        means = [df.loc[df["llm_label"] == lvl, metric].mean() for lvl in LEVELS]
        print(f"{metric + ' (mean)':<22}" + "".join(f"{m:>16.1f}" for m in means))
    medians = [df.loc[df["llm_label"] == lvl, "Total Interactions"].median()
               for lvl in LEVELS]
    print(f"{'Total Int. (median)':<22}" + "".join(f"{m:>16.1f}" for m in medians))

    print("\n  Kruskal-Wallis across the three levels:")
    for metric in ["Total Interactions", "Shares"]:
        groups = [df.loc[df["llm_label"] == lvl, metric].dropna() for lvl in LEVELS]
        h_stat, p_value = kruskal(*groups)
        print(f"    {metric:<20} H = {h_stat:7.1f}   p = {p_value:.2e}")


def adjusted_model(df):
    """Does fear level still predict engagement once we account for confounds?

    The raw comparison can't tell whether high-fear posts are shared more
    because of the fear or because they happen to be photos from big pages.
    To separate those, we fit a negative binomial regression (the right choice
    for over-dispersed counts) with three controls: content type, page category
    and page size (log followers). 'No Fear' is the baseline, so each number is
    an incidence-rate ratio — how many times more (or fewer) shares a post gets
    relative to a comparable no-fear post. The model uses only rows with a known
    follower count, which is about two-thirds of the sample.
    """
    print("\n--- Negative binomial regression (adjusted for confounds) ---")

    data = df.copy()
    data["fear"] = pd.Categorical(data["llm_label"], categories=LEVELS)
    data["log_followers"] = np.log1p(data["Followers at Posting"])

    # Keep the handful of common content types / page categories as themselves
    # and fold the long tail into "Other", so rare levels don't destabilise the fit.
    top_types = data["Type"].value_counts().head(4).index
    data["ctype"] = np.where(data["Type"].isin(top_types), data["Type"], "Other")

    categories = data["Page Category"].replace("", "Unknown")
    top_cats = categories.value_counts().head(6).index
    data["pcat"] = np.where(categories.isin(top_cats), categories, "Other")

    data = data.dropna(subset=["Shares", "Total Interactions", "log_followers"])
    print(f"  Posts with complete metadata: {len(data):,}")

    for outcome in ["Shares", "Total Interactions"]:
        formula = (f"Q('{outcome}') ~ C(fear, Treatment('No Fear')) "
                   "+ C(ctype) + C(pcat) + log_followers")
        model = smf.glm(formula, data=data,
                        family=sm.families.NegativeBinomial(alpha=1.0)).fit()

        print(f"\n  Outcome: {outcome}  (ratios relative to No Fear)")
        ci = model.conf_int()
        for term in model.params.index:
            if "fear" in term:
                level = term.split("T.")[-1].rstrip("]")
                rate = np.exp(model.params[term])
                low, high = np.exp(ci.loc[term])
                print(f"    {level:<15} IRR = {rate:4.2f}  "
                      f"[{low:.2f}, {high:.2f}]  p = {model.pvalues[term]:.2e}")


def plot_distribution(df):
    """Bar chart of the fear-level split (Figure: fear distribution)."""
    counts = [(df["llm_label"] == lvl).sum() for lvl in LEVELS]
    share = [100 * c / len(df) for c in counts]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(LEVELS, share, color=COLORS)
    for i, (c, s) in enumerate(zip(counts, share)):
        ax.text(i, s + 0.6, f"{s:.1f}%\n(n={c})", ha="center", fontsize=9)
    ax.set_ylabel("Share of annotated posts (%)")
    ax.set_title(f"Digital fear level distribution (n={len(df):,})")
    ax.set_ylim(0, 48)
    fig.savefig(os.path.join(FIG_DIR, "fig_fear_distribution.png"))
    plt.close(fig)


def plot_engagement(df):
    """Grouped bars of mean engagement per fear level (Figure: engagement)."""
    metrics = ["Total Interactions", "Likes", "Comments", "Shares"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, level in enumerate(LEVELS):
        means = [df.loc[df["llm_label"] == level, m].mean() for m in metrics]
        ax.bar(x + (i - 1) * width, means, width, label=level, color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Mean count per post")
    ax.set_title("Average engagement by fear level")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "fig_engagement_by_fear.png"))
    plt.close(fig)


# ======================================================================
# RQ3 — annotator agreement
# ======================================================================
def agreement(df):
    """Report how closely FANAR's labels match the human annotator.

    Accuracy alone is misleading when classes are uneven, so we also give
    Cohen's kappa (chance-corrected). The weighted versions reward the model
    for being 'close' — mixing up Moderate and High matters less than mixing up
    No Fear and High — which makes sense for an ordered scale like this one.
    """
    fanar = df["llm_label"]
    human = df["Manual label"]

    print("\n--- FANAR vs. human agreement ---")
    print(f"  Paired posts: {len(df)}")
    print(f"  Accuracy:            {100 * accuracy_score(human, fanar):.1f}%")
    print(f"  Cohen's kappa:       {cohen_kappa_score(human, fanar, labels=LEVELS):.3f}")
    print(f"  Kappa (linear):      "
          f"{cohen_kappa_score(human, fanar, labels=LEVELS, weights='linear'):.3f}")
    print(f"  Kappa (quadratic):   "
          f"{cohen_kappa_score(human, fanar, labels=LEVELS, weights='quadratic'):.3f}")

    print("\n  Per-class (human treated as the reference):")
    print(classification_report(human, fanar, labels=LEVELS,
                                digits=3, zero_division=0))

    # How far apart are the disagreements? Adjacent slips are forgivable;
    # jumping two levels (No Fear <-> High Fear) is the real worry.
    rank = {lvl: i for i, lvl in enumerate(LEVELS)}
    gap = (human.map(rank) - fanar.map(rank)).abs()
    print("  Disagreement size:")
    print(f"    exact match : {100 * (gap == 0).mean():.1f}%")
    print(f"    off by one  : {100 * (gap == 1).mean():.1f}%")
    print(f"    off by two  : {100 * (gap == 2).mean():.1f}%")


def plot_confusion(df):
    """Confusion matrix heatmap, human rows vs. FANAR columns."""
    matrix = confusion_matrix(df["Manual label"], df["llm_label"], labels=LEVELS)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(LEVELS, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(LEVELS)
    ax.set_xlabel("FANAR label")
    ax.set_ylabel("Human label")
    ax.set_title(f"FANAR vs. human (n={len(df)})")

    midpoint = matrix.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > midpoint else "black",
                    fontsize=12)
    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(FIG_DIR, "fig_confusion_matrix.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("RQ2  Prevalence and reception of digital fear")
    print("=" * 60)
    fanar = load_fanar()
    fear_distribution(fanar)
    engagement_by_level(fanar)
    adjusted_model(fanar)
    plot_distribution(fanar)
    plot_engagement(fanar)

    print("\n" + "=" * 60)
    print("RQ3  Reliability of FANAR as an annotator")
    print("=" * 60)
    human = load_human()
    agreement(human)
    plot_confusion(human)

    print(f"\nFigures written to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
