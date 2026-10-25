"""
RQ1 analysis: characterize Arabic cybersecurity discourse on Facebook.
Computes content, engagement, temporal, and sentiment metrics for both
datasets and saves publication-ready figures + a summary table.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend: write figures to disk, don't open windows
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "figure.autolayout": True,
})

DS1 = "الجرائم الإلكترونية OR القرصنة الإلكترونية OR الأمن السيبراني.csv"
DS2 = "19_Cyber_Threat_Intelligence_filtered.csv"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

NUMERIC = ["Total Interactions", "Likes", "Comments", "Shares", "Love",
           "Wow", "Haha", "Sad", "Angry", "Care", "Overperforming Score",
           "Followers at Posting"]

def load(path, name):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in NUMERIC:
        if c in df.columns:
            # strip thousands separators (e.g. "49,695") before converting
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(),
                                  errors="coerce")
    df["dt"] = pd.to_datetime(df["Post Created Date"], errors="coerce")
    df["year"] = df["dt"].dt.year
    df["dow"] = df["dt"].dt.day_name()
    df["msg_len"] = df["Message"].str.len()
    df["has_url"] = df["Message"].str.contains("http", na=False).astype(int)
    df["dataset"] = name
    return df

# ---- Reaction-based sentiment ----------------
def sentiment(df):
    pos = df["Likes"].fillna(0) + 2*df["Love"].fillna(0) + 0.5*df["Care"].fillna(0)
    neg = 2*df["Angry"].fillna(0) + df["Sad"].fillna(0)
    total = pos + neg
    score = np.where(total > 0, (pos - neg) / total, 0.0)
    label = np.where(score >= 0.3, "Positive",
             np.where(score <= -0.3, "Negative", "Neutral"))
    return pd.Series(score, index=df.index), pd.Series(label, index=df.index)

def describe(df, name):
    print(f"\n{'='*64}\n{name}: {len(df):,} posts\n{'='*64}")
    print(f"Date range: {df['dt'].min().date()} -> {df['dt'].max().date()}")
    print(f"Unique pages: {df['Page Name'].nunique():,}")
    print(f"Median message length: {df['msg_len'].median():.0f} chars")
    print(f"Posts containing URL: {100*df['has_url'].mean():.1f}%")
    print("\n-- Engagement (mean / median) --")
    for c in ["Total Interactions", "Likes", "Comments", "Shares"]:
        print(f"  {c:20s} {df[c].mean():8.1f} / {df[c].median():6.1f}")
    print("\n-- Post types (%) --")
    print((df["Type"].value_counts(normalize=True)*100).round(1).head(5).to_string())
    print("\n-- Top page categories (%) --")
    print((df["Page Category"].replace("", np.nan).dropna()
           .value_counts(normalize=True)*100).round(1).head(5).to_string())
    s_score, s_label = sentiment(df)
    print("\n-- Reaction-based sentiment (%) --")
    print((s_label.value_counts(normalize=True)*100).round(1).to_string())
    return s_label

def main():
    d1 = load(DS1, "DS1")
    d2 = load(DS2, "DS2")
    s1 = describe(d1, "DS1: Arabic Cybersecurity Discourse")
    s2 = describe(d2, "DS2: Cyber Threat Intelligence (filtered)")
    palette = ["#2C5F8A", "#C44E52"]

    # FIG 1: Posts per year
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for df, name, col in [(d1, "DS1", palette[0]), (d2, "DS2", palette[1])]:
        yr = df[df["year"].between(2014, 2024)]["year"].value_counts().sort_index()
        ax.plot(yr.index, yr.values, marker="o", label=name, color=col, lw=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of posts")
    ax.set_title("Temporal distribution of posts (2014–2024)")
    ax.legend()
    fig.savefig(f"{OUT}/fig1_temporal.pdf"); fig.savefig(f"{OUT}/fig1_temporal.png"); plt.close(fig)

    # FIG 2: Post type distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    types = ["Photo", "Link", "Status", "Native Video"]
    x = np.arange(len(types)); w = 0.38
    for i, (df, name, col) in enumerate([(d1, "DS1", palette[0]), (d2, "DS2", palette[1])]):
        vc = df["Type"].value_counts(normalize=True)*100
        ax.bar(x + (i-0.5)*w, [vc.get(t, 0) for t in types], w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(types)
    ax.set_ylabel("Share of posts (%)"); ax.set_title("Content type distribution")
    ax.legend()
    fig.savefig(f"{OUT}/fig2_types.pdf"); fig.savefig(f"{OUT}/fig2_types.png"); plt.close(fig)

    # FIG 3: Mean engagement by post type (DS1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = d1.groupby("Type")[["Likes", "Comments", "Shares"]].mean().loc[
        [t for t in types if t in d1["Type"].unique()]]
    g.plot(kind="bar", ax=ax, color=["#2C5F8A", "#55A868", "#C44E52"])
    ax.set_ylabel("Mean count per post"); ax.set_xlabel("")
    ax.set_title("Average engagement by content type (DS1)")
    plt.xticks(rotation=0)
    fig.savefig(f"{OUT}/fig3_eng_type.pdf"); fig.savefig(f"{OUT}/fig3_eng_type.png"); plt.close(fig)

    # FIG 4: Sentiment distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = ["Positive", "Neutral", "Negative"]
    x = np.arange(len(order)); w = 0.38
    for i, (s, name, col) in enumerate([(s1, "DS1", palette[0]), (s2, "DS2", palette[1])]):
        vc = s.value_counts(normalize=True)*100
        ax.bar(x + (i-0.5)*w, [vc.get(o, 0) for o in order], w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Share of posts (%)")
    ax.set_title("Reaction-based audience sentiment")
    ax.legend()
    fig.savefig(f"{OUT}/fig4_sentiment.pdf"); fig.savefig(f"{OUT}/fig4_sentiment.png"); plt.close(fig)

    # FIG 6: Geographic distribution (top 10 countries, both datasets)
    names = {"JO":"Jordan","EG":"Egypt","AE":"UAE","PS":"Palestine","SY":"Syria",
             "IQ":"Iraq","DZ":"Algeria","MA":"Morocco","LY":"Libya","SD":"Sudan",
             "TR":"Turkey","LB":"Lebanon","SA":"Saudi Arabia","TN":"Tunisia",
             "KW":"Kuwait","YE":"Yemen","QA":"Qatar","OM":"Oman","BH":"Bahrain"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, df, name, col in [(axes[0], d1, "DS1", palette[0]),
                              (axes[1], d2, "DS2", palette[1])]:
        c = df["Page Admin Top Country"].replace("", np.nan).dropna()
        top = c.value_counts().head(10)[::-1]
        labels = [names.get(k, k) for k in top.index]
        ax.barh(range(len(top)), 100*top.values/len(c), color=col)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(labels)
        ax.set_xlabel("Share of posts (%)"); ax.set_title(name)
    fig.suptitle("Geographic distribution (top 10 admin countries)")
    fig.savefig(f"{OUT}/fig6_geo.pdf"); fig.savefig(f"{OUT}/fig6_geo.png"); plt.close(fig)

    # FIG 5: Posts by day of week (DS1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = d1["dow"].value_counts().reindex(days)
    ax.bar(range(7), dow.values, color=palette[0])
    ax.set_xticks(range(7)); ax.set_xticklabels([d[:3] for d in days])
    ax.set_ylabel("Number of posts"); ax.set_title("Posting activity by day of week (DS1)")
    fig.savefig(f"{OUT}/fig5_dow.pdf"); fig.savefig(f"{OUT}/fig5_dow.png"); plt.close(fig)

    print(f"\nFigures saved to {OUT}")

if __name__ == "__main__":
    main()