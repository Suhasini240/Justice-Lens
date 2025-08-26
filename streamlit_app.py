#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Paths (repo/app aware)
#HERE = Path(__file__).resolve().parent              # .../mediabiasanalyzer/app
#DATA_PATH = HERE.parent / "data" / "corpus_with_sentiment.parquet"


st.set_page_config(page_title="Media Bias Analyser", layout="wide")

# --- First Page ---
st.title("Justice Lens")                         # big, bold title
st.markdown("**A media bias analyzer tool**")    # subtitle

st.header("About")
st.markdown("""
**Justice Lens** explores how tone differs across outlets on the same topics.
We don’t rate the media — we reveal it: by topic, by tone, by truth.

### What this app shows
1. **Overall polarity by topic** — Mean sentiment (headline 60% + abstract 40%) for each outlet, with **95% CI** and **Cohen’s d**.  
2. **Who uses more strong sentiment?** — Share of articles with **|sentiment| ≥ threshold**, split into positive vs negative extremes.  
3. **Which outlet varies more in tone over time?** — Monthly averages with **±σ** or **IQR** bands (volatility).  
4. **Headline vs body gap (clickbait lens)** — **Sent(headline) − Sent(abstract)**; dumbbell plot per outlet/topic.
""")

st.header("NLP & analysis used")
st.markdown("""
- **VADER sentiment (NLTK)** — compound score blended as **60% headline + 40% abstract**.  
- **Bootstrap 95% confidence intervals** for means.  
- **Effect size (Cohen’s d)** for practical differences (not a p-value).  
- **Volatility bands** using **standard deviation (±σ)** or **interquartile range (IQR)**.  
- *(Planned/optional)* **TF-IDF + cosine similarity** for same-story matching across outlets.
""")

st.caption(
    "Notes: sentiment uses VADER in the range −1…+1 (higher = more positive). "
    "Results depend on topic mix and coverage choices; lexicon methods can miss sarcasm/nuance."
)

st.divider() 

# -------- Data loader (reads your prebuilt parquet) --------
@st.cache_data(show_spinner=True)
def load_data(path: str = "data/corpus_with_sentiment.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["pub_dt"] = pd.to_datetime(df["pub_dt"], utc=True, errors="coerce")
    return df

df = load_data()

OUTLETS = ["The New York Times", "Fox News"]
topics = ["All topics"] + sorted(df["topic"].dropna().unique().tolist())
topic = st.selectbox("Topic", options=topics, index=0)
dfv = df if topic == "All topics" else df[df["topic"] == topic]

st.title("1. Overall polarity by topic")
st.caption("Sentiment: VADER compound (headline 60% + abstract 40%). Higher is more positive; range [-1, 1].")

# -------- helpers --------
def bootstrap_ci(series: pd.Series, func=np.mean, n_boot=2000, alpha=0.05, seed=1337):
    x = pd.Series(series).dropna().to_numpy()
    if x.size == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.array([func(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2: return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return np.nan if sp == 0 else (a.mean() - b.mean()) / sp

# -------- compute summary for the selected slice --------
rows = []
for o in OUTLETS:
    vals = dfv.loc[dfv["outlet"] == o, "sent_overall"]
    lo, hi = bootstrap_ci(vals, func=np.mean)
    rows.append({
        "outlet": o,
        "n": int(vals.notna().sum()),
        "mean": float(vals.mean()) if vals.notna().any() else np.nan,
        "median": float(vals.median()) if vals.notna().any() else np.nan,
        "ci95_lo": lo, "ci95_hi": hi
    })
summary = pd.DataFrame(rows)

d_val = cohens_d(
    dfv.loc[dfv["outlet"] == OUTLETS[0], "sent_overall"],
    dfv.loc[dfv["outlet"] == OUTLETS[1], "sent_overall"]
)

# -------- display metrics --------
c1, c2, c3 = st.columns(3)
with c1:
    lo, hi = summary.loc[summary["outlet"] == OUTLETS[0], ["ci95_lo","ci95_hi"]].iloc[0]
    st.metric(f"{OUTLETS[0]} — mean (95% CI)",
              f'{summary.loc[summary["outlet"]==OUTLETS[0],"mean"].iloc[0]:+.3f}',
              f'{lo:+.3f} to {hi:+.3f}')
    st.caption(f"n = {int(summary.loc[summary['outlet']==OUTLETS[0],'n'].iloc[0])}")

with c2:
    lo, hi = summary.loc[summary["outlet"] == OUTLETS[1], ["ci95_lo","ci95_hi"]].iloc[0]
    st.metric(f"{OUTLETS[1]} — mean (95% CI)",
              f'{summary.loc[summary["outlet"]==OUTLETS[1],"mean"].iloc[0]:+.3f}',
              f'{lo:+.3f} to {hi:+.3f}')
    st.caption(f"n = {int(summary.loc[summary['outlet']==OUTLETS[1],'n'].iloc[0])}")

with c3:
    st.metric("Effect size (Cohen’s d)", f"{d_val:+.3f}" if pd.notna(d_val) else "—")
    st.caption("Magnitude guide: 0.2 small, 0.5 medium, 0.8 large")

with st.expander("Details"):
    st.dataframe(
        summary.assign(ci95=lambda x: x.apply(lambda r: f"({r.ci95_lo:+.3f}, {r.ci95_hi:+.3f})", axis=1))
                .drop(columns=["ci95_lo","ci95_hi"])
    )

#visual------mean sentiment with 95% CI---
import matplotlib.pyplot as plt

st.subheader("Visual: Mean sentiment by outlet")

# pull stats from the 'summary' table you already computed
x = np.arange(len(OUTLETS))
means = [summary.loc[summary["outlet"]==o, "mean"].iloc[0] for o in OUTLETS]
los   = [summary.loc[summary["outlet"]==o, "ci95_lo"].iloc[0] for o in OUTLETS]
his   = [summary.loc[summary["outlet"]==o, "ci95_hi"].iloc[0] for o in OUTLETS]
ns    = [int(summary.loc[summary["outlet"]==o, "n"].iloc[0]) for o in OUTLETS]

# asymmetrical error bars: distance from mean to CI bounds
yerr = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])

fig, ax = plt.subplots()
ax.bar(x, means, yerr=yerr, capsize=6)
ax.axhline(0.0, linestyle="--", linewidth=1)              # neutral line
ax.set_xticks(x, OUTLETS)
ax.set_ylabel("Mean sentiment (−1 … +1)")
ax.set_title(f"Mean sentiment — {topic}")

# add labels with mean & n
for i, m in enumerate(means):
    ax.text(i, m, f"{m:+.3f}\n(n={ns[i]})", ha="center", va="bottom")

st.pyplot(fig, clear_figure=True)

st.caption("Bars show the average sentiment (headline 60% + abstract 40%). Whiskers = 95% bootstrap CI. Dashed line marks neutral (0).")

st.divider()  
#---------------
st.header("2. Which news platform uses more strong sentiment?")

topics_all = ["All topics"] + sorted(df["topic"].dropna().unique().tolist())
c1, c2 = st.columns([2,1])
with c1:
    q2_topic = st.selectbox("Topic (extremes)", options=topics_all, index=0, key="q2_ext_topic")
with c2:
    thr = st.slider("Extreme threshold (|sent| ≥)", min_value=0.50, max_value=0.95, value=0.70, step=0.01, key="q2_ext_thr")

# filter
dfx = df.copy() if q2_topic == "All topics" else df[df["topic"] == q2_topic].copy()
dfx = dfx[["outlet", "sent_overall"]].dropna()
if dfx.empty:
    st.info("No articles in this slice.")
else:
    # classify extremes
    dfx["is_extreme"] = dfx["sent_overall"].abs() >= thr
    dfx["polarity"] = np.where(
        dfx["sent_overall"] >= thr, "positive",
        np.where(dfx["sent_overall"] <= -thr, "negative", "other")
    )

    # aggregates
    g_total   = dfx.groupby("outlet")["sent_overall"].size().rename("n")
    g_extreme = dfx.groupby("outlet")["is_extreme"].sum().rename("n_extreme")
    g_pos     = dfx[dfx["polarity"]=="positive"].groupby("outlet")["polarity"].size().rename("n_pos")
    g_neg     = dfx[dfx["polarity"]=="negative"].groupby("outlet")["polarity"].size().rename("n_neg")

    summary = pd.concat([g_total, g_extreme, g_pos, g_neg], axis=1).fillna(0).reset_index()
    summary["pct_extreme"] = (summary["n_extreme"] / summary["n"] * 100.0).round(2)
    summary["pct_pos_ext"] = (summary["n_pos"] / summary["n"] * 100.0).round(2)
    summary["pct_neg_ext"] = (summary["n_neg"] / summary["n"] * 100.0).round(2)
    summary = summary.sort_values("outlet")

    # --- Visual: stacked bar of extreme sentiment (pos/neg) ---
    st.subheader("Visual: % of extreme-tone articles by outlet")
    x = np.arange(len(summary))
    pos_pct = summary["pct_pos_ext"].to_numpy()
    neg_pct = summary["pct_neg_ext"].to_numpy()

    fig, ax = plt.subplots()
    ax.bar(x, pos_pct, label="Positive extremes")
    ax.bar(x, neg_pct, bottom=pos_pct, label="Negative extremes")
    ax.set_xticks(x, summary["outlet"])
    ax.set_ylabel(f"% of articles (|sent| ≥ {thr:.2f})")
    ax.set_title(f"Share of extremes — {q2_topic}")

    # annotate totals
    for i, (p, n) in enumerate(zip(pos_pct, neg_pct)):
        ax.text(i, p + n, f"{p+n:.1f}%", ha="center", va="bottom")

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # --- Details table ---
    st.subheader("Details")
    st.dataframe(
        summary[["outlet","n","n_extreme","pct_extreme","n_pos","pct_pos_ext","n_neg","pct_neg_ext"]],
        use_container_width=True
    )

    st.caption(
        "Extreme = |VADER compound (headline 60% + abstract 40%)| ≥ threshold. "
        "Bars show positive/negative extremes as stacked percentages; labels show total extreme share."
    )
    
st.divider()  
#--------------
# Q3 — Which outlet varies more in tone over time?
st.header("3. Which outlet varies more in tone over time?")

topics_all = ["All topics"] + sorted(df["topic"].dropna().unique().tolist())
c1, c2 = st.columns([2,1])
with c1:
    q3_topic = st.selectbox("Topic (volatility)", options=topics_all, index=0, key="q3_topic")
with c2:
    band_stat = st.radio("Uncertainty band", ["Std dev (±σ)", "IQR (Q1–Q3)"], index=0, horizontal=True, key="q3_band")

# ---------- data slice ----------
dfv = df.copy() if q3_topic == "All topics" else df[df["topic"] == q3_topic].copy()
dfv = dfv.dropna(subset=["sent_overall", "pub_dt"])
if dfv.empty:
    st.info("No articles in this slice.")
else:
    dfv["month"] = dfv["pub_dt"].dt.to_period("M").dt.to_timestamp()

    @st.cache_data(show_spinner=True)
    def monthly_stats(slice_df: pd.DataFrame) -> pd.DataFrame:
        g = slice_df.groupby(["month", "outlet"])["sent_overall"]
        agg = g.agg(
            n="size",
            mean="mean",
            std="std",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        ).reset_index()
        agg["iqr"] = agg["q3"] - agg["q1"]
        return agg

    mstats = monthly_stats(dfv)

    # ---------- quick metrics: average volatility across months ----------
    vol_metric = "std" if band_stat.startswith("Std") else "iqr"
    avg_vol = (
        mstats.pivot_table(index="outlet", values=vol_metric, aggfunc="mean")
        .reindex(OUTLETS)
        .rename(columns={vol_metric: "avg_vol"})
    )

    c1, c2 = st.columns(2)
    with c1:
        v = avg_vol.loc[OUTLETS[0], "avg_vol"]
        st.metric(f"{OUTLETS[0]} — average monthly {vol_metric}", f"{v:+.3f}" if pd.notna(v) else "—")
    with c2:
        v = avg_vol.loc[OUTLETS[1], "avg_vol"]
        st.metric(f"{OUTLETS[1]} — average monthly {vol_metric}", f"{v:+.3f}" if pd.notna(v) else "—")

    # ---------- visual: mean with band per outlet ----------
    st.subheader("Visual: monthly mean sentiment with uncertainty band")

    fig, ax = plt.subplots()
    for outlet in OUTLETS:
        sub = mstats[mstats["outlet"] == outlet].sort_values("month")
        if sub.empty:
            continue
        ax.plot(sub["month"], sub["mean"], label=outlet)
        if band_stat.startswith("Std"):
            lo = sub["mean"] - sub["std"]
            hi = sub["mean"] + sub["std"]
        else:
            lo = sub["q1"]
            hi = sub["q3"]
        ax.fill_between(sub["month"], lo, hi, alpha=0.2, label=None)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sentiment (−1 … +1)")
    ax.set_title(f"Volatility over time — {q3_topic}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # ---------- details ----------
    with st.expander("Details (monthly table)"):
        show_cols = ["month", "outlet", "n", "mean", "std", "q1", "q3", "iqr"]
        st.dataframe(
            mstats[show_cols].sort_values(["month", "outlet"]).reset_index(drop=True),
            use_container_width=True
        )

    st.caption(
        "Mean sentiment is VADER compound (headline 60% + abstract 40%). "
        "Shaded band shows ± standard deviation or the interquartile range (IQR). "
        "Dashed line marks neutral (0). Higher average band width indicates greater volatility."
    )
st.divider()  
#--------------
st.header("4. Is the headline consistently more negative/positive than the abstract?")

# Controls
topics_all = ["All topics"] + sorted(df["topic"].dropna().unique().tolist())
q4_topic = st.selectbox("Topic (headline vs abstract)", options=topics_all, index=0, key="q4_topic")

# Slice + compute gap
dft = df.copy() if q4_topic == "All topics" else df[df["topic"] == q4_topic].copy()
dft = dft.dropna(subset=["sent_headline", "sent_abstract"])
if dft.empty:
    st.info("No articles in this slice.")
else:
    dft["gap_h_minus_a"] = dft["sent_headline"] - dft["sent_abstract"]

    # ---- aggregate per outlet (ensure 'outlet' is a column) ----
    agg = (
    dft.groupby("outlet", as_index=False)
       .agg(
           n=("gap_h_minus_a","size"),
           mean_gap=("gap_h_minus_a","mean"),
           median_gap=("gap_h_minus_a","median"),
           mean_headline=("sent_headline","mean"),
           mean_abstract=("sent_abstract","mean"),
       )
    )
    
    # ---- robust 95% CI: one tuple per outlet, then expand ----
    ci = (
        dft.groupby("outlet")["gap_h_minus_a"]
           .apply(lambda s: bootstrap_ci(s))           # -> tuple (lo, hi)
           .reset_index(name="ci")
    )
    ci[["ci95_lo","ci95_hi"]] = pd.DataFrame(ci["ci"].tolist(), index=ci.index)
    ci = ci.drop(columns="ci")
    
    # ---- merge & enforce outlet order (no duplicate index) ----
    g = (
        agg.merge(ci, on="outlet", how="left")
           .drop_duplicates(subset=["outlet"])         # safety
           .set_index("outlet")
           .reindex(OUTLETS)                           # ["The New York Times","Fox News"]
           .reset_index()
    )



    # ---- safe metric helper ----
    def outlet_metric(df_out, outlet_name):
        row = df_out[df_out["outlet"] == outlet_name]
        if row.empty:
            return "—", "—", 0
        r = row.iloc[0]
        mean_txt = f"{r['mean_gap']:+.3f}"
        ci_txt   = f"{r['ci95_lo']:+.3f} to {r['ci95_hi']:+.3f}"
        return mean_txt, ci_txt, int(r["n"])

    # Metrics row (avoid referencing g[...] inside f-strings when empty)
    c1, c2 = st.columns(2)
    m, ci_txt, nval = outlet_metric(g, OUTLETS[0])
    with c1:
        st.metric(f"{OUTLETS[0]} — mean gap (headline − abstract)", m, ci_txt)
        st.caption(f"n = {nval}")
    m, ci_txt, nval = outlet_metric(g, OUTLETS[1])
    with c2:
        st.metric(f"{OUTLETS[1]} — mean gap (headline − abstract)", m, ci_txt)
        st.caption(f"n = {nval}")

    # ---- Dumbbell plot (mean headline vs mean abstract per outlet) ----
    st.subheader("Visual: headline vs abstract (mean) — dumbbell plot")
    fig, ax = plt.subplots()
    
    labels = g["outlet"].tolist()
    y = np.arange(len(labels))
    ab = g["mean_abstract"].to_numpy(dtype=float)
    hl = g["mean_headline"].to_numpy(dtype=float)
    
    for i in range(len(labels)):
        if np.isnan(ab[i]) or np.isnan(hl[i]):
            continue  # skip rows with no data for this outlet
        ax.plot([ab[i], hl[i]], [i, i], marker="o")
        gap = g["mean_gap"].iloc[i]
        gap_txt = f"{gap:+.3f}" if pd.notna(gap) else "—"
        ax.text(max(ab[i], hl[i]), i, f" {gap_txt}", va="center", ha="left")
    
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Sentiment (−1 … +1)")
    ax.set_title(f"Headline vs abstract — {q4_topic}")
    st.pyplot(fig, clear_figure=True)

    # ---- Details table ----
    with st.expander("Details"):
        st.dataframe(
            g[["outlet","n","mean_headline","mean_abstract","mean_gap","median_gap","ci95_lo","ci95_hi"]],
            use_container_width=True
        )

    st.caption(
        "Gap = Sent(headline) − Sent(abstract). Positive gap ⇒ headline is more positive than the abstract on average; "
        "negative gap ⇒ headline is more negative. Means use VADER compound; dashed line marks neutral (0)."
    )


# In[ ]:




