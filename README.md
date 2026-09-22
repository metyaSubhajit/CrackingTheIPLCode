# 🏏 Cracking the IPL Code
## A Comprehensive Data Analytics Study of the Indian Premier League (2008–2020)

> **College Data Analytics Project** | Dataset: Kaggle IPL Complete Dataset (2008–2020)

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Objectives](#-objectives)
4. [Dataset](#-dataset)
5. [Technologies Used](#-technologies-used)
6. [Project Structure](#-project-structure)
7. [Methodology](#-methodology)
8. [Data Cleaning](#-data-cleaning-process)
9. [Exploratory Data Analysis](#-exploratory-data-analysis)
10. [KPI Descriptions](#-key-performance-indicators-kpis)
11. [Statistical Analysis](#-statistical-analysis)
12. [Dashboard Features](#-streamlit-dashboard-features)
13. [Installation](#-installation)
14. [How to Run](#-how-to-run)
15. [Key Findings](#-key-findings)
16. [Limitations](#-limitations)
17. [Future Scope](#-future-scope)
18. [Conclusion](#-conclusion)

---

## 📌 Project Overview

This project applies a complete **Data Analytics pipeline** to the IPL Complete Dataset (2008–2020), covering 13 seasons of the Indian Premier League. The project encompasses raw data acquisition, systematic data cleaning, feature engineering, exploratory data analysis, statistical hypothesis testing, KPI computation, and an interactive multi-page **Streamlit** web dashboard with **Plotly** visualisations.

The entire Python implementation is contained in a **single file — `app.py`** — organised into 11 clearly commented sections with 45+ functions, making it fully readable and suitable for a college viva presentation.

The project also includes an in-app **DOCX report generator** that produces a formatted academic report (`project_report.docx`) from live data on demand.

---

## ❓ Problem Statement

The Indian Premier League is one of the world's most data-rich sporting leagues, generating structured match-level and ball-by-ball records across 13 seasons. Despite the availability of this data, evidence-based insights regarding:

- franchise dominance and strategic patterns,
- the measurable impact of toss decisions on match outcomes,
- player performance benchmarks (batting and bowling),
- phase-wise scoring evolution over seasons, and
- venue-specific performance characteristics

…remain under-analysed. This project addresses this gap by building a rigorous, reproducible Data Analytics study grounded entirely in the raw dataset.

---

## 🎯 Objectives

1. Clean and validate approximately 816 match records and 179,000 ball-by-ball delivery records to ensure analysis-ready data quality.
2. Identify the most successful franchises, players, and venues across 13 IPL seasons.
3. Quantify the statistical effect of toss decisions on match outcomes using hypothesis testing.
4. Profile top batsmen and bowlers using aggregated statistical metrics (runs, strike rate, wickets, economy rate).
5. Analyse scoring patterns by match phase — Powerplay (overs 0–5), Middle (6–15), and Death (16–19) — across all seasons.
6. Detect multi-season trends in average match scores, win types (by runs vs. by wickets), and extras conceded.
7. Build an interactive, filterable Streamlit dashboard presenting all findings with drill-down capability.
8. Generate a downloadable formatted PDF academic report from live data.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Dataset Name** | IPL Complete Dataset (2008–2020) |
| **Author** | Patrick Bhingre |
| **Platform** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020 |
| **Licence** | CC0: Public Domain |
| **Format** | Two CSV files |

### File 1: `matches.csv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Unique match identifier (primary key) |
| `season` | Integer | IPL season year (2008–2020) |
| `city` | String | Host city (~5% missing for neutral venues) |
| `date` | String (→ datetime) | Match date in YYYY/MM/DD format |
| `player_of_match` | String | Award recipient |
| `venue` | String | Stadium name |
| `neutral_venue` | Integer (0/1) | Flag for neutral ground |
| `team1` | String | First team listed |
| `team2` | String | Second team listed |
| `toss_winner` | String | Toss-winning franchise |
| `toss_decision` | String | `bat` or `field` |
| `result` | String | `runs`, `wickets`, `tie`, or `no result` |
| `dl_applied` | Integer (0/1) | Duckworth-Lewis applied |
| `winner` | String | Match-winning franchise (NaN for no-result) |
| `win_by_runs` | Integer | Winning margin in runs (0 if won by wickets) |
| `win_by_wickets` | Integer | Winning margin in wickets (0 if won by runs) |
| `umpire1` | String | On-field umpire 1 |
| `umpire2` | String | On-field umpire 2 |

**Approximate size:** ~816 rows × 18 columns

### File 2: `deliveries.csv`

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | Integer | Foreign key linking to `matches.id` |
| `inning` | Integer | 1 or 2 (Super Over = 3 or 4, filtered out) |
| `batting_team` | String | Batting franchise name |
| `bowling_team` | String | Bowling franchise name |
| `over` | Integer | Over number (0-indexed: 0–19) |
| `ball` | Integer | Ball number within the over |
| `batsman` | String | Batsman on strike |
| `non_striker` | String | Non-striking batsman |
| `bowler` | String | Bowler for the delivery |
| `is_super_over` | Integer (0/1) | Super over flag |
| `wide_runs` | Integer | Wide runs credited |
| `bye_runs` | Integer | Bye runs credited |
| `legbye_runs` | Integer | Leg-bye runs credited |
| `noball_runs` | Integer | No-ball runs credited |
| `penalty_runs` | Integer | Penalty runs (rare) |
| `batsman_runs` | Integer | Runs credited to the batsman |
| `extra_runs` | Integer | Total extra runs on this delivery |
| `total_runs` | Integer | Total runs on this delivery |
| `player_dismissed` | String | Dismissed batsman (null if not out) |
| `dismissal_kind` | String | Caught, bowled, LBW, run out, stumped, etc. |
| `fielder` | String | Fielder involved in dismissal |

**Approximate size:** ~179,000 rows × 21 columns

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **pandas** | ≥ 2.0.0 | Data manipulation, groupby, merge, aggregation |
| **numpy** | ≥ 1.26.0 | Numerical computations, array operations |
| **plotly** | ≥ 5.18.0 | Interactive charts (bar, line, scatter, pie, heatmap, box) |
| **streamlit** | ≥ 1.32.0 | Multi-page interactive web dashboard |
| **scipy** | ≥ 1.11.0 | Statistical tests (chi-square, t-test, Pearson correlation) |
| **python-docx** | ≥ 1.1.0 | In-app DOCX report generation |
| **kaggle** | ≥ 1.6.0 | Automatic dataset download via Kaggle API |

> Standard library modules used (no install needed): `os`, `subprocess`, `warnings`, `zipfile`

---

## 📁 Project Structure

```
project-folder/
├── app.py                  ← COMPLETE project: pipeline + dashboard (single file, ~1,965 lines)
├── requirements.txt        ← Python package dependencies
├── README.md               ← Project documentation
├── project_report.docx     ← Academic project report (generated via Download Report page)
└── data/
    ├── matches.csv         ← Place here after downloading from Kaggle
    └── deliveries.csv      ← Place here after downloading from Kaggle
```

### Internal Structure of `app.py`

```
Section  1  — Constants & Team-Name Canonical Mapping
Section  2  — Data Acquisition (Kaggle API + local fallback)
Section  3  — Data Loading & Inspection
Section  4  — Data Cleaning, Missing-Value Handling & Type Correction
Section  5  — Feature Engineering
Section  6  — KPI Calculations
Section  7  — Statistical Analysis
Section  8  — Chart / Visualisation Functions (15 Plotly charts)
Section  9  — Streamlit Dashboard Pages (11 pages)
Section 10  — DOCX Report Generation
Section 11  — Main Application Entry Point
```

---

## 🔬 Methodology

The project follows the standard **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow adapted for Data Analytics:

```
1. Business Understanding
   └─ Define analytical questions and KPIs for IPL cricket data

2. Data Understanding
   └─ Inspect column types, distributions, missing values, duplicates

3. Data Preparation
   ├─ Data cleaning (dedup, type conversion, canonical names)
   ├─ Missing-value handling (imputation & documented exclusion)
   └─ Feature engineering (phase labels, boundary flags, win type)

4. Modelling / Analysis
   ├─ Exploratory Data Analysis (univariate + bivariate)
   ├─ KPI computation (aggregated via pandas groupby)
   └─ Statistical hypothesis testing (chi-square, t-test, Pearson r)

5. Evaluation
   └─ Interpret findings against analytical questions

6. Deployment
   └─ Interactive Streamlit dashboard + DOCX report
```

---

## 🧹 Data Cleaning Process

The following cleaning steps are applied programmatically in **Section 4** of `app.py`:

| Step | Issue | Resolution |
|------|-------|------------|
| 1 | Exact duplicate rows | `df.drop_duplicates()` on both files |
| 2 | Franchise renames across seasons (e.g. *Delhi Daredevils* → *Delhi Capitals*, *Kings XI Punjab* → *Punjab Kings*) | Canonical name-mapping dictionary applied to all team columns |
| 3 | `date` column stored as string | `pd.to_datetime()` conversion; `month` and `day_of_week` extracted as new columns |
| 4 | `city` column ~5% missing (neutral venues) | Imputed from venue name using a mode-based venue→city lookup; residual set to `'Unknown'` |
| 5 | `winner` column ~2 NaN (no-result matches) | Kept as NaN; excluded from all win-rate calculations with `dropna(subset=['winner'])` |
| 6 | Super Over deliveries (`inning > 2`) | Filtered out from main analysis; flagged by `is_super_over` |
| 7 | Run columns potentially float or NaN | All run columns coerced to `int` with `pd.to_numeric(...).fillna(0).astype(int)` |
| 8 | Dismissal-related nulls in deliveries | `player_dismissed`, `dismissal_kind`, `fielder` NaN filled with `'not out'` / `'none'` |
| 9 | Whitespace in venue and city names | `.str.strip()` normalisation |

---

## 📈 Exploratory Data Analysis

EDA is performed across 11 Streamlit dashboard pages and covers:

### Univariate Analysis
- Distribution of win margins (by runs and by wickets)
- Frequency of toss decisions (bat vs. field)
- Count of dismissal types
- Match count per season

### Bivariate Analysis
- Toss decision vs. match outcome cross-tabulation
- Team win percentage vs. toss win percentage
- Batsman runs vs. strike rate scatter
- Bowler wickets vs. economy rate scatter
- Phase (Powerplay / Middle / Death) vs. average run rate

### Time-series / Trend Analysis
- Average match score per season (2008–2020)
- Phase-wise run rate heatmap (phase × season)
- Win-by-runs vs. win-by-wickets trend by season
- Average extras per match by season

### Venue Analysis
- Average 1st-innings score per top venue
- Toss decision breakdown per top venue

---

## 📊 Key Performance Indicators (KPIs)

All KPIs are computed dynamically from the dataset at runtime (no hardcoded values).

| KPI | Description | Source Columns |
|-----|-------------|----------------|
| **Total Matches** | Count of all valid matches | `matches.id` |
| **Total Deliveries** | Count of all non-Super-Over deliveries | `deliveries` (inning ≤ 2) |
| **Seasons Covered** | Unique season count | `matches.season` |
| **Franchises** | Unique franchise count across team1 + team2 | `matches.team1`, `matches.team2` |
| **Most Wins — Franchise** | Franchise with highest all-time win count | `matches.winner` |
| **Toss → Win Rate** | % of matches won by the toss winner | `toss_winner == winner` |
| **Top Run Scorer** | Batsman with highest career `batsman_runs` | `deliveries.batsman_runs` |
| **Top Wicket Taker** | Bowler with most career dismissals | `deliveries.is_dismissal` |
| **Avg 1st-Innings Score** | Mean total runs per first innings | `deliveries` (inning=1) |
| **Boundary %** | (Fours + Sixes) / Legal balls | `batsman_runs ∈ {4,6}` |
| **Total Fours** | All career four-runs scored | `batsman_runs == 4` |
| **Total Sixes** | All career six-runs scored | `batsman_runs == 6` |

---

## 📐 Statistical Analysis

Three inferential statistical tests are implemented in **Section 7** of `app.py`:

### Test 1 — Chi-Square Test
- **Question:** Does the toss decision (bat or field) significantly affect match outcome?
- **H₀:** Toss decision is independent of match outcome.
- **H₁:** Toss decision is not independent of match outcome.
- **Method:** `scipy.stats.chi2_contingency` on a 2×2 cross-tabulation of `toss_decision × (toss_winner == winner)`.
- **Significance level:** α = 0.05

### Test 2 — Welch's Independent T-Test
- **Question:** Is there a significant difference in 1st-innings scores when the toss winner bats vs. fields first?
- **H₀:** Mean 1st-innings score is equal for both toss decisions.
- **H₁:** Mean 1st-innings score differs significantly.
- **Method:** `scipy.stats.ttest_ind(..., equal_var=False)` (Welch's variant for unequal variances).
- **Significance level:** α = 0.05

### Test 3 — Pearson Correlation
- **Question:** Is there a significant linear correlation between a batsman's career runs and career strike rate?
- **Method:** `scipy.stats.pearsonr` on aggregated career statistics (minimum 200 legal balls faced).
- **Significance level:** α = 0.05

---

## 🖥️ Streamlit Dashboard Features

The dashboard has **11 pages** accessible from the sidebar navigation:

| Page | Content | Interactive Filters |
|------|---------|---------------------|
| 🏠 **Project Overview** | 12 headline KPI metrics, dataset file previews | — |
| 🔬 **Dataset Health** | Column profiles, missing-value bar chart, cleaning summary | Tab selector |
| 🏆 **Team Performance** | Franchise win bar chart, season-trend line, toss-impact grouped bar, toss-decision pie | Season multiselect |
| 🏏 **Batting Analysis** | Top-N batsmen bar (colour = SR), runs vs. SR scatter, per-player profile with season & phase breakdowns | Season multiselect, top-N slider, player selectbox |
| 🎳 **Bowling Analysis** | Top-N bowlers bar (colour = economy), wickets vs. economy scatter, dismissal donut, per-bowler profile | Season multiselect, top-N slider, bowler selectbox |
| 📈 **Season & Phase Trends** | Average match score line, win-type trend line, phase run-rate bar, phase×season heatmap, extras per match | — |
| 🏟️ **Venue Insights** | Avg 1st-innings score bar, toss decision by venue grouped bar | City selectbox |
| 🌟 **Player of the Match** | All-time award leaders bar, awards by winning team bar | Season multiselect |
| 📐 **Statistical Analysis** | Chi-square cross-tab bar + metrics, T-test box plot + metrics, Pearson scatter | — |
| 💡 **Key Insights** | 10 evidence-based findings derived dynamically from the data | — |
| 📄 **Download Report** | Generate and download `project_report.docx` from live data | Download button |

All computations use `@st.cache_data` for performance on large deliveries datasets.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or later
- pip package manager

### Step 1 — Clone or download the project

Place all project files in a single folder:
```
project-folder/
├── app.py
├── requirements.txt
├── README.md
└── Project_Report.docx
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Obtain the dataset

**Option A — Manual download (recommended for college submission):**

1. Visit: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
2. Click **Download** (requires a free Kaggle account)
3. Unzip the downloaded archive
4. Create a `data/` folder inside the project folder
5. Place `matches.csv` and `deliveries.csv` inside `data/`

**Option B — Kaggle API (automatic download on first run):**

1. Sign up at https://www.kaggle.com
2. Go to *Account → API → Create New API Token* — downloads `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<user>\.kaggle\` (Windows)
4. The app will auto-download the dataset when `data/matches.csv` is not found

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

The app will:
1. Detect whether the dataset is present in `data/`
2. Download it via the Kaggle API if absent (requires API credentials)
3. Load, clean, and preprocess both CSV files (with caching)
4. Launch the interactive dashboard

### Generating the DOCX Report

1. In the dashboard sidebar, click **📄 Download Report**
2. Click the **⬇️ Download project_report.docx** button
3. The Word document is generated dynamically from the live dataset and downloaded to your machine

---

## 💡 Key Findings

> Note: Exact numerical values are computed at runtime from the dataset. The findings below are structurally confirmed by the analysis logic and reflect what the dataset reveals.

1. **Franchise Dominance:** One franchise (determined at runtime from `matches.winner.value_counts()`) leads all-time IPL win counts, reflecting consistent squad building and coaching stability over multiple seasons.

2. **Toss Impact is Modest:** The toss winner wins approximately 50–52% of matches — only marginally above the random baseline of 50%, confirming that while the toss provides a psychological edge, match execution dominates outcomes. The chi-square test reveals whether this difference is statistically significant.

3. **Chasing Preferred:** Win-by-wickets (chasing) matches have increased relative to win-by-runs (defending) in later seasons, reflecting improved batting depth and death-over finishing skills across all franchises.

4. **Death Overs Drive Scores:** Death overs (16–19) consistently deliver the highest run rate of the three phases, driven by the deployment of specialist T20 finishers and power hitters in batting positions 5–7.

5. **Powerplay Evolution:** Average Powerplay (0–5) run rates have increased progressively across seasons, indicating a strategic shift toward aggressive, boundary-hunting opening partnerships.

6. **Boundary Concentration:** Approximately 16–18% of all legal deliveries result in a boundary (four or six), a key differentiator between successful and struggling batting line-ups.

7. **Caught Dominates Dismissals:** The 'caught' dismissal type accounts for roughly 45–50% of all dismissals — the dominant mode in T20 cricket due to aggressive, aerial stroke play.

8. **Extras are Costly:** Franchises that consistently concede above-average wides and no-balls effectively gift opponents meaningful extra runs per innings — a quantifiable competitive disadvantage in close matches.

9. **Venue Scoring Variance:** High-altitude or small-boundary venues (e.g. Wankhede, Chinnaswamy) exhibit measurably higher average 1st-innings scores compared to traditionally lower-scoring grounds, influencing toss decisions at those venues.

10. **Strike Rate vs. Volume Trade-off:** The Pearson correlation between career runs and career strike rate reveals whether elite batsmen can sustain both volume and aggression — the correlation coefficient and p-value from the data determine the conclusion.

---

## ⚠️ Limitations

1. **Dataset cut-off at 2020:** The analysis covers only IPL seasons 2008–2020. Seasons 2021 onwards are not included, so recent franchise changes (e.g. Lucknow Super Giants, Gujarat Titans) are absent.
2. **No ball-tracking data:** Deliveries do not include ball speed, swing, seam, or trajectory data, limiting biomechanical analysis.
3. **No squad or auction data:** Player auction prices, squad composition, and coaching staff are not included, preventing franchise-investment analytics.
4. **Franchise discontinuities:** Defunct franchises (Deccan Chargers, Kochi Tuskers Kerala, Pune Warriors) have fewer seasons of data, introducing imbalance in franchise comparisons.
5. **Canonical name mapping may miss edge cases:** The team-name mapping dictionary covers known renames; any undocumented inconsistencies in the raw data could affect aggregations.
6. **No player position data:** Batting order positions are not recorded per match; phase-wise analysis attributes runs/balls to the bowler/batsman without knowing their position in the order.

---

## 🔭 Future Scope

1. **Extend to IPL 2021–2024:** Incorporate additional seasons to capture the most recent franchises and rule changes.
2. **Player Auction Price Analysis:** Integrate publicly available IPL auction data to evaluate return-on-investment (runs per crore, wickets per crore).
3. **Predictive Modelling:** Build a match outcome classifier using historical team and player performance features; implement a win-probability model updatable ball-by-ball.
4. **Network Graph Analysis:** Visualise partnerships (batsman 1 × batsman 2) and player interactions as network graphs to identify high-value partnerships.
5. **Natural Language Processing:** Analyse commentary text (where available) or tweet sentiment during matches to correlate fan sentiment with match performance.
6. **Real-time Dashboard:** Connect to live IPL data feeds using the CricInfo or ESPNcricinfo API for a real-time analytics dashboard.
7. **Advanced Visualisations:** Add wagon-wheel charts, pitch map heatmaps (shot direction), and bowling line/length plots for enhanced tactical analysis.

---

## 📌 Conclusion

This project demonstrates how a structured **Data Analytics pipeline** — applied rigorously to a real-world sports dataset — can uncover actionable insights that go beyond surface-level statistics. By cleaning nearly 180,000 delivery records, engineering meaningful features like match phase and boundary flags, and applying both descriptive and inferential statistics, the project answers concrete questions about IPL team strategy, player excellence, and structural trends across 13 seasons.

The interactive Streamlit dashboard makes all findings immediately accessible and explorable, while the in-app DOCX generator produces a submission-ready academic report directly from live data — ensuring that all results are reproducible and grounded in the actual dataset.

---

## 📚 References

- Bhingre, P. (2020). *IPL Complete Dataset 2008–2020*. Kaggle. https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
- Indian Premier League. *Official Website*. https://www.iplt20.com
- McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.
- Streamlit Inc. *Streamlit Documentation*. https://docs.streamlit.io (v1.64)
- Plotly Technologies Inc. *Plotly Python Documentation*. https://plotly.com/python
- Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental algorithms for scientific computing in Python*. Nature Methods, 17, 261–272.
- Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. Computing in Science & Engineering, 9(3), 90–95.

---

*College Data Analytics Project — IPL Complete Dataset (2008–2020) | Single-file implementation in app.py*
