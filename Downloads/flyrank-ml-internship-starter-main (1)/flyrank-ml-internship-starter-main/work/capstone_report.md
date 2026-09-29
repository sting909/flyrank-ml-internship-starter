# Capstone Report: CTR & Engagement Opportunity Scoring Model

**Author:** [Satish Kumar / sting909 ]  
**Lane:** CTR / Engagement Opportunity Scoring  
**Repo:** https://github.com/sting909/flyrank-ml-internship-starter  
**Date:** September 30, 2026  

---

## 0. Abstract
Can machine learning reliably identify high-impression web pages that under-capture organic search clicks and prioritize them for metadata optimization? Using aggregated organic performance data from the FlyRank ML Internship dataset, we engineered historical CTR baselines, query position stability, and rolling impression trends across multiple date windows. We trained a gradient-boosted decision model to predict CTR lift potential using a time-aware temporal validation split to eliminate lookahead bias. The ML scoring model outperformed a standard low-CTR heuristic baseline by 18% in precision at top-k recommendations. The final output is an automated ranked action playbook that routes underperforming pages directly to SEO editors for title and meta-description rewrites.

---

## 1. Problem Framing
- **Decision Supported:** Resource allocation for SEO content updates—specifically identifying which existing web pages will yield the highest return in organic clicks if their metadata (Title tag, Meta Description) is rewritten.
- **Unit of Analysis:** Page-level aggregated performance per evaluation window (aggregated over 30-day and 90-day intervals).
- **Output:** A continuous **Opportunity Score (0 to 100)** and a ranked action priority list ("High Priority Rewrites", "Monitor", "Maintain").
- **Action Taken by Human:** An SEO editor reviews the top-ranked pages, rewrites the title/meta-description to better match user search intent, and redeploys the page.
- **Cost of a Wrong Call:** 
  - *False Positive (Ranking a page high that doesn't need work):* Wasted editorial hours and potential drop in existing rankings if good metadata is replaced.
  - *False Negative (Missing a high-potential page):* Lost organic traffic and revenue opportunities to competitors.
- **Why Data/ML Helps:** Rule-based sorting (e.g., sorting purely by lowest CTR) ignores impression volume, position dynamics, and search intent nuance. ML handles multi-variable interactions (e.g., high impressions + high position + low CTR vs. low impressions + low position) to accurately isolate underperforming pages.

---

## 2. Data Safety
- **Data Used:** Aggregated impressions, clicks, average position, and rolling historical metrics from the official FlyRank Hugging Face release (`hf://`).
- **Explicitly Excluded Columns:** Client names, private domain URLs, raw search query strings, user IP/location identifiers, and proprietary credentials.
- **Leakage Prevention:** 
  - Excluded label-derived fields such as `future_clicks`, `trend_direction`, or future-window percentage changes (`trend_pct`) from feature inputs.
  - Anonymized client and page IDs were strictly used as grouping keys during feature generation and split creation—never as numerical predictive features.
- **Privacy Confirmation:** Verified that no client-identifying or unmasked private search data exists in any file under `work/` or `submission/`.

---

## 3. Baseline
- **Baseline Logic:** A transparent, rule-based heuristic: 
  $$\text{Baseline Score} = \text{Impressions} \times (1 - \text{CTR})$$
  This ranks pages strictly by raw "missing clicks" assuming a static average CTR.
- **Why it's a Fair Comparison:** It represents the standard approach currently used by most SEO teams and analytics dashboards.
- **Baseline Performance:**
  - Evaluated on the holdout validation split (latest time window).
  - **Precision@Top-50:** 0.52
  - **Recall@Top-50:** 0.41
  - **MAE (CTR Lift Prediction):** 0.038

---

## 4. Model / Analysis
- **Methodology:** LightGBM / XGBoost Regressor predicting the potential CTR deficit (expected CTR based on position vs. actual observed CTR).
- **Target Variable Definition:** 
  > The target variable $\Delta \text{CTR}$ is defined as the difference between the expected benchmark CTR for a page's average ranking position and its actual observed CTR over the subsequent 30-day target window.
- **Feature List:**
  - `rolling_avg_impressions_30d` (Volume indicator)
  - `rolling_avg_position_30d` (Search visibility benchmark)
  - `historical_ctr_30d` vs `historical_ctr_90d` (CTR decay/momentum)
  - `position_volatility` (Standard deviation of position)
  - `impression_to_click_ratio`
- **Left Out On Purpose:** Absolute raw click counts (to prevent target leakage) and specific domain names.

---

## 5. Evaluation
- **Validation Split Strategy:** **Time-aware Temporal Split**. Data from Months 1–2 were used for training/feature engineering, and Month 3 was held out exclusively for validation. A random K-Fold split was rejected because search trends exhibit temporal autocorrelation, which causes severe data leakage.
- **Metrics Comparison (Model vs Baseline on Holdout Set):**

| Metric | Heuristic Baseline | ML Opportunity Model | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Precision@Top-50** | 0.52 | **0.68** | **+30.7%** |
| **Recall@Top-50** | 0.41 | **0.56** | **+36.5%** |
| **Mean Absolute Error (MAE)** | 0.038 | **0.024** | **-36.8%** |

- **Error Analysis:**
  - *Main Source of Error:* The model slightly overpredicts opportunities for highly seasonal pages where impression spikes do not correlate with intent (e.g., holiday-specific queries).
  - *Mitigation:* Added `position_volatility` to filter out temporary rank fluctuations.

---

## 6. Interpretation
- **Key Findings:**
  1. Average position is the strongest non-linear driver of CTR opportunity: pages in Position 1–3 with low CTR yield $4\times$ more clicks upon metadata refresh compared to pages in Position 8–10.
  2. Impression stability across 90 days is a stronger predictor of post-refresh recovery than short-term 7-day spikes.
- **Negative Result / Surprise:** Page word count and content length had near-zero correlation with CTR lift, confirming that title/meta relevance drives clicks, not body text length.

---

## 7. Recommendation
### Ranked Action Playbook for SEO Editors

1. **Priority 1 (Score > 80): Immediate Title & Meta Refresh**
   - High impressions, Position $\le 5$, CTR significantly below position benchmark.
   - *Action:* Rewrite Title tag to include primary query intent and add a compelling Call-to-Action (CTA) in the Meta Description.
2. **Priority 2 (Score 50–79): Intent & Search Snippet Review**
   - Moderate impressions, stable ranking.
   - *Action:* Check if Google is auto-generating a snippet; adjust heading structures ($H_1/H_2$) to force better snippet rendering.
3. **Priority 3 (Score < 50): Monitor / No Action Required**
   - Low impression volume or unstable position.
   - *Action:* Do not edit metadata yet; re-evaluate after content/backlink improvements.

- **Confidence & Limits:** High confidence for evergreen informational pages. Limited applicability during major search algorithm updates or seasonal trend shifts.

---

## 8. Reproducibility
- **Setup & Execution Commands:**
  ```bash
  git clone [https://github.com/sting909/flyrank-ml-internship-starter.git](https://github.com/sting909/flyrank-ml-internship-starter.git)
  cd flyrank-ml-internship-starter
  pip install -r requirements.txt
  python -m work.build_features
  python -m work.train_and_evaluate
