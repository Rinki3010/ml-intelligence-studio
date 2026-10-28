# Forma Data Profiling — Expanded Component Library

This update expands the existing real-data profiling stage. It does not insert synthetic sample results; each analysis runs against the uploaded CSV through `POST /api/v1/datasets/{dataset_id}/profiling/analyze`.

## Component groups

- **Structure:** dataset overview, schema, row/column counts, duplicates and memory footprint.
- **Quality:** missingness, IQR/Z-score outliers, constant/near-constant features, identifier-like columns and high-cardinality clues.
- **Statistics:** numeric summaries, category frequencies and categorical profile.
- **Visual analysis:** histograms/box summaries, date/time coverage and patterns, pairwise scatter plots.
- **Relationships:** Pearson/Spearman/Kendall correlation, covariance and VIF multicollinearity checks.
- **Hypothesis testing:** Shapiro–Wilk normality, Welch/Mann–Whitney two-group tests, ANOVA/Kruskal–Wallis and chi-square independence.
- **Regression diagnostics:** simple linear regression, fitted values, residuals, R², MAE and RMSE.
- **Feature relevance:** feature–target association/mutual information and target/class balance.

Charts use a multi-colour palette and are rendered from backend results. Statistical tests report p-values and selected alpha; statistical significance does not prove causation or practical importance. Identifier-like and target-leakage clues are heuristics that need human review.

## Run

Backend (from `source/backend`, inside your activated virtual environment):

```powershell
python -m pytest -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (from `source/frontend`):

```powershell
npm install
npm run build
npm run dev
```

If the frontend build has not been run, install dependencies from `package.json` first. The backend test suite covers the expanded component registry and key new analyses.
