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


# Data Cleaning & Preparation — Next Pipeline Stage

The update adds a dedicated **Data Cleaning & Preparation** page to the Forma sidebar. It uses the uploaded CSV and runs each operation in the order shown in the recipe. Users can preview the results before applying the recipe. Applying writes a new dataset file with a new dataset ID; the original upload is not overwritten, and the newly cleaned dataset becomes the active dataset in the UI.

## Cleaning groups and operations

- **Missing data & duplicates:** remove exact duplicate rows, remove rows with missing values, and impute missing values with mean, median, mode, or a custom constant.
- **Structure & types:** drop selected columns, remove constant columns, and convert selected columns to numeric, text, category, or datetime. Values that cannot be parsed during numeric/datetime conversion become missing and are reported.
- **Text standardization:** trim whitespace, normalize case to lowercase or uppercase, and replace a specified value.
- **Outlier treatment:** cap numeric values at IQR fences or remove rows outside IQR fences for selected numeric columns. Row removal is explicit because it can discard valid rare observations.
- **Preview and quality comparison:** compare row count, column count, missing cells and duplicate rows before and after; inspect the operation log and a resulting-row preview; download the saved cleaned CSV.

## Cleaning API

- `GET /api/v1/cleaning/operations` — list supported operations.
- `POST /api/v1/datasets/{dataset_id}/cleaning/preview` — run a recipe in memory and return before/after quality summaries, operation logs, and a preview without saving changes.
- `POST /api/v1/datasets/{dataset_id}/cleaning/apply` — run the same recipe and save the result as a new dataset. Returns the new dataset ID and download URL.
- `GET /api/v1/datasets/{dataset_id}/download` — download the dataset CSV.

Example request body:

```json
{
  "operations": [
    {"type": "fill_missing", "columns": ["age"], "strategy": "median"},
    {"type": "trim_text", "columns": ["city"]},
    {"type": "drop_duplicates", "columns": []}
  ],
  "preview_rows": 10
}
```

The backend cleaning logic is in `backend/app/pipeline/cleaning.py` and is kept separate from the API layer so the operations can be reused by future pipeline/tool integrations. The frontend page is `frontend/src/DataCleaning.tsx`.

## Safety and ML workflow notes

- Preview the recipe before applying it; dropping rows or columns is irreversible in the derived file. The source file remains intact.
- A parse failure during type conversion becomes a missing value and is counted in the operation log. Review the new missingness before continuing.
- This stage handles dataset-level cleaning. Learned transformations such as imputation/scaling/encoding used for model evaluation should be fitted on training data only, using a model pipeline, to avoid data leakage. See the [scikit-learn pipeline and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Verification

Backend tests: `python -m pytest -q` — 12 tests passed in the build environment. A direct FastAPI test-client smoke test also confirmed upload, preview and apply endpoints. The frontend dependency installation timed out in the build environment, so run `npm install` and `npm run build` locally to verify the frontend compile.
