# Forma Data Cleaning & Preparation

This stage follows Data Profiling and operates on the uploaded CSV using a reusable operation recipe.

## UI

- Open **Data Cleaning** from the Forma sidebar.
- Choose an operation group and transformation.
- Select columns and configure supported parameters.
- Add one or more operations to the recipe. Operations execute in displayed order.
- Select **Preview changes** to run against the source data in memory.
- Review before/after row counts, column counts, missing cells, duplicate rows, the execution log, and resulting data preview.
- Select **Apply & save new dataset** to write a separate cleaned CSV. The source upload is not overwritten; the new dataset becomes the active dataset in Forma.

## Supported groups

1. **Missing data & duplicates:** remove duplicate rows; remove rows with missing values; impute missing values using mean, median, mode, or a constant.
2. **Structure & types:** drop selected columns; drop constant columns; convert selected columns to numeric, text, category, or datetime.
3. **Text standardization:** trim whitespace; normalize text case; replace a specified value.
4. **Outlier treatment:** cap outliers at IQR fences; remove rows outside IQR fences.
5. **Preview and quality comparison:** before/after metrics, operation-by-operation logs, preview table, and CSV download.

## API

- `GET /api/v1/cleaning/operations`
- `POST /api/v1/datasets/{dataset_id}/cleaning/preview`
- `POST /api/v1/datasets/{dataset_id}/cleaning/apply`
- `GET /api/v1/datasets/{dataset_id}/download`

Request example:

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

Implementation: `backend/app/pipeline/cleaning.py`; FastAPI routes: `backend/app/main.py`; UI: `frontend/src/DataCleaning.tsx`.

## Start and test

Backend:

```powershell
python -m pytest -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```powershell
npm install
npm run build
npm run dev
```

The backend tests passed (12 tests) in the build environment. Frontend dependency installation timed out in the build environment, so run the build commands locally to verify TypeScript compilation.

## Important ML note

Use preview before applying destructive operations. Type conversion failures become missing values and are reported. For predictive ML, transformations that learn parameters from data must be fitted on the training split only; use a scikit-learn pipeline to prevent leakage into validation/test data. See [scikit-learn's guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Stage 3 — Feature Engineering

Feature Engineering is available in the Forma sidebar and uses the active uploaded/cleaned dataset. It provides a selectable catalogue grouped into missing-value handling, scaling, numeric transforms, feature creation, date/time, text, categorical encoding, and feature selection. Users can configure a method, select columns, add it to an ordered recipe, reorder/remove steps, preview the transformed data, and apply the recipe to create a separate CSV-backed dataset. The original dataset is retained.

API endpoints:
- `GET /api/v1/feature-engineering/operations`
- `POST /api/v1/datasets/{dataset_id}/feature-engineering/preview`
- `POST /api/v1/datasets/{dataset_id}/feature-engineering/apply`

Implemented methods: numeric and categorical imputation; standard, min-max, robust and unit-norm scaling; log1p, square-root, Yeo–Johnson and quantile transforms; equal-width and quantile binning; polynomial and pairwise interaction features; datetime-part extraction; text character/word/digit counts; one-hot, ordinal and frequency encoding; drop selected columns; low-variance filtering; and correlation-based filtering.

Important ML note: this interactive stage materializes a dataset-level transformation for inspection/reuse. For unbiased model evaluation, learned preprocessing (imputation, scaling, category mappings, quantile/power transforms and feature selection) should be fitted only on training folds and then applied to validation/test data. The target selector protects the chosen target column from transformations.
