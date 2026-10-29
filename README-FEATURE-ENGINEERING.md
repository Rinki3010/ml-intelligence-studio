# Forma — Stage 3: Feature Engineering

## Goal
Let users choose feature transformations from a grouped catalogue, build an ordered recipe, preview the real transformed CSV, and save the output as a new dataset without overwriting the source.

## UI flow
1. Upload a CSV or open a dataset already used in the workflow.
2. Optionally choose a target column to protect it from transformation.
3. Filter the catalogue by group and select any method.
4. Select columns and configure method-specific settings.
5. Add one or more steps to the ordered recipe. Reorder or remove steps.
6. Preview to execute against the real CSV without saving.
7. Apply to write a new CSV-backed dataset; the original is retained.

## Groups and methods

### Missing values
- Numeric imputation: mean, median, constant.
- Categorical imputation: most frequent, constant.

### Scaling
- Standard scaling (zero mean/unit variance).
- Min-max scaling.
- Robust scaling (median/IQR based).
- Row-wise unit-norm normalization.

### Numeric transformations
- Log1p-derived feature.
- Square-root-derived feature.
- Yeo–Johnson power-transformed feature.
- Quantile transform to uniform or normal-like output.
- Equal-width or quantile binning with a configurable bin count.

### Feature creation
- Polynomial expansion, degree 2 or 3 (limited to 12 selected columns).
- Pairwise numeric interactions (limited to 12 selected columns).

### Date and time
- Year, month, day, weekday, hour, quarter, and weekend indicator.

### Text
- Character count, word count, and digit count.

### Categorical encoding
- One-hot encoding (limited to 100 generated categories per source column).
- Ordinal encoding.
- Frequency encoding.

### Feature selection
- Drop selected columns.
- Remove selected numeric columns below a variance threshold.
- Remove redundant numeric columns using an absolute-correlation threshold.

## API
- `GET /api/v1/feature-engineering/operations`
- `POST /api/v1/datasets/{dataset_id}/feature-engineering/preview`
- `POST /api/v1/datasets/{dataset_id}/feature-engineering/apply`

Request shape:
```json
{
  "operations": [
    {"type": "log_transform", "columns": ["revenue"]},
    {"type": "one_hot_encode", "columns": ["region"]}
  ],
  "target_column": "label",
  "preview_rows": 12
}
```

## Important ML caveat
This workbench can materialize a transformed dataset for inspection and reuse. For honest model evaluation, learned transforms—including imputation, scaling, category encoding, power/quantile transforms, and feature selection—should be fitted inside each training fold only, then applied to validation/test folds. Applying them to an entire dataset before splitting can leak information. Keep the target column selected in the protection control.

## Verification
Backend tests cover the catalogue, numeric feature creation, categorical/text methods, and target-column protection. Run from `source/backend` with the active virtual environment:
```powershell
python -m pytest -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Run the frontend from `source/frontend`:
```powershell
npm install
npm run build
npm run dev
```
