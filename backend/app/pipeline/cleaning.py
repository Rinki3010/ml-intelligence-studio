"""Reusable, auditable CSV cleaning operations for Forma."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

SUPPORTED_OPERATIONS = [
    {"id":"drop_duplicates","name":"Remove duplicate rows","group":"Missing data & duplicates","description":"Keep the first occurrence of each exact duplicate row."},
    {"id":"drop_missing_rows","name":"Remove rows with missing values","group":"Missing data & duplicates","description":"Drop rows missing values in selected columns; use cautiously."},
    {"id":"fill_missing","name":"Impute missing values","group":"Missing data & duplicates","description":"Fill missing numeric or categorical values using a chosen strategy."},
    {"id":"drop_columns","name":"Drop selected columns","group":"Structure & types","description":"Remove columns explicitly selected by the user."},
    {"id":"drop_constant_columns","name":"Remove constant columns","group":"Structure & types","description":"Remove columns with zero or one observed unique value."},
    {"id":"convert_type","name":"Convert column type","group":"Structure & types","description":"Convert selected columns to numeric, text, category or datetime."},
    {"id":"trim_text","name":"Trim text whitespace","group":"Text standardization","description":"Trim leading and trailing whitespace in text columns."},
    {"id":"normalize_text","name":"Normalize text case","group":"Text standardization","description":"Normalize text values to lower or upper case."},
    {"id":"cap_outliers_iqr","name":"Cap outliers (IQR) ","group":"Outlier treatment","description":"Cap numeric values at IQR fences without removing rows."},
    {"id":"remove_outlier_rows_iqr","name":"Remove outlier rows (IQR)","group":"Outlier treatment","description":"Remove rows outside IQR fences for selected numeric columns."},
    {"id":"replace_value","name":"Replace a specific value","group":"Text standardization","description":"Replace a specified value in selected columns."},
]

def _quality(df: pd.DataFrame) -> dict[str, Any]:
    total = int(df.size)
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().sum().sum() / max(total, 1) * 100), 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
    }

def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    selected = columns or []
    unknown = [c for c in selected if c not in df.columns]
    if unknown: raise ValueError(f"Unknown column(s): {unknown}")
    return selected

def apply_operations(source: pd.DataFrame, operations: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = source.copy()
    logs: list[dict[str, Any]] = []
    if not operations: raise ValueError("Add at least one cleaning operation before previewing or applying.")
    for index, op in enumerate(operations, start=1):
        kind = str(op.get("type", ""))
        columns = _ensure_columns(df, [str(c) for c in op.get("columns", [])])
        before_rows, before_cols = len(df), len(df.columns)
        changed = 0
        detail = ""
        if kind == "drop_duplicates":
            subset = columns or None
            before = len(df); df = df.drop_duplicates(subset=subset, keep="first"); changed = before-len(df); detail = f"Removed {changed} duplicate row(s)."
        elif kind == "drop_missing_rows":
            subset = columns or list(df.columns)
            before = len(df); df = df.dropna(subset=subset, how="any"); changed = before-len(df); detail = f"Removed {changed} row(s) missing values in {len(subset)} selected column(s)."
        elif kind == "fill_missing":
            if not columns: raise ValueError("Select at least one column for missing-value imputation.")
            strategy = str(op.get("strategy", "median")); value = op.get("value", "Missing")
            for col in columns:
                missing = int(df[col].isna().sum())
                if not missing: continue
                s = df[col]
                if strategy in ("mean", "median"):
                    numeric = pd.to_numeric(s, errors="coerce")
                    if numeric.notna().sum() == 0: raise ValueError(f"Column {col!r} has no numeric values for {strategy} imputation.")
                    fill = numeric.mean() if strategy == "mean" else numeric.median()
                    df[col] = numeric.fillna(fill)
                elif strategy == "mode":
                    mode = s.mode(dropna=True)
                    fill = mode.iloc[0] if not mode.empty else value
                    df[col] = s.fillna(fill)
                elif strategy == "constant":
                    fill = value
                    if pd.api.types.is_numeric_dtype(s):
                        try: fill = float(value)
                        except (ValueError, TypeError): raise ValueError(f"Constant value for numeric column {col!r} must be numeric.")
                    df[col] = s.fillna(fill)
                else: raise ValueError("Imputation strategy must be mean, median, mode, or constant.")
                changed += missing
            detail = f"Filled {changed} missing cell(s) using {strategy}."
        elif kind == "drop_columns":
            if not columns: raise ValueError("Select at least one column to drop.")
            if len(columns) >= len(df.columns): raise ValueError("Cannot drop every column.")
            df = df.drop(columns=columns); changed = len(columns); detail = f"Dropped {len(columns)} column(s)."
        elif kind == "drop_constant_columns":
            candidates = columns or list(df.columns)
            drop = [c for c in candidates if df[c].nunique(dropna=True) <= 1]
            if len(drop) >= len(df.columns): raise ValueError("All columns are constant; refusing to create an empty dataset.")
            df = df.drop(columns=drop); changed = len(drop); detail = f"Dropped {len(drop)} constant column(s)."
        elif kind == "convert_type":
            if not columns: raise ValueError("Select at least one column to convert.")
            target = str(op.get("target_type", "numeric"))
            for col in columns:
                old_missing = int(df[col].isna().sum())
                if target == "numeric": df[col] = pd.to_numeric(df[col], errors="coerce")
                elif target == "text": df[col] = df[col].astype("string")
                elif target == "category": df[col] = df[col].astype("category")
                elif target == "datetime": df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                else: raise ValueError("Target type must be numeric, text, category, or datetime.")
                changed += max(0, int(df[col].isna().sum())-old_missing)
            detail = f"Converted {len(columns)} column(s) to {target}; {changed} additional value(s) could not be parsed and became missing."
        elif kind == "trim_text":
            targets = columns or [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
            for col in targets:
                mask = df[col].notna(); old = df.loc[mask, col].astype(str); new = old.str.strip(); changed += int((old != new).sum()); df.loc[mask, col] = new
            detail = f"Trimmed whitespace in {len(targets)} text column(s); {changed} value(s) changed."
        elif kind == "normalize_text":
            targets = columns or [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
            case = str(op.get("case", "lower"))
            if case not in ("lower", "upper"): raise ValueError("Text case must be lower or upper.")
            for col in targets:
                mask = df[col].notna(); old = df.loc[mask, col].astype(str); new = old.str.lower() if case == "lower" else old.str.upper(); changed += int((old != new).sum()); df.loc[mask, col] = new
            detail = f"Converted text to {case} case in {len(targets)} column(s); {changed} value(s) changed."
        elif kind in ("cap_outliers_iqr", "remove_outlier_rows_iqr"):
            if not columns: raise ValueError("Select at least one numeric column for IQR outlier treatment.")
            factor = float(op.get("factor", 1.5))
            if not 0 < factor <= 10: raise ValueError("IQR factor must be greater than 0 and at most 10.")
            masks = []
            for col in columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                valid = numeric.dropna()
                if valid.empty: raise ValueError(f"Column {col!r} has no numeric values.")
                q1, q3 = valid.quantile(.25), valid.quantile(.75); iqr = q3-q1; low, high = q1-factor*iqr, q3+factor*iqr
                if kind == "cap_outliers_iqr":
                    mask = numeric.notna(); old = numeric.loc[mask].copy(); df.loc[mask, col] = numeric.loc[mask].clip(low, high); changed += int((old != df.loc[mask, col]).sum())
                else: masks.append((numeric < low) | (numeric > high))
            if kind == "remove_outlier_rows_iqr":
                combined = pd.concat(masks, axis=1).fillna(False).any(axis=1); changed = int(combined.sum()); df = df.loc[~combined].copy(); detail = f"Removed {changed} row(s) outside IQR fences across {len(columns)} column(s)."
            else: detail = f"Capped {changed} value(s) at IQR fences across {len(columns)} column(s)."
        elif kind == "replace_value":
            if not columns: raise ValueError("Select at least one column for value replacement.")
            old = op.get("old_value"); new = op.get("new_value", "")
            if old is None: raise ValueError("Provide the value to replace.")
            for col in columns:
                before = df[col].astype("string"); changed += int((before == str(old)).sum()); df[col] = df[col].replace(old, new)
            detail = f"Replaced {changed} occurrence(s) across {len(columns)} column(s)."
        else:
            raise ValueError(f"Unsupported cleaning operation: {kind!r}")
        logs.append({"step": index, "type": kind, "columns": columns, "rows_before": int(before_rows), "rows_after": int(len(df)), "columns_before": int(before_cols), "columns_after": int(len(df.columns)), "changed": int(changed), "detail": detail})
        if df.empty: raise ValueError(f"Operation {index} left no rows. Adjust the configuration and preview again.")
        if len(df.columns) == 0: raise ValueError(f"Operation {index} removed every column.")
    return df.reset_index(drop=True), logs

def make_preview(df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    sample = df.head(limit).copy().astype(object).where(pd.notna(df.head(limit)), None)
    return [{str(k): (v.isoformat() if hasattr(v, "isoformat") else v.item() if isinstance(v, np.generic) else v) for k, v in row.items()} for row in sample.to_dict(orient="records")]
