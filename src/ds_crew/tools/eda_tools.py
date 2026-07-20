"""Read-only, deterministic EDA tooling. Never mutates the dataset."""

from __future__ import annotations


import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel

from ds_crew import settings
from ds_crew.schemas import ColumnProfile, EdaReport, TaskType
from ds_crew.state import get_data_store


def infer_task_type(y: pd.Series, max_classes: int | None = None) -> TaskType:
    max_classes = max_classes if max_classes is not None else settings.MAX_CLASSIFICATION_CLASSES
    if y.dtype == object or str(y.dtype) in ("category", "bool"):
        return "classification"
    return "classification" if y.nunique(dropna=True) <= max_classes else "regression"


def _profile_column(s: pd.Series, name: str) -> ColumnProfile:
    non_null = s.dropna()
    n_unique = int(s.nunique(dropna=True))
    is_id_like = len(non_null) > 1 and n_unique == len(non_null)
    is_constant = n_unique <= 1
    samples = [str(v) for v in non_null.unique()[:5]]
    return ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        null_pct=round(float(s.isna().mean() * 100), 2),
        n_unique=n_unique,
        is_constant=is_constant,
        is_id_like=is_id_like,
        sample_values=samples,
    )


def _correlations_with_target(
    df: pd.DataFrame, target: str, task_type: TaskType, cols: list[str]
) -> dict[str, float]:
    numeric_cols = [c for c in cols if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {}
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df[target]):
            return {}
        y = df[target]
    else:
        uniques = df[target].dropna().unique()
        if len(uniques) != 2:
            return {}
        mapping = {v: i for i, v in enumerate(sorted(uniques, key=str))}
        y = df[target].map(mapping)
    corrs: dict[str, float] = {}
    for c in numeric_cols:
        corr = df[c].corr(y)
        if corr is not None and not pd.isna(corr):
            corrs[c] = round(float(corr), 4)
    return corrs


def _near_duplicate_pairs(
    df: pd.DataFrame, cols: list[str], threshold: float = 0.98
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if df[c1].astype(str).equals(df[c2].astype(str)):
                pairs.append((c1, c2))
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1 :]:
                if (c1, c2) in pairs or (c2, c1) in pairs:
                    continue
                v = corr_matrix.loc[c1, c2]
                if pd.notna(v) and v > threshold:
                    pairs.append((c1, c2))
    return pairs


def _leakage_flags(
    columns: list[ColumnProfile], correlations: dict[str, float], target: str
) -> list[str]:
    flags: list[str] = []
    for cp in columns:
        if cp.name == target:
            continue
        if cp.is_id_like:
            flags.append(f"Column '{cp.name}' looks like an identifier (all values unique).")
    for col, corr in correlations.items():
        if abs(corr) > 0.99:
            flags.append(
                f"Column '{col}' is near-perfectly correlated with the target ({corr}); "
                "possible leakage."
            )
    return flags


def build_eda_report(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType,
    include_correlations: bool = True,
    detailed_column_limit: int | None = None,
) -> EdaReport:
    detailed_column_limit = (
        detailed_column_limit
        if detailed_column_limit is not None
        else settings.EDA_DETAILED_COLUMN_LIMIT
    )
    ordered_cols = [target] + [c for c in df.columns if c != target]
    truncated = len(ordered_cols) > detailed_column_limit
    profiled_cols = ordered_cols[:detailed_column_limit] if truncated else ordered_cols

    columns = [_profile_column(df[c], c) for c in profiled_cols]
    feature_cols = [c for c in profiled_cols if c != target]

    correlations = (
        _correlations_with_target(df, target, task_type, feature_cols)
        if include_correlations
        else {}
    )
    near_dupes = _near_duplicate_pairs(df, feature_cols)

    class_balance = None
    if task_type == "classification":
        vc = (df[target].value_counts(normalize=True) * 100).round(2)
        class_balance = {str(k): float(v) for k, v in vc.items()}

    leakage_flags = _leakage_flags(columns, correlations, target)

    notes = ""
    if truncated:
        notes = (
            f"Dataset has {len(ordered_cols)} columns; detailed profiling was limited to the "
            f"first {detailed_column_limit} (target always included). Increase "
            "EDA_DETAILED_COLUMN_LIMIT to profile more."
        )

    return EdaReport(
        run_id="",  # filled in by the caller (EdaSummaryTool), which knows the run_id
        n_rows=len(df),
        n_cols=len(df.columns),
        target=target,
        task_type=task_type,
        columns=columns,
        correlations_with_target=correlations,
        near_duplicate_column_pairs=near_dupes,
        class_balance=class_balance,
        leakage_flags=leakage_flags,
        truncated=truncated,
        notes=notes,
    )


class EdaSummaryInput(BaseModel):
    include_correlations: bool = True


class EdaSummaryTool(BaseTool):
    name: str = "eda_summary"
    description: str = (
        "Read-only. Profiles the current dataset for this run: schema, missingness, "
        "cardinality, target correlation, near-duplicate columns, class balance, and "
        "leakage flags. Never mutates data. Call this before proposing any cleaning "
        "or feature engineering plan."
    )
    args_schema: type[BaseModel] = EdaSummaryInput
    run_id: str = ""

    def _run(self, include_correlations: bool = True) -> str:
        state = get_data_store().get(self.run_id)
        if state.task_type is None:
            state.task_type = infer_task_type(state.df[state.target])
        report = build_eda_report(state.df, state.target, state.task_type, include_correlations)
        report.run_id = self.run_id
        state.record(
            "eda", "profile_computed", {"n_cols": report.n_cols, "truncated": report.truncated}
        )
        return report.model_dump_json()
