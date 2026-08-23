import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def resolve_machine_id_col(df: pd.DataFrame) -> str:
    """Detects whether the machine ID column is 'machineID' or 'machine_id'."""
    for candidate in ["machineID", "machine_id"]:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"No machine ID column found. Available columns: {list(df.columns)}"
    )


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Clean telemetry data: parse datetimes, drop duplicates, sort, and interpolate/fill."""
    df = df.copy()
    id_col = resolve_machine_id_col(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates()
    df = df.sort_values(by=[id_col, "datetime"]).reset_index(drop=True)

    sensor_cols = ["volt", "rotate", "pressure", "vibration"]
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    df[sensor_cols] = df.groupby(id_col)[sensor_cols].transform(
        lambda grp: grp.ffill().bfill()
    )

    return df


def clean_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Clean error logs: parse datetimes, drop duplicates, and sort."""
    df = df.copy()
    id_col = resolve_machine_id_col(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates()

    return df.sort_values(by=[id_col, "datetime"]).reset_index(drop=True)


def clean_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Clean failure records: parse datetimes, drop duplicates, and sort."""
    df = df.copy()
    id_col = resolve_machine_id_col(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates()

    return df.sort_values(by=[id_col, "datetime"]).reset_index(drop=True)


def clean_maint(df: pd.DataFrame) -> pd.DataFrame:
    """Clean maintenance logs: parse datetimes, drop duplicates, and sort."""
    df = df.copy()
    id_col = resolve_machine_id_col(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates()

    return df.sort_values(by=[id_col, "datetime"]).reset_index(drop=True)


def clean_machines(df: pd.DataFrame) -> pd.DataFrame:
    """Clean machine metadata: drop duplicates, strip whitespace, handle missing values."""
    df = df.copy()
    df = df.drop_duplicates()

    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip()
        df["model"] = df["model"].replace("nan", np.nan).fillna("Unknown")

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median()).astype(int)

    return df


def build_unified_dataset(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all cleaned datasets into a single unified hourly DataFrame."""

    merged = telemetry.copy()
    id_col = resolve_machine_id_col(merged)

    # 1. Error Events (One-Hot Encoded)
    if not errors.empty:
        errors_encoded = pd.get_dummies(
            errors,
            columns=["errorID"],
            prefix="err",
            dtype=int,
        )

        errors_agg = (
            errors_encoded
            .groupby([id_col, "datetime"])
            .max()
            .reset_index()
        )

        merged = pd.merge(
            merged,
            errors_agg,
            on=[id_col, "datetime"],
            how="left",
        )

        err_cols = [
            c for c in merged.columns
            if c.startswith("err_")
        ]

        merged[err_cols] = (
            merged[err_cols]
            .fillna(0)
            .astype(int)
        )

    # 2. Failure Events
    if not failures.empty:
        failures_renamed = failures.rename(
            columns={"failure": "comp_failure"}
        )

        merged = pd.merge(
            merged,
            failures_renamed[
                [id_col, "datetime", "comp_failure"]
            ],
            on=[id_col, "datetime"],
            how="left",
        )

        merged["comp_failure"] = (
            merged["comp_failure"]
            .fillna("none")
        )

    else:
        merged["comp_failure"] = "none"

    # 3. Maintenance Events
    if not maint.empty:
        maint_encoded = pd.get_dummies(
            maint,
            columns=["comp"],
            prefix="maint",
            dtype=int,
        )

        maint_agg = (
            maint_encoded
            .groupby([id_col, "datetime"])
            .max()
            .reset_index()
        )

        merged = pd.merge(
            merged,
            maint_agg,
            on=[id_col, "datetime"],
            how="left",
        )

        maint_cols = [
            c for c in merged.columns
            if c.startswith("maint_")
        ]

        merged[maint_cols] = (
            merged[maint_cols]
            .fillna(0)
            .astype(int)
        )

    # 4. Static Machine Metadata
    if not machines.empty:
        merged = pd.merge(
            merged,
            machines,
            on=id_col,
            how="left",
        )

    return merged


def engineer_features(
    df: pd.DataFrame,
    prediction_window_hours: int = 24,
) -> pd.DataFrame:
    """
    Compute lag-based rolling features without lookahead bias
    and formulate the predictive maintenance target window.

    Important:
    - Rolling telemetry features are shifted by 1 timestep.
    - No backward filling is performed after the shift.
    - Initial observations without historical data are filled with
      a constant zero rather than future observations.
    """

    df = df.copy()
    id_col = resolve_machine_id_col(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = (
        df.sort_values(by=[id_col, "datetime"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 1. Target: Failure in the NEXT N hours
    # ---------------------------------------------------------
    if "comp_failure" in df.columns:
        immediate_failure = (
            df["comp_failure"].notna()
            & (df["comp_failure"] != "none")
        ).astype(int)

    elif "failed" in df.columns:
        immediate_failure = df["failed"].astype(int)

    else:
        immediate_failure = pd.Series(
            0,
            index=df.index,
        )

    immediate_failure.name = "_immediate_failure"
    df["_immediate_failure"] = immediate_failure

    # FixedForwardWindowIndexer looks forward from the current
    # timestep to construct the predictive target.
    #
    # This is intentional because the target represents whether
    # a failure occurs in the NEXT N hours.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(
        window_size=prediction_window_hours
    )

    df["target_failure_window"] = (
        df.groupby(id_col)["_immediate_failure"]
        .apply(
            lambda s: (
                s.rolling(
                    window=indexer,
                    min_periods=1,
                )
                .sum()
                > 0
            ).astype(int)
        )
        .reset_index(level=0, drop=True)
    )

    df = df.drop(columns=["_immediate_failure"])

    # ---------------------------------------------------------
    # 2. Lagged Rolling Telemetry
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # shift(1) ensures that the current timestep is NEVER
    # included in its own rolling feature.
    #
    # We deliberately DO NOT use .bfill() here.
    #
    # Example:
    #
    # t0 -> no historical observation exists
    # t1 -> uses t0
    # t2 -> uses t0, t1
    #
    # The t0 value cannot legitimately use t1 or any later
    # observation.
    #
    # Missing initial values are therefore replaced with 0,
    # which is a synthetic constant and contains no future
    # information.
    # ---------------------------------------------------------

    sensor_cols = [
        c
        for c in ["volt", "rotate", "pressure", "vibration"]
        if c in df.columns
    ]

    for col in sensor_cols:
        for window in [3, 24]:

            # Historical rolling mean.
            #
            # shift(1) is applied BEFORE rolling, ensuring that
            # the current observation is excluded.
            mean_feature = (
                df.groupby(id_col)[col]
                .transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(
                            window=window,
                            min_periods=1,
                        )
                        .mean()
                    )
                )
            )

            # First observation of each machine has no history.
            # Use zero rather than bfill() to avoid lookahead.
            df[f"{col}_mean_{window}h"] = (
                mean_feature.fillna(0)
            )

            # Historical rolling standard deviation.
            std_feature = (
                df.groupby(id_col)[col]
                .transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(
                            window=window,
                            min_periods=1,
                        )
                        .std()
                    )
                )
            )

            # With fewer than two historical observations,
            # standard deviation is undefined. Zero is used
            # without introducing future information.
            df[f"{col}_std_{window}h"] = (
                std_feature.fillna(0)
            )

    # ---------------------------------------------------------
    # 3. Categorical Encoding
    # ---------------------------------------------------------
    if "model" in df.columns:
        df = pd.get_dummies(
            df,
            columns=["model"],
            drop_first=True,
            dtype=int,
        )

    return df


def split_and_scale_features(
    df: pd.DataFrame,
    split_date: str = "2015-10-01",
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Time-based train/test split with strictly leakage-free feature scaling.
    """

    id_col = resolve_machine_id_col(df)

    non_feature_cols = [
        "datetime",
        id_col,
        "failed",
        "comp_failure",
        "target_failure_window",
    ]

    # Only keep numeric feature columns.
    candidate_cols = [
        c
        for c in df.columns
        if c not in non_feature_cols
        and not c.startswith("comp_failure_")
    ]

    feature_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    dropped_cols = [
        c
        for c in candidate_cols
        if c not in feature_cols
    ]

    if dropped_cols:
        print(
            f"[WARN] Dropping non-numeric columns from features: "
            f"{dropped_cols}"
        )

    train_mask = (
        df["datetime"] < pd.to_datetime(split_date)
    )

    train_df = (
        df[train_mask]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[~train_mask]
        .copy()
        .reset_index(drop=True)
    )

    # Fit scaler ONLY on training data.
    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(
        train_df[feature_cols]
    )

    test_df[feature_cols] = scaler.transform(
        test_df[feature_cols]
    )

    return train_df, test_df, scaler


if __name__ == "__main__":
    # Resolve the project root (one level up from pipelines/)
    project_root = Path(__file__).resolve().parents[1]

    # Target path:
    # datasets/clean_dataset/PdM_combined_cleaned.csv
    input_file = (
        project_root
        / "datasets"
        / "clean_dataset"
        / "PdM_combined_cleaned.csv"
    )

    output_dir = (
        project_root
        / "datasets"
        / "model_train_dataset"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading dataset from: {input_file}")

    cleaned_df = pd.read_csv(input_file)

    print(
        f"Columns found: {list(cleaned_df.columns)}"
    )

    print("Engineering features...")

    engineered_df = engineer_features(
        cleaned_df,
        prediction_window_hours=24,
    )

    print(
        "Performing temporal split and scaling..."
    )

    train_data, test_data, scaler = (
        split_and_scale_features(
            engineered_df,
            split_date="2015-10-01",
        )
    )

    train_out = (
        output_dir / "train_scaled.csv"
    )

    test_out = (
        output_dir / "test_scaled.csv"
    )

    train_data.to_csv(
        train_out,
        index=False,
    )

    test_data.to_csv(
        test_out,
        index=False,
    )

    print(
        f"Saved outputs to:\n"
        f" - {train_out}\n"
        f" - {test_out}"
    )