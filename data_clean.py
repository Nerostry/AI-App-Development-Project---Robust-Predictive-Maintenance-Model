import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set up clean logging output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PredictiveMaintenancePreprocessor:
    """Handles dataset loading, merging, feature imputation using KNN,

    and target engineering for Predictive Maintenance data.
    """

    def __init__(self, data_dir: Union[str, Path] = "/content"):
        self.data_dir = Path(data_dir)
        self.df_combined: Optional[pd.DataFrame] = None

    def load_and_merge(self) -> pd.DataFrame:
        """Loads raw CSV files, converts timestamps, deduplicates records,

        and merges them into a single base DataFrame.
        """
        logging.info("Loading dataset files...")
        file_map = {
            "telemetry": self.data_dir / "PdM_telemetry.csv",
            "machines": self.data_dir / "PdM_machines.csv",
            "failures": self.data_dir / "PdM_failures.csv",
            "errors": self.data_dir / "PdM_errors.csv",
            "maint": self.data_dir / "PdM_maint.csv",
        }

        # Validate file existence
        for name, path in file_map.items():
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {path}")

        # Read CSVs
        dfs = {name: pd.read_csv(path) for name, path in file_map.items()}

        # Deduplicate individual tables prior to merging
        for name in dfs:
            dfs[name] = dfs[name].drop_duplicates()

        # Convert datetime columns
        for key in ["telemetry", "failures", "errors", "maint"]:
            dfs[key]["datetime"] = pd.to_datetime(dfs[key]["datetime"])

        logging.info("Merging datasets...")
        # Step 1: Base merge (telemetry + machine static metadata)
        merged = pd.merge(dfs["telemetry"], dfs["machines"], on="machineID", how="left")

        # Step 2: Left joins with event tables
        merged = pd.merge(merged, dfs["failures"], on=["datetime", "machineID"], how="left")
        merged = pd.merge(merged, dfs["errors"], on=["datetime", "machineID"], how="left")
        merged = pd.merge(merged, dfs["maint"], on=["datetime", "machineID"], how="left")

        # Ensure no accidental duplicates were created during joins
        self.df_combined = merged.drop_duplicates()
        logging.info(f"Datasets successfully merged. Initial shape: {self.df_combined.shape}")
        return self.df_combined

    def impute_and_engineer_features(
        self,
        feature_cols: Optional[List[str]] = None,
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """Combines failure/maintenance component columns, imputes missing component failures

        using KNN, and creates the target binary flag `failed`.
        """
        if self.df_combined is None:
            raise ValueError("Data has not been merged yet. Call `load_and_merge()` first.")

        df = self.df_combined.copy()

        # 1. Combine 'comp' (from maintenance) and 'failure' into 'comp_failure'
        if "failure" in df.columns and "comp" in df.columns:
            df["comp_failure"] = df["failure"].combine_first(df["comp"])
            df.drop(columns=["failure", "comp"], inplace=True)
        elif "failure" in df.columns:
            df["comp_failure"] = df["failure"]
            df.drop(columns=["failure"], inplace=True)

        # Default features for KNN imputation
        if feature_cols is None:
            feature_cols = ["volt", "rotate", "pressure", "vibration", "model", "age"]

        # 2. Identify missing target values associated with error events
        train_mask = df["comp_failure"].notna()
        target_mask = df["comp_failure"].isna() & df["errorID"].notna()

        # 3. Perform KNN Imputation if missing targets exist
        if target_mask.sum() > 0 and train_mask.sum() > 0:
            logging.info(f"Imputing {target_mask.sum()} missing component failure records using KNN...")

            # Feature Encoding & Scaling
            X_encoded = pd.get_dummies(df[feature_cols], columns=["model"], drop_first=True)
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_encoded),
                index=df.index,
                columns=X_encoded.columns,
            )

            X_train = X_scaled[train_mask]
            y_train = df.loc[train_mask, "comp_failure"].astype(str)
            X_target = X_scaled[target_mask]

            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            predicted_comps = knn.predict(X_target)

            # Assign predicted component failure labels back to masked rows
            df.loc[target_mask, "comp_failure"] = predicted_comps
        else:
            logging.info("No rows required KNN imputation.")

        # 4. Create target indicator column 'failed'
        df["failed"] = df["comp_failure"].notna().astype(int)

        self.df_combined = df
        return self.df_combined

    def save(self, output_path: Union[str, Path] = "df_combined.csv") -> Path:
        """Saves the final clean dataset to a CSV file."""
        if self.df_combined is None:
            raise ValueError("No DataFrame available to save. Run pipeline processing first.")

        out_file = Path(output_path)
        self.df_combined.to_csv(out_file, index=False)
        logging.info(f"Saved processed dataset to: {out_file.resolve()}")
        return out_file

    def run_pipeline(self, output_path: Union[str, Path] = "df_combined.csv") -> pd.DataFrame:
        """Executes the full pipeline sequentially: load/merge -> impute -> save."""
        self.load_and_merge()
        self.impute_and_engineer_features()
        self.save(output_path=output_path)
        return self.df_combined


# --- Execution ---
if __name__ == "__main__":
    # Initialize processor with data directory
    processor = PredictiveMaintenancePreprocessor(data_dir="/content")

    # Run the complete end-to-end processing pipeline
    processed_df = processor.run_pipeline(output_path="df_combined.csv")