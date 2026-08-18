import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set up clean logging output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PredictiveMaintenancePreprocessor:
    """Handles combined dataset loading, feature imputation using KNN,

    and target engineering for Predictive Maintenance data.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "datasets/ingested_dataset",
        output_dir: Union[str, Path] = "datasets/clean_dataset",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.df_combined: Optional[pd.DataFrame] = None

    def load_data(self, filename: str = "PdM_combined.csv") -> pd.DataFrame:
        """Loads the pre-combined CSV file, converts timestamps, and deduplicates records."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required combined dataset file not found: {file_path.resolve()}"
            )

        logging.info(f"Loading combined dataset from {file_path}...")
        df = pd.read_csv(file_path)

        # Deduplicate records
        df = df.drop_duplicates()

        # Convert datetime column if present
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        self.df_combined = df
        logging.info(
            f"Dataset successfully loaded. Initial shape: {self.df_combined.shape}"
        )
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
            raise ValueError(
                "Data has not been loaded yet. Call `load_data()` first."
            )

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
            feature_cols = [
                "volt",
                "rotate",
                "pressure",
                "vibration",
                "model",
                "age",
            ]

        # Filter feature_cols to only those available in the DataFrame
        available_features = [col for col in feature_cols if col in df.columns]

        # 2. Identify missing target values associated with error events
        train_mask = df["comp_failure"].notna()
        target_mask = (
            df["comp_failure"].isna() & df["errorID"].notna()
            if "errorID" in df.columns
            else pd.Series(False, index=df.index)
        )

        # 3. Perform KNN Imputation if missing targets exist
        if target_mask.sum() > 0 and train_mask.sum() > 0:
            logging.info(
                f"Imputing {target_mask.sum()} missing component failure records using KNN..."
            )

            # Feature Encoding & Scaling
            if "model" in available_features:
                X_encoded = pd.get_dummies(
                    df[available_features], columns=["model"], drop_first=True
                )
            else:
                X_encoded = pd.get_dummies(
                    df[available_features], drop_first=True
                )

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

    def save(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Saves the final clean dataset to a CSV file in clean_dataset/."""
        if self.df_combined is None:
            raise ValueError(
                "No DataFrame available to save. Run pipeline processing first."
            )

        if output_path is None:
            out_file = self.output_dir / "PdM_combined_cleaned.csv"
        else:
            out_file = Path(output_path)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        self.df_combined.to_csv(out_file, index=False)
        logging.info(f"Saved processed dataset to: {out_file.resolve()}")
        return out_file

    def run_pipeline(
        self,
        input_filename: str = "PdM_combined.csv",
        output_filename: str = "PdM_combined_cleaned.csv",
    ) -> pd.DataFrame:
        """Executes the full pipeline sequentially: load -> impute -> save."""
        self.load_data(filename=input_filename)
        self.impute_and_engineer_features()
        self.save(output_path=self.output_dir / output_filename)
        return self.df_combined


if __name__ == "__main__":
    # Points automatically to datasets/ingested_dataset and datasets/clean_dataset
    processor = PredictiveMaintenancePreprocessor(
        data_dir="datasets/ingested_dataset",
        output_dir="datasets/clean_dataset",
    )

    processed_df = processor.run_pipeline(
        input_filename="PdM_combined.csv",
        output_filename="PdM_combined_cleaned.csv",
    )