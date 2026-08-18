import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set up clean logging output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PredictiveMaintenancePreprocessor:
    """Handles dataset ingestion, column normalization, domain boundary cleaning,

    sensor anomaly/outlier attenuation, missingness imputation, and target engineering.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "datasets/ingested_dataset",
        output_dir: Union[str, Path] = "datasets/clean_dataset",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, filename: str) -> pd.DataFrame:
        """Loads a CSV file from the ingested data directory, deduplicates, and parses dates."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path.resolve()}"
            )

        logging.info(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        df = df.drop_duplicates().reset_index(drop=True)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        self.df = df
        logging.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return self.df

    def standardize_and_sanitize_maintenance_data(self) -> pd.DataFrame:
        """Standardizes column names, applies boundary validations, and trims physical limits."""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call `load_data()` first.")

        df = self.df.copy()

        # 1. Standardize column names (snake_case, normalize machine ID and units)
        df.columns = (
            df.columns.str.strip()
            .str.replace("machineID", "machine_id", case=False)
            .str.lower()
            .str.replace(r"[\(\)\%]", "", regex=True)
            .str.replace("/", "_per_")
            .str.strip()
            .str.replace(" ", "_")
        )

        # 2. Domain boundary validation: load percentage (0 - 100)
        if "load_percentage" in df.columns:
            df["load_percentage"] = df["load_percentage"].clip(0, 100)

        # 3. Domain boundary validation: non-negative physical values
        potential_non_negative = [
            "operating_hours",
            "temperature_c",
            "vibration_level_mm_per_s",
            "maintenance_due_in_hours",
            "volt",
            "rotate",
            "pressure",
            "vibration",
            "age",
        ]
        active_non_neg = [col for col in potential_non_negative if col in df.columns]
        for col in active_non_neg:
            # Set negative sensor readings as NaN for subsequent imputation
            df.loc[df[col] < 0, col] = np.nan

        # 4. Filter or attenuate extreme sensor glitches (e.g. 5x vibration spikes)
        if "vibration_level_mm_per_s" in df.columns:
            # Physical limit cap for vibration sensors
            df.loc[df["vibration_level_mm_per_s"] > 15.0, "vibration_level_mm_per_s"] = np.nan

        if "temperature_c" in df.columns:
            # Physical limit cap for thermal sensors
            df.loc[df["temperature_c"] > 150.0, "temperature_c"] = np.nan

        self.df = df
        logging.info(f"Sanitization complete. Cleaned shape: {self.df.shape}")
        return self.df

    def impute_continuous_sensor_data(
        self,
        sensor_cols: Optional[List[str]] = None,
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """Imputes realistic sensor dropout/missingness using KNN over machine context."""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call `load_data()` first.")

        df = self.df.copy()

        if sensor_cols is None:
            sensor_cols = [
                "temperature_c",
                "vibration_level_mm_per_s",
                "load_percentage",
                "operating_hours",
                "age",
            ]

        available_sensors = [col for col in sensor_cols if col in df.columns]
        missing_counts = df[available_sensors].isna().sum().sum()

        if missing_counts > 0:
            logging.info(f"Imputing {missing_counts} missing sensor values across features...")
            
            # Encode categorical features for spatial distance computation
            cat_cols = [c for c in ["model"] if c in df.columns]
            df_for_impute = pd.get_dummies(df[available_sensors + cat_cols], drop_first=True)

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_for_impute)

            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_scaled = imputer.fit_transform(scaled_data)

            imputed_data = scaler.inverse_transform(imputed_scaled)
            df[available_sensors] = imputed_data[:, : len(available_sensors)]

        self.df = df
        return self.df

    def impute_and_engineer_features(
        self,
        feature_cols: Optional[List[str]] = None,
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """Combines failure components, imputes missing failure modes, and flags failures."""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call `load_data()` first.")

        df = self.df.copy()

        # Combine 'comp' (from maintenance) and 'failure' into 'comp_failure'
        if "failure" in df.columns and "comp" in df.columns:
            df["comp_failure"] = df["failure"].combine_first(df["comp"])
            df.drop(columns=["failure", "comp"], inplace=True)
        elif "failure" in df.columns:
            df["comp_failure"] = df["failure"]
            df.drop(columns=["failure"], inplace=True)

        if "comp_failure" not in df.columns:
            self.df = df
            return self.df

        if feature_cols is None:
            feature_cols = [
                "volt", "rotate", "pressure", "vibration",
                "temperature_c", "vibration_level_mm_per_s",
                "load_percentage", "operating_hours", "model", "age",
            ]

        available_features = [col for col in feature_cols if col in df.columns]
        train_mask = df["comp_failure"].notna()
        error_col = "errorid" if "errorid" in df.columns else ("errorID" if "errorID" in df.columns else None)

        target_mask = (
            df["comp_failure"].isna() & df[error_col].notna()
            if error_col
            else pd.Series(False, index=df.index)
        )

        # KNN classification for component failure types
        if target_mask.sum() > 0 and train_mask.sum() > 0 and available_features:
            logging.info(f"Imputing {target_mask.sum()} missing component failures via KNN...")

            X_encoded = pd.get_dummies(
                df[available_features],
                columns=[c for c in ["model"] if c in available_features],
                drop_first=True,
            )

            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_encoded),
                index=df.index,
                columns=X_encoded.columns,
            )

            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_scaled[train_mask], df.loc[train_mask, "comp_failure"].astype(str))
            df.loc[target_mask, "comp_failure"] = knn.predict(X_scaled[target_mask])

        df["failed"] = df["comp_failure"].notna().astype(int)
        self.df = df
        return self.df

    def format_numeric_precision(self) -> pd.DataFrame:
        """Rounds continuous floating-point fields to clean decimal precision."""
        if self.df is None:
            return self.df

        float_cols = [
            "temperature_c",
            "vibration_level_mm_per_s",
            "load_percentage",
            "maintenance_due_in_hours",
            "volt",
            "rotate",
            "pressure",
            "vibration",
        ]
        active_float_cols = [col for col in float_cols if col in self.df.columns]
        if active_float_cols:
            self.df[active_float_cols] = self.df[active_float_cols].round(2)

        return self.df

    def save(self, output_filename: str) -> Path:
        """Saves the processed DataFrame to the clean_dataset output directory."""
        if self.df is None:
            raise ValueError("No DataFrame available to save.")

        out_file = self.output_dir / output_filename
        out_file.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out_file, index=False)
        logging.info(f"Saved processed dataset to: {out_file.resolve()}")
        return out_file

    def run_cleaning_pipeline(
        self,
        input_filename: str,
        output_filename: str,
        run_knn_imputation: bool = False,
    ) -> pd.DataFrame:
        """Sequential pipeline execution: load -> sanitize -> impute -> engineer -> round -> save."""
        self.load_data(filename=input_filename)
        self.standardize_and_sanitize_maintenance_data()
        self.impute_continuous_sensor_data()

        if run_knn_imputation:
            self.impute_and_engineer_features()

        self.format_numeric_precision()
        self.save(output_filename=output_filename)
        return self.df


if __name__ == "__main__":
    processor = PredictiveMaintenancePreprocessor(
        data_dir="datasets/ingested_dataset",
        output_dir="datasets/clean_dataset",
    )

    # 1. Clean the synthetic / maintenance dataset (handles noise, dropouts, and outliers)
    df_maintenance = processor.run_cleaning_pipeline(
        input_filename="machine_maintenance_dataset_ingested.csv",
        output_filename="cleaned_maintenance_dataset.csv",
        run_knn_imputation=False,
    )

    # 2. Clean and run failure mode imputation on telemetry failure logs
    df_pdm = processor.run_cleaning_pipeline(
        input_filename="PdM_combined.csv",
        output_filename="PdM_combined_cleaned.csv",
        run_knn_imputation=True,
    )