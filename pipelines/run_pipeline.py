# ============================================================
# FIX 1: pipelines/model_save.py
# Problem: `model` was never defined/imported — NameError on run.
# Fix: turn this into a reusable function that takes the model in.
# ============================================================
import os
import torch


def save_model(model, output_dir="saved_models", filename="predictive_maintenance_model.pth"):
    """Save a PyTorch model's state_dict to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model state saved successfully to {model_path}")
    return model_path


class ModelSaver:
    """Wrapper class so run_pipeline.py can call saver.save(model, output_path=...)."""

    def save(self, model, output_path="saved_models/predictive_maintenance_model.pth"):
        output_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        return save_model(model, output_dir=output_dir, filename=filename)


if __name__ == "__main__":
    # Example standalone usage — replace with your actual trained model instance
    from pipelines.model_train import StreamingMultimodalTransformer

    model = StreamingMultimodalTransformer()
    save_model(model)


# ============================================================
# FIX 2: pipelines/run_pipeline.py
# Problem: imports non-existent modules/classes (model_training,
# model_evaluate, DataIngestion, DataCleaning, FeatureEngineering,
# ModelTrainer, ModelEvaluator, ModelSaver) that don't match the
# real files (data_ingest.py, data_clean.py, model_train.py,
# model_eval.py, model_save.py) or their real class/function names.
# Fix: point imports at the files/functions that actually exist,
# and adjust the flow to match their real signatures.
# ============================================================
import logging

from pipelines.data_ingest import DataIngestor, DatasetCombiner
from pipelines.data_clean import PredictiveMaintenancePreprocessor
from pipelines.model_train import (
    MultimodalMaintenanceDataset,
    StreamingMultimodalTransformer,
    train_model,
)
from pipelines.model_eval import evaluate_and_tune, save_model_if_better
from pipelines.model_save import ModelSaver

logging.basicConfig(level=logging.INFO)


def run_pipeline(
    raw_datasets_dir: str = "datasets/raw_datasets",
    ingested_dir: str = "datasets/ingested_dataset",
    clean_dir: str = "datasets/clean_dataset",
    model_checkpoint_path: str = "model.pt",
):
    # Step 1: Data Ingestion
    logging.info("Starting Data Ingestion...")
    ingestor = DataIngestor(datasets_dir=raw_datasets_dir, output_dir=ingested_dir)
    ingested_data = ingestor.run()

    combiner = DatasetCombiner(data_sources=ingested_data)
    combined_df = combiner.combine()

    if combined_df is None:
        logging.warning("No combined dataframe produced (missing telemetry/machines tables). Stopping.")
        return

    combined_csv_path = f"{ingested_dir}/PdM_combined.csv"
    combined_df.to_csv(combined_csv_path, index=False)

    # Step 2: Data Cleaning
    logging.info("Starting Data Cleaning...")
    cleaner = PredictiveMaintenancePreprocessor(data_dir=ingested_dir, output_dir=clean_dir)
    cleaned_df = cleaner.run_cleaning_pipeline(
        input_filename="PdM_combined.csv",
        output_filename="PdM_combined_cleaned.csv",
        run_knn_imputation=True,
    )

    # Step 3: Model Training
    logging.info("Starting Model Training...")
    train_model(save_path=model_checkpoint_path)

    # Step 4: Model Evaluation
    logging.info("Starting Model Evaluation...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreamingMultimodalTransformer().to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))

    val_dataset = MultimodalMaintenanceDataset(num_samples=300, sequence_length=64, failure_rate=0.08)
    best_f1, optimal_threshold = evaluate_and_tune(model, val_dataset)
    logging.info(f"Evaluation F1: {best_f1:.4f}, Optimal Threshold: {optimal_threshold:.4f}")

    # Step 5: Conditional Model Save
    logging.info("Saving Model (if quality threshold met)...")
    save_model_if_better(model, best_f1, min_f1_threshold=0.60)


if __name__ == "__main__":
    run_pipeline()