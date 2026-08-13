import logging
from pipelines.data_ingestion import DataIngestion
from pipelines.data_clean import DataCleaning
from pipelines.feature_engineering import FeatureEngineering
from pipelines.model_training import ModelTrainer
from pipelines.model_evaluate import ModelEvaluator
from pipelines.model_save import ModelSaver

logging.basicConfig(level=logging.INFO)

def run_pipeline(data_path: str):
    # Step 1: Data Ingestion
    logging.info("Starting Data Ingestion...")
    ingestion = DataIngestion(data_path=data_path)
    df = ingestion.read_data()

    # Step 2: Data Cleaning
    logging.info("Starting Data Cleaning...")
    cleaner = DataCleaning()
    cleaned_df = cleaner.clean(df)

    # Step 3: Feature Engineering
    logging.info("Starting Feature Engineering...")
    feature_engineer = FeatureEngineering()
    X_train, X_test, y_train, y_test = feature_engineer.transform(cleaned_df)

    # Step 4: Model Training
    logging.info("Starting Model Training...")
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)

    # Step 5: Model Evaluation
    logging.info("Starting Model Evaluation...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)
    logging.info(f"Evaluation Metrics: {metrics}")

    # Step 6: Model Save
    logging.info("Saving Model...")
    saver = ModelSaver()
    saver.save(model, output_path="models/model.pkl")

if __name__ == "__main__":
    run_pipeline(data_path="data/raw_data.csv")