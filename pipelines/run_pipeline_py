# run_pipeline.py
import model_training
import model_evaluate

if __name__ == "__main__":
    print("--- Step 1: Training Model ---")
    model_training.train_model(save_path="model.pt")
    
    print("\n--- Step 2: Evaluating & Saving Best Model ---")
    # Call evaluation function from model_evaluate.py
    model_evaluate.run_evaluation_pipeline()