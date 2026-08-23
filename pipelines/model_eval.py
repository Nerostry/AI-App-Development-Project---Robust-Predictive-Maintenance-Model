import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score

# Import model architecture and dataset from training script
from pipelines.model_train import MultimodalMaintenanceDataset, StreamingMultimodalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_DIR = "saved_models"
DEFAULT_MODEL_NAME = "predictive_maintenance_model.pth"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)


def evaluate_and_tune(model, dataset):
    model.eval()
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_targets, all_probs, all_ttfs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            ts = batch["ts"].to(device)
            num = batch["num"].to(device)
            img = batch["img"].to(device)
            
            logits, ttf_pred = model(ts, num, img)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch["label"].numpy())
            all_ttfs.extend(ttf_pred.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Calculate metrics
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    predictions = (all_probs >= optimal_threshold).astype(int)
    tuned_precision = precision_score(all_targets, predictions, zero_division=0)
    tuned_recall = recall_score(all_targets, predictions, zero_division=0)

    print("\n" + "=" * 45)
    print("      EVALUATION & THRESHOLD RESULTS       ")
    print("=" * 45)
    print(f"PR-AUC (Average Precision) : {pr_auc:.4f}")
    print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
    print(f"F1-Score at Threshold      : {best_f1:.4f}")
    print(f"Precision at Threshold     : {tuned_precision:.4f}")
    print(f"Recall at Threshold        : {tuned_recall:.4f}")
    print("=" * 45)

    return best_f1, optimal_threshold


def save_model_if_better(model, current_f1, min_f1_threshold=0.60, output_dir=DEFAULT_MODEL_DIR):
    """Saves the model if its F1 score exceeds the quality threshold."""
    if current_f1 >= min_f1_threshold:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, DEFAULT_MODEL_NAME)
        torch.save(model.state_dict(), model_path)
        print(f"\n[QUALITY CHECK PASSED] Model F1 ({current_f1:.4f}) >= Threshold ({min_f1_threshold:.4f}).")
        print(f"Model successfully saved to '{model_path}'.")
    else:
        print(f"\n[QUALITY CHECK FAILED] Model F1 ({current_f1:.4f}) < Threshold ({min_f1_threshold:.4f}).")
        print(f"Model was NOT saved to '{output_dir}/'.")


if __name__ == "__main__":
    # Resolve model checkpoint path (primary: saved_models/, fallback: root model.pt)
    if os.path.exists(DEFAULT_MODEL_PATH):
        model_path = DEFAULT_MODEL_PATH
    elif os.path.exists("model.pt"):
        model_path = "model.pt"
    else:
        model_path = DEFAULT_MODEL_PATH

    # 1. Load newly trained model checkpoint explicitly with weights_only=True
    model = StreamingMultimodalTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"Loaded weights successfully from '{model_path}'")

    # 2. Generate Validation Data
    val_dataset = MultimodalMaintenanceDataset(
        num_samples=300, sequence_length=64, failure_rate=0.08
    )

    # 3. Evaluate Model Quality
    best_f1, optimal_threshold = evaluate_and_tune(model, val_dataset)

    # 4. Save model if quality criteria is met
    QUALITY_BENCHMARK_F1 = 0.60
    save_model_if_better(model, best_f1, min_f1_threshold=QUALITY_BENCHMARK_F1)