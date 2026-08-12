import os
import torch

# Create the target directory if it doesn't exist
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

# Define saved model path
model_path = os.path.join(output_dir, "predictive_maintenance_model.pth")

# Save model state_dict (recommended)
torch.save(model.state_dict(), model_path)
print(f"PyTorch model state saved successfully to {model_path}")