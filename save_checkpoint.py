# Save model checkpoint so we don't lose training if Colab disconnects.
# Also downloads it to your machine as a backup.
import torch
from google.colab import files

CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"

torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": EPOCHS,
    "embedding_dim": 128,
}, CHECKPOINT_PATH)

print(f"Saved checkpoint: {CHECKPOINT_PATH}")

# Download to your local machine so it's not lost if the Colab runtime resets
files.download(CHECKPOINT_PATH)
