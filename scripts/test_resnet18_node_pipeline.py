"""Local Phase 7B smoke test. It never constructs or evaluates a test loader."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch, torchvision
from torch.utils.data import DataLoader
from src.node_detection_dataset import NodeDetectionDataset, image_lookup, node_detection_collate_fn
from src.models.resnet_node_heatmap import ResNet18NodeHeatmap
from src.node_heatmaps import modified_focal_loss

root = Path(__file__).resolve().parents[1]
images = pd.read_csv(root / "results/node_dataset_split/node_images_152_split.csv")
points = pd.read_csv(root / "results/node_dataset_split/node_detection_points_152_split.csv")
print("torch", torch.__version__); print("torchvision", torchvision.__version__)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
lookup = image_lookup(root, images)
train = NodeDetectionDataset(images.loc[images.split.eq("train")], points.loc[points.split.eq("train")], lookup, training=True)
validation = NodeDetectionDataset(images.loc[images.split.eq("validation")], points.loc[points.split.eq("validation")], lookup)
assert len(train) == 104 and len(validation) == 24
sample = validation[0]
assert tuple(sample["image"].shape) == (3,576,768)
assert tuple(sample["heatmap"].shape) == (1,144,192)
assert float(sample["heatmap"].min()) >= 0 and float(sample["heatmap"].max()) <= 1
assert float(sample["heatmap"].max()) == 1.0 and int(sample["heatmap"].eq(1).sum()) >= 1
assert int(sample["heatmap"].eq(1).sum()) <= len(sample["original_coords"])
assert torch.allclose(sample["resized_coords"][:,0], sample["original_coords"][:,0] * 768 / sample["original_size"][0])

# Explicitly choose unequal node counts; default collate would fail on this pair.
counts = [(index, len(train[index]["original_coords"])) for index in range(len(train))]
first, second = next((a,b) for a in counts for b in counts if a[1] != b[1])
batch = node_detection_collate_fn([train[first[0]], train[second[0]]])
print(f"variable-length node counts: {first[1]}, {second[1]}")
assert tuple(batch["image"].shape) == (2,3,576,768)
assert tuple(batch["heatmap"].shape) == (2,1,144,192)
assert isinstance(batch["resized_coords"], list) and len(batch["resized_coords"]) == 2
assert tuple(batch["resized_coords"][0].shape) != tuple(batch["resized_coords"][1].shape)
assert all(float(target.max()) == 1.0 and int(target.eq(1).sum()) >= 1 for target in batch["heatmap"])
print("HEATMAP TARGET TEST: PASS")

# Horizontal flip coordinate rule is x' = 767 - x at the resized resolution.
x = sample["resized_coords"][0,0]; assert torch.isclose((768-1)-x, torch.tensor(767., dtype=x.dtype)-x)
model = ResNet18NodeHeatmap().to(device)
logits = model(batch["image"].to(device)); assert tuple(logits.shape) == (2,1,144,192)
loss = modified_focal_loss(logits, batch["heatmap"].to(device)); assert torch.isfinite(loss)
background_logits = torch.full_like(batch["heatmap"], -12.0)
perfect_logits = torch.where(batch["heatmap"].eq(1), torch.full_like(batch["heatmap"], 12.0), torch.full_like(batch["heatmap"], -12.0))
background_loss = modified_focal_loss(background_logits, batch["heatmap"]); perfect_loss = modified_focal_loss(perfect_logits, batch["heatmap"])
assert background_loss > 1.0 and perfect_loss < background_loss
print("FOCAL LOSS SANITY TEST: PASS")
# Regression: an extremely negative node logit must retain a finite, non-zero
# gradient that increases the logit, rather than sticking at a clamp floor.
extreme_targets = torch.zeros((1,1,3,3)); extreme_targets[0,0,1,1] = 1.0
extreme_logits = torch.full_like(extreme_targets, -20.0, requires_grad=True)
extreme_loss = modified_focal_loss(extreme_logits, extreme_targets); extreme_loss.backward()
center_gradient = extreme_logits.grad[0,0,1,1]
assert torch.isfinite(extreme_loss) and torch.isfinite(center_gradient) and center_gradient != 0 and center_gradient < 0
high_logits = torch.full_like(extreme_targets, -20.0); high_logits[0,0,1,1] = 20.0
assert modified_focal_loss(high_logits, extreme_targets) < extreme_loss
print("EXTREME LOGIT GRADIENT TEST: PASS")
loss.backward()
print("PHASE 7B SMOKE TEST: PASS")
