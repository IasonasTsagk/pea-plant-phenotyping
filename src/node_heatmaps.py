"""Heatmap targets and modified focal loss for single-class node detection."""
import torch
import torch.nn.functional as F

def gaussian_heatmap(points, height=144, width=192, sigma=2.0):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    heatmap = torch.zeros((height, width), dtype=torch.float32)
    for x, y in points:
        # Continuous coordinates remain available for evaluation; heatmap
        # supervision uses their nearest valid pixel so each node has target==1.
        cx = max(0, min(width - 1, int(torch.round(torch.as_tensor(x)).item())))
        cy = max(0, min(height - 1, int(torch.round(torch.as_tensor(y)).item())))
        gaussian = torch.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma*sigma))
        heatmap = torch.maximum(heatmap, gaussian)
        heatmap[cy, cx] = 1.0
    return heatmap.unsqueeze(0)

def modified_focal_loss(logits, targets, alpha=2.0, beta=4.0, return_components=False):
    # Keep probabilities only for focal weights. Stable logs come directly from
    # logits, avoiding hard-clamp regions with zero recovery gradient.
    probabilities = torch.sigmoid(logits)
    log_p = F.logsigmoid(logits)
    log_one_minus_p = F.logsigmoid(-logits)
    positive = targets.eq(1).float(); negative = targets.lt(1).float()
    negative_weights = (1-targets).pow(beta)
    positive_loss = -(log_p * (1-probabilities).pow(alpha) * positive).sum()
    negative_loss = -(log_one_minus_p * probabilities.pow(alpha) * negative_weights * negative).sum()
    count = positive.sum()
    loss = negative_loss if count == 0 else (positive_loss + negative_loss) / count
    if return_components:
        divisor = count.clamp_min(1)
        return loss, {"num_positive": int(count.item()), "positive_loss": float((positive_loss/divisor).detach()), "negative_loss": float((negative_loss/divisor).detach())}
    return loss
