"""Training utilities; validation only, never test evaluation."""
import time, torch
from .node_heatmaps import modified_focal_loss
def device(): return torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
def run_epoch(model,loader,optimizer,dev):
    train=optimizer is not None; model.train(train); total=0.; batches=0
    for batch in loader:
        images=batch['image'].to(dev); targets=batch['heatmap'].to(dev)
        if targets.ndim != 4 or targets.requires_grad or float(targets.min()) < 0 or float(targets.max()) > 1 or float(targets.max()) != 1.0 or int(targets.eq(1).sum()) == 0:
            raise RuntimeError("Heatmap batch has zero exact positive pixels; target generation is invalid.")
        with torch.set_grad_enabled(train):
            logits=model(images)
            if logits.ndim != 4 or logits.shape != targets.shape: raise RuntimeError(f"Invalid logits/target shape: {logits.shape} vs {targets.shape}")
            loss=modified_focal_loss(logits,targets)
        if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
        total+=loss.item(); batches+=1
    return total/batches
