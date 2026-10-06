"""Local V3 FPN smoke test; no test loader or evaluation."""
from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from src.models.resnet_node_heatmap_fpn import ResNet18NodeHeatmapFPN; from src.node_heatmaps import modified_focal_loss
m=ResNet18NodeHeatmapFPN(); x=torch.randn(2,3,576,768); c=m.features(x); assert [tuple(v.shape[-2:]) for v in c]==[(144,192),(72,96),(36,48),(18,24)]; logits=m(x); assert tuple(logits.shape)==(2,1,144,192); target=torch.zeros_like(logits); target[:,:,30,40]=1; loss=modified_focal_loss(logits,target); assert torch.isfinite(loss); loss.backward(); assert m.l1[0].conv1.weight.grad is not None and m.lat1.weight.grad is not None and m.head[-1].weight.grad is not None; assert not any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in m.parameters() if p.grad is not None); print('EXTREME LOGIT GRADIENT TEST: PASS'); print('V3 FPN MODEL TEST: PASS')
