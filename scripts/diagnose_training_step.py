"""Fixed train-batch learning diagnostic; never loads validation or test data."""
from pathlib import Path
import argparse,sys,torch,pandas as pd
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from src.node_detection_dataset import NodeDetectionDataset,image_lookup,node_detection_collate_fn; from src.models.resnet_node_heatmap import ResNet18NodeHeatmap; from src.models.resnet_node_heatmap_fpn import ResNet18NodeHeatmapFPN; from src.node_heatmaps import modified_focal_loss; from src.train_node_detector import device
p=argparse.ArgumentParser(); p.add_argument('--model-variant',choices=['baseline','fpn'],default='baseline'); p.add_argument('--overfit-steps',action='store_true'); p.add_argument('--compare-initialization',action='store_true'); a=p.parse_args(); torch.manual_seed(42)
root=Path(__file__).resolve().parents[1]; im=pd.read_csv(root/'results/node_dataset_split/node_images_152_split.csv'); pt=pd.read_csv(root/'results/node_dataset_split/node_detection_points_152_split.csv'); ds=NodeDetectionDataset(im[im.split.eq('train')],pt[pt.split.eq('train')],image_lookup(root,im),training=False); batch=next(iter(DataLoader(ds,batch_size=4,shuffle=False,collate_fn=node_detection_collate_fn))); x=batch['image'].to(device()); targets=batch['heatmap'].to(device()); Model=ResNet18NodeHeatmapFPN if a.model_variant=='fpn' else ResNet18NodeHeatmap
def norm(parts): return torch.sqrt(sum((v.detach()**2).sum() for v in parts)).detach().item() if parts else 0.
def run(prior=.10):
 torch.manual_seed(42); m=Model(prior).to(device()); final=m.head[-1]; enc=list(m.stem.parameters())+list(m.l1.parameters())+list(m.l2.parameters())+list(m.l3.parameters())+list(m.l4.parameters()); ids={id(v) for v in enc}; dec=[v for v in m.parameters() if id(v) not in ids]; assert enc and dec; opt=torch.optim.AdamW([{'params':enc,'lr':1e-4},{'params':dec,'lr':1e-3}],weight_decay=1e-4); old=[v.detach().clone() for v in m.parameters()]; values=[]; finite=True
 for step in range(21):
  m.train(); opt.zero_grad(set_to_none=True); logits=m(x); loss,parts=modified_focal_loss(logits,targets,return_components=True); prob=torch.sigmoid(logits); center=prob[targets.eq(1)].mean(); bg=prob[targets.lt(1)].mean(); finite &= bool(torch.isfinite(loss));
  if step in {0,1,2,5,10,20}: print('step',step,{'loss':loss.detach().item(),**parts,'positive_centers':int(targets.eq(1).sum()),'gt_center_probability':center.detach().item(),'background_probability':bg.detach().item()})
  values.append((loss.detach().item(),center.detach().item(),bg.detach().item()))
  if step==20: break
  loss.backward(); eg=[v.grad for v in enc if v.grad is not None]; dg=[v.grad for v in dec if v.grad is not None]; fg=[final.weight.grad,final.bias.grad]; status={'encoder':norm(eg),'fpn_decoder':norm(dg),'final_head':norm(fg),'nan':any(torch.isnan(g).any() for g in eg+dg),'inf':any(torch.isinf(g).any() for g in eg+dg)}; print('gradients',step,status); finite &= not status['nan'] and not status['inf']; opt.step(); now=[v.detach().clone() for v in m.parameters()]; print('parameter_delta',step,{'encoder':norm([u-v for u,v in zip(now[:len(enc)],old[:len(enc)])]),'decoder':norm([u-v for u,v in zip(now[-len(dec):],old[-len(dec):])])}); old=now
 healthy=finite and values[-1][1]>values[-1][2] and values[-1][1]>1e-4 and values[-1][0]<values[0][0]
 label='V3 FPN FIXED-BATCH TEST' if a.model_variant=='fpn' else 'BASELINE FIXED-BATCH TEST'; print(label+': '+('PASS' if healthy else 'FAIL'),'' if healthy else 'reason=loss/center-probability/gradient health criterion not met')
run(.10)
