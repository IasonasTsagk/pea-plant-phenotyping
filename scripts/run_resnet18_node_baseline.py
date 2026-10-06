"""Manual Phase 7B training entrypoint. Validation only; test split is never loaded."""
from pathlib import Path
import argparse,csv,json,random,sys,time
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from src.node_detection_dataset import NodeDetectionDataset,image_lookup,node_detection_collate_fn; from src.models.resnet_node_heatmap import ResNet18NodeHeatmap; from src.models.resnet_node_heatmap_fpn import ResNet18NodeHeatmapFPN; from src.train_node_detector import device,run_epoch
parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',default='results/node_detection_resnet18_v1'); parser.add_argument('--model-variant',choices=['v2','fpn'],default='v2'); args=parser.parse_args(); Model=ResNet18NodeHeatmapFPN if args.model_variant=='fpn' else ResNet18NodeHeatmap
root=Path(__file__).resolve().parents[1]; out=root/args.output_dir; (out/'checkpoints').mkdir(parents=True,exist_ok=True); random.seed(42); np.random.seed(42); torch.manual_seed(42); dev=device()
images=pd.read_csv(root/'results/node_dataset_split/node_images_152_split.csv'); points=pd.read_csv(root/'results/node_dataset_split/node_detection_points_152_split.csv'); lookup=image_lookup(root,images); tr=NodeDetectionDataset(images[images.split=='train'],points[points.split=='train'],lookup,True); va=NodeDetectionDataset(images[images.split=='validation'],points[points.split=='validation'],lookup); batch=4
try: tl=DataLoader(tr,batch_size=batch,shuffle=True,collate_fn=node_detection_collate_fn); vl=DataLoader(va,batch_size=batch,collate_fn=node_detection_collate_fn); model=Model(heatmap_prior_probability=0.10).to(dev); next(iter(tl))
except RuntimeError as e:
 if 'memory' not in str(e).lower(): raise
 batch=2; tl=DataLoader(tr,batch_size=batch,shuffle=True,collate_fn=node_detection_collate_fn); vl=DataLoader(va,batch_size=batch,collate_fn=node_detection_collate_fn); model=Model(heatmap_prior_probability=0.10).to(dev)
backbone_params=list(model.stem.parameters())+list(model.l1.parameters())+list(model.l2.parameters())+list(model.l3.parameters())+list(model.l4.parameters()); backbone_ids={id(p) for p in backbone_params}; decoder_params=[p for p in model.parameters() if id(p) not in backbone_ids]; assert backbone_params and decoder_params
opt=torch.optim.AdamW([{'params':backbone_params,'lr':1e-4},{'params':decoder_params,'lr':1e-3}],weight_decay=1e-4); best=float('inf'); wait=0; rows=[]; start=time.time()
for epoch in range(1,81):
 train_loss=run_epoch(model,tl,opt,dev); val_loss=run_epoch(model,vl,None,dev); rows.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss,'backbone_lr':1e-4,'decoder_lr':1e-3}); torch.save({'epoch':epoch,'model_state':model.state_dict(),'validation_loss':val_loss},out/'checkpoints/last_model.pt')
 if val_loss<best: best=val_loss; wait=0; torch.save({'epoch':epoch,'model_state':model.state_dict(),'validation_loss':val_loss},out/'checkpoints/best_model.pt')
 else: wait+=1
 if wait>=12: break
pd.DataFrame(rows).to_csv(out/'training_history.csv',index=False); (out/'config.json').write_text(json.dumps({'dataset_version':'V1','seed':42,'input_resolution':[768,576],'heatmap_resolution':[192,144],'gaussian_sigma':2.0,'backbone':'ResNet-18','pretrained_weights':'ResNet18_Weights.DEFAULT','batch_size':batch,'optimizer':'AdamW','backbone_lr':1e-4,'decoder_lr':1e-3,'weight_decay':1e-4,'max_epochs':80,'early_stopping_patience':12,'augmentation':'horizontal flip p=0.5; mild brightness/contrast/saturation','loss':'CenterNet modified focal','nms':'3x3','threshold_sweep':[round(x/100,2) for x in range(10,91,5)],'model_selection':'validation loss','device':str(dev),'test_evaluated':False},indent=2)); print('Training complete. Test split was not loaded or evaluated.')
