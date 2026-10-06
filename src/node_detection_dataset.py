"""Dataset V1 image lookup, resize, conservative augmentation, and node heatmaps."""
from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from .node_heatmaps import gaussian_heatmap

INPUT_W, INPUT_H, HM_W, HM_H = 768, 576, 192, 144
MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)

def node_detection_collate_fn(samples):
    """Stack fixed spatial tensors while retaining variable-length node data per image."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "heatmap": torch.stack([sample["heatmap"] for sample in samples]),
        "filename": [sample["filename"] for sample in samples],
        "original_size": [sample["original_size"] for sample in samples],
        "resized_coords": [sample["resized_coords"] for sample in samples],
        "original_coords": [sample["original_coords"] for sample in samples],
        "visibility": [sample["visibility"] for sample in samples],
    }

def image_lookup(root, image_rows):
    root=Path(root); locations={}
    for folder in (root/'data/node_annotation_pilot/images', root/'data/node_annotation_scaleup/images'):
        for path in folder.glob('*'):
            if path.is_file():
                if path.name in locations: raise ValueError(f'duplicate image filename: {path.name}')
                locations[path.name]=path
    expected=set(image_rows.filename)
    if set(locations)!=expected: raise ValueError(f'image lookup mismatch: missing={len(expected-set(locations))}, extra={len(set(locations)-expected)}')
    return locations

class NodeDetectionDataset(Dataset):
    def __init__(self, images, points, lookup, training=False, seed=42):
        self.images=images.reset_index(drop=True); self.points=points; self.lookup=lookup; self.training=training; self.seed=seed
    def __len__(self): return len(self.images)
    def __getitem__(self, index):
        row=self.images.iloc[index]; pts=self.points.loc[self.points.filename.eq(row.filename)]
        with Image.open(self.lookup[row.filename]) as image: image=image.convert('RGB'); ow,oh=image.size
        coords=torch.tensor(pts[['x','y']].to_numpy(),dtype=torch.float32); coords[:,0]*=INPUT_W/ow; coords[:,1]*=INPUT_H/oh
        if self.training and random.random()<.5:
            image=TF.hflip(image); coords[:,0]=(INPUT_W-1)-coords[:,0]
        if self.training:
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1)); image=ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1)); image=ImageEnhance.Color(image).enhance(random.uniform(.9,1.1))
        image=image.resize((INPUT_W,INPUT_H),Image.Resampling.BILINEAR); tensor=TF.normalize(TF.to_tensor(image),MEAN,STD)
        hm=gaussian_heatmap([(x*HM_W/INPUT_W,y*HM_H/INPUT_H) for x,y in coords])
        return {'image':tensor,'heatmap':hm,'filename':row.filename,'original_size':torch.tensor([ow,oh]),'resized_coords':coords,'original_coords':torch.tensor(pts[['x','y']].to_numpy(),dtype=torch.float32),'visibility':list(pts.visibility)}
