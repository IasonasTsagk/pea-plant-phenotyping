"""Local diagnostic for exact CenterNet-style heatmap-positive pixels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.node_detection_dataset import NodeDetectionDataset,image_lookup
root=Path(__file__).resolve().parents[1]; images=pd.read_csv(root/'results/node_dataset_split/node_images_152_split.csv'); points=pd.read_csv(root/'results/node_dataset_split/node_detection_points_152_split.csv'); lookup=image_lookup(root,images)
rows=[]
for split,expected in [('train',658),('validation',152)]:
 ds=NodeDetectionDataset(images[images.split.eq(split)],points[points.split.eq(split)],lookup)
 for index in range(len(ds)):
  s=ds[index]; h=s['heatmap']; rows.append({'split':split,'filename':s['filename'],'gt_node_count':len(s['original_coords']),'heatmap_max':float(h.max()),'exact_one_pixels':int(h.eq(1).sum()),'above_099_pixels':int(h.gt(.99).sum()),'nonzero_pixels':int(h.gt(0).sum())})
frame=pd.DataFrame(rows); out=root/'results/node_detection_resnet18_v2'; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/'heatmap_diagnostic.csv',index=False)
for split,expected in [('train',658),('validation',152)]:
 f=frame[frame.split.eq(split)]; print(split,{'images':len(f),'points':int(f.gt_node_count.sum()),'expected_points':expected,'min_max':float(f.heatmap_max.min()),'max_max':float(f.heatmap_max.max()),'mean_max':float(f.heatmap_max.mean()),'no_exact_one':int(f.exact_one_pixels.eq(0).sum()),'total_exact_positive':int(f.exact_one_pixels.sum())})
