"""Validation-only peak extraction and one-to-one Hungarian point matching."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
def peaks(logits, threshold, width, height):
    p=torch.sigmoid(logits)[0,0]; keep=(p==F.max_pool2d(p[None,None],3,1,1)[0,0]) & (p>=threshold); ys,xs=keep.nonzero(as_tuple=True); return [(float(x*width/p.shape[1]),float(y*height/p.shape[0]),float(p[y,x])) for y,x in zip(ys,xs)]
def match(pred, gt, tolerance):
    if not pred and not gt: return [], [], []
    if not pred: return [], [], list(range(len(gt)))
    if not gt: return [], list(range(len(pred))), []
    pred_xy=np.asarray([point[:2] for point in pred],dtype=np.float64)
    gt_xy=np.asarray([point[:2] for point in gt],dtype=np.float64)
    distances=np.linalg.norm(pred_xy[:,None,:]-gt_xy[None,:,:],axis=2)
    row_ind,col_ind=linear_sum_assignment(distances)
    pairs=[(int(i),int(j),float(distances[i,j])) for i,j in zip(row_ind,col_ind) if distances[i,j]<=tolerance]
    used_predictions={item[0] for item in pairs}; used_gt={item[1] for item in pairs}
    return pairs,[i for i in range(len(pred)) if i not in used_predictions],[j for j in range(len(gt)) if j not in used_gt]
