"""Deterministic regression tests for validation-only one-to-one point matching."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.node_detection_metrics import match
def check(pred,gt,tol,tp,fp,fn):
 pairs,up,ug=match(pred,gt,tol); assert len(pairs)==tp and len(up)==fp and len(ug)==fn
# Unequal, equal, and all empty combinations.
check([(0,0,.9),(10,0,.8)],[(0,0),(10,0),(20,0),(30,0)],1,2,0,2)
check([(0,0,.9),(10,0,.8),(20,0,.7),(30,0,.6)],[(0,0),(10,0)],1,2,2,0)
check([(0,0,.9),(10,0,.8),(20,0,.7)],[(0,0),(10,0),(20,0)],1,3,0,0)
check([],[(0,0),(1,1),(2,2),(3,3)],1,0,0,4)
check([(0,0,.9),(1,1,.8),(2,2,.7),(3,3,.6)],[],1,0,4,0)
check([],[],1,0,0,0)
# Hungarian may assign a far pair, but it must be rejected by tolerance.
check([(0,0,.9),(100,100,.8)],[(0,0),(200,200)],5,1,1,1)
print('HUNGARIAN MATCHING TEST: PASS')
