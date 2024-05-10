import sys, numpy as np, pandas as pd, os
from plotly import graph_objects as go, express as px
sys.path.append('/Users/kingsbury/work/NSD/')
from .scsc.scsd import *
from .scsc.scsd_models_user import *


def find_multiplier_series(seq, ptgr = 'Oh'):
    pgt = scsd_symmetry('Oh').pgt
    #len 1
    for k,v in pgt.items():
        if str(v) == seq:
            return mondrian_lookup_dict.get('Oh').get(seq) + ' : ' + k
    for k1,v1 in pgt.items():
        for k2,v2 in pgt.items():
            if str([a*b for a,b in zip(v1,v2)]) == seq:
                return mondrian_lookup_dict.get('Oh').get(seq) + ' : ' +  ','.join([k1,k2])
    for k1,v1 in pgt.items():
        for k2,v2 in pgt.items():
            for k3,v3 in pgt.items():
                if str([a*b*c for a,b,c in zip(v1,v2,v3)]) == seq:
                    return ','.join([k1,k2,k3])
    return("not found "+mondrian_lookup_dict.get('Oh').get(seq))


def elim_series(ptgr = 'Oh'):
    mold = mondrian_lookup_dict.get(ptgr).copy()
    mold2 = mondrian_lookup_dict.get(ptgr).copy()
    for k1,v1 in mold.items():
        for k2,v2 in mold.items():
            for k3,v3 in mold.items():
                if str([int(a)*int(b) for a,b in zip(k2[1:-1:3],k3[1:-1:3])]) == k1 and k1 not in [k2,k3]:
                    try: mold2.pop(k1)
                    except KeyError: pass
    return mold2

def print_formal_subgroups(ptgr = 'Oh'):
    pgt = scsd_symmetry(ptgr).pgt
    pd = {k:[kt for kt, vt in pgt.items() if all(np.multiply(vt,v) == v) and (k is not kt) and (kt not in ['A1','Ag','A1g'])] for k,v in pgt.items()}
    pd = {k:v for k,v in pd.items() if len(v)>0}
    print(pd)