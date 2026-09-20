#!/usr/bin/env python3
"""Original neural-network inference under the frozen HPC PyTorch module."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--contract',type=Path,required=True);args=a.parse_args()
r=args.root.resolve();c=json.loads(args.contract.read_text());out=r/c['output_dir']
assert torch.__version__.startswith('2.1.2') and torch.cuda.is_available()
receipt=json.loads((out/'FEATURES_AND_TREE_PASS.json').read_text())
assert sha(out/'full_deep_X.npy')==receipt['feature_matrix_sha256']
source=r/'scripts/neural_train_deep.py'
expected=next(x['sha256'] for x in c['inputs'] if x['path']=='scripts/neural_train_deep.py');assert sha(source)==expected
weights=r/'models/phase11/siteguard_global_multitask.pt';expected=next(x['sha256'] for x in c['inputs'] if x['path']=='models/phase11/siteguard_global_multitask.pt');assert sha(weights)==expected
spec=importlib.util.spec_from_file_location('frozen_phase11',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=torch.device('cuda');torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
checkpoint=torch.load(weights,map_location=device)
model=m.SiteGuardGlobalNet(checkpoint['input_features'],checkpoint['width'],checkpoint['dropout']).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
x=np.load(out/'full_deep_X.npy',mmap_mode='r');pred=np.empty((len(x),3),np.float32)
with torch.no_grad():
    for start in range(0,len(x),65536):
        batch=torch.from_numpy(np.array(x[start:start+65536],copy=True)).to(device)
        pred[start:start+len(batch)]=torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
        if start%655360==0:print('DEEP_INFERENCE',start,len(x),flush=True)
ix=np.load(out/'deep_replay_rows.npy');expected=np.load(out/'deep_replay_expected.npy')
error=float(np.max(np.abs(pred[ix]-expected)))
if error>5e-6:
    with (out/'FAILED_deep_replay.json').open('x') as f:json.dump({'max_abs_error':error,'tolerance':5e-6},f)
    raise RuntimeError('Frozen neural replay discrepancy')
assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
with (out/'full_deep_predictions.npy').open('xb') as f:np.save(f,pred)
summary={'status':'DEEP_INFERENCE_PASS','rows':len(x),'replay_rows':len(ix),'replay_max_abs':error,'torch':torch.__version__,'python':sys.version,'model_sha256':sha(weights),'prediction_sha256':sha(out/'full_deep_predictions.npy'),'query_labels_read':False}
with (out/'DEEP_INFERENCE_PASS.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
