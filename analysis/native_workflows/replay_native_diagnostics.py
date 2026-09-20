"""Replay two native workflows without SiteGuard or model weights.

python replay_native_diagnostics.py --input SOURCE_DIR --output NEW_DIRECTORY
Requires NumPy; all other imports are Python standard library.
"""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
from diagnostic_core import trace
def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.input;out=args.output;assert not out.exists()
    cohort=read(root/'cohort.tsv');assert len({r['protein_id'] for r in cohort})==len(cohort)
    truth={(r['protein_id'],r['level']):set(json.loads(r['truth_labels'])) for r in read(root/'query_truth.tsv')}
    levels=['EC_L3','EC_L4'];assert set(truth)=={(r['protein_id'],lev) for r in cohort for lev in levels}
    axes=json.loads((root/'clean_distance_axes.json').read_text());dist=np.load(root/'clean_distances.npy',allow_pickle=False)
    ec=axes['ec4_columns'];nodes={str(n):i for i,n in enumerate(axes['node_ids'])}
    assert dist.shape==(len(nodes),len(ec)) and np.isfinite(dist).all()
    order=np.argsort(dist,axis=1,kind='stable')
    ref={lev:collections.defaultdict(set) for lev in levels}
    for r in read(root/'reference_labels.tsv'):
        for lev,col in zip(levels,['ec_l3','ec_l4']):
            if r[col]:ref[lev][r['protein_id']].add(r[col])
    libs={lev:set().union(*ref[lev].values()) for lev in levels}
    hits=collections.defaultdict(list)
    with (root/'diamond_hits.tsv').open() as f:
        for r in csv.reader(f,delimiter='\t'):hits[r[0]].append((r[1],float(r[9]),float(r[8])))
    for node in hits:hits[node].sort(key=lambda x:(-x[1],x[2],x[0]))
    rows=[]
    for r in cohort:
        node=r['node_id'];i=nodes[node];indices=order[i];values=dist[i,indices[:10]]
        centre=np.mean(np.concatenate((values[1:],np.repeat(values[-1],10))))
        delta=np.abs(np.diff(np.abs(values-centre)));choices=np.flatnonzero(delta>delta.mean());assert len(choices)
        n=int(choices[0])+1 if choices[0]<5 else 1
        for lev in levels:
            cv=lambda x:'.'.join(x.split('.')[:3]) if lev=='EC_L3' else x
            universe={cv(e) for e in ec}
            candidates=[(ec[j],{cv(ec[j])},-float(dist[i,j])) for j in indices]
            for method,count in [('CLEAN_TOP1',1),('CLEAN_MAXSEP',n)]:
                output={cv(ec[j]) for j in indices[:count]}
                rows.append(dict(protein_id=r['protein_id'],node_id=node,component_id=r['component_id'],method=method,level=lev,
                    output_labels=json.dumps(sorted(output)),**trace(truth[(r['protein_id'],lev)],universe,candidates,output,10)))
            candidates=[(subject,ref[lev][subject],score) for subject,score,evalue in hits.get(node,[])]
            output=candidates[0][1] if candidates else set()
            rows.append(dict(protein_id=r['protein_id'],node_id=node,component_id=r['component_id'],method='DIAMOND_TOP_HIT_SET',level=lev,
                output_labels=json.dumps(sorted(output)),**trace(truth[(r['protein_id'],lev)],libs[lev],candidates,output,len(candidates))))
    out.mkdir();write(out/'per_query.tsv',rows)
    counts=collections.Counter((r['method'],r['level'],r['state']) for r in rows)
    write(out/'state_counts.tsv',[dict(method=m,level=l,state=s,records=n) for (m,l,s),n in sorted(counts.items())])
    print(json.dumps({'query_endpoint_outputs':len(rows),'protein_records':len(cohort),'status':'REPLAY_COMPLETE'}))
if __name__=='__main__':main()
