"""Recompute reported metrics from native per-query outputs and saved draws."""
from pathlib import Path
import argparse,csv,collections
import numpy as np
def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    rows=read(a.results/'native_query_diagnostics.tsv');draws=np.load(a.results/'component_draws.npy',allow_pickle=False)
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['method'],r['level'])].append(r)
    comps=sorted({int(r['component_id']) for r in rows});assert draws.shape==(2000,len(comps))
    result=[]
    for (method,level),sub in sorted(groups.items()):
        aggregate={c:np.zeros(6,dtype=int) for c in comps}
        for r in sub:aggregate[int(r['component_id'])]+=np.array([1,int(r['evaluable']),int(r['hit']),int(r['has_output']),int(r['output_count']),int(r['matched_output_count'])])
        values=np.array([aggregate[c] for c in comps])
        for metric,i,j in [('any_match',2,1),('output_coverage',3,0),('label_concordance',5,4)]:
            num=values[:,i];den=values[:,j];bootden=den[draws].sum(1);assert np.all(bootden>0)
            estimates=100*num[draws].sum(1)/bootden;lo,hi=np.quantile(estimates,[.025,.975])
            result.append(dict(method=method,level=level,metric=metric,numerator=int(num.sum()),denominator=int(den.sum()),percent=100*num.sum()/den.sum(),lower95=float(lo),upper95=float(hi),valid_replicates=len(estimates),equal_component_percent=float(np.mean(100*num[den>0]/den[den>0])),components=len(comps),protein_records=len(sub)))
    expected={(r['method'],r['level'],r['metric']):r for r in read(a.results/'metrics.tsv')}
    for row in result:
        saved=expected[(row['method'],row['level'],row['metric'])]
        for k,v in row.items():
            if k not in ['method','level','metric']:assert np.isclose(v,float(saved[k]),rtol=0,atol=1e-10),(k,v,saved[k])
    with a.output.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),delimiter='\t');w.writeheader();w.writerows(result)
    print(f'Reproduced all {len(result)} metric rows, including saved component intervals.')
if __name__=='__main__':main()
