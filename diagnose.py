"""Query-level annotation diagnosis from four portable TSV tables (standard library).

python diagnose.py --example --output demo_output
python diagnose.py --input my_tables --output my_diagnosis
"""
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path

def read(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def flag(value,optional=False):
    if optional and value=='':return None
    if value not in ('0','1'):raise ValueError('Boolean fields require 0 or 1, or an explicitly allowed blank')
    return value=='1'

def diagnose(root):
    queries=read(root/'queries.tsv');candidates=read(root/'candidates.tsv');outputs=read(root/'outputs.tsv')
    library=defaultdict(set)
    for r in read(root/'library.tsv'):library[r['endpoint']].add(r['label'])
    key=lambda r:(r['query_id'],r['endpoint'])
    qmap={key(r):r for r in queries};omap={key(r):r for r in outputs}
    if len(qmap)!=len(queries) or len(omap)!=len(outputs) or set(qmap)!=set(omap):raise ValueError('Queries and outputs need the same unique query/endpoint keys')
    pools=defaultdict(list)
    seen=set()
    for r in candidates:
        k=key(r);identity=(*k,r['candidate_id'])
        if k not in qmap or identity in seen:raise ValueError('Unknown query or duplicated candidate identity')
        if r['label'] not in library[r['endpoint']]:raise ValueError('Candidate label absent from supplied reference library')
        seen.add(identity);r=dict(r);r['retained']=flag(r['retained'])
        r['score']=float(r['score']) if r['score'] else None
        if r['retained'] and (r['score'] is None or not math.isfinite(r['score'])):raise ValueError('Retained candidates require finite scores')
        pools[k].append(r)
    result=[]
    for k,q in qmap.items():
        values=json.loads(q['truth_labels'])
        if not isinstance(values,list) or not all(isinstance(x,str) for x in values):raise ValueError('truth_labels must be a JSON list of strings')
        truth=set(values);allrows=pools[k];retained=[r for r in allrows if r['retained']]
        o=omap[k];accepted=flag(o['accepted'],optional=True);cid=o['selected_candidate_id']
        chosen=next((r for r in retained if r['candidate_id']==cid),None) if cid else None
        if cid and chosen is None:raise ValueError('Selected identity is absent from retained candidates')
        if accepted and not chosen:raise ValueError('An accepted query requires a selected output')
        lib=bool(truth & library[q['endpoint']]);union=any(r['label'] in truth for r in allrows)
        scored=any(r['label'] in truth for r in retained)
        concordant=bool(chosen and chosen['label'] in truth)
        top=max((r['score'] for r in retained),default=None)
        top_labels={r['label'] for r in retained if r['score']==top}
        if not truth:state='NO_RECORDED_ACTIVITY'
        elif not lib:state='NO_LIBRARY_SUPPORT'
        elif not union:state='ABSENT_FROM_SAVED_UNION'
        elif not scored:state='RETENTION_LOSS'
        elif not chosen:state='SCORED_NO_SELECTION'
        elif concordant:state='CONCORDANT_TOP1'
        elif chosen['score']==top and bool(truth & top_labels):state='TOP_SCORE_TIE_LOSS'
        elif chosen['score']>max(r['score'] for r in retained if r['label'] in truth):state='STRICT_SCORE_LOSS'
        else:state='OTHER_SELECTION_LOSS'
        result.append({'query_id':k[0],'endpoint':k[1],'state':state,'evaluable':int(bool(truth)),
          'library_match':int(lib),'saved_union_match':int(union),'retained_match':int(scored),
          'selected_candidate_id':cid,'selected_label':chosen['label'] if chosen else '',
          'selected_concordant':int(concordant) if truth and chosen else '',
          'top_score_labels':len(top_labels),'accepted':'' if accepted is None else int(accepted)})
    states=[{'endpoint':e,'state':s,'queries':n} for (e,s),n in sorted(Counter((r['endpoint'],r['state']) for r in result).items())]
    summaries=[]
    for endpoint in sorted({r['endpoint'] for r in result}):
        g=[r for r in result if r['endpoint']==endpoint]
        a=[r for r in g if r['accepted']==1];ae=[r for r in a if r['evaluable']]
        summaries.append({'endpoint':endpoint,'queries':len(g),'evaluable_queries':sum(r['evaluable'] for r in g),
          'selected_queries':sum(bool(r['selected_candidate_id']) for r in g),
          'selected_concordant':sum(r['selected_concordant']==1 for r in g),
          'acceptance_known_queries':sum(r['accepted']!='' for r in g),'accepted_queries':len(a),
          'evaluable_accepted':len(ae),'accepted_concordant':sum(r['selected_concordant']==1 for r in ae),
          'accepted_disagreements':sum(r['selected_concordant']==0 for r in ae)})
    return {'per_query':result,'state_counts':states,'output_summary':summaries}

def write_tables(tables,out):
    out.mkdir(exist_ok=False)
    for name,rows in tables.items():
        if not rows:raise ValueError('Empty output table')
        with (out/(name+'.tsv')).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--example',action='store_true');g.add_argument('--input',type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    tables=diagnose(Path(__file__).parent/'example' if args.example else args.input)
    write_tables(tables,args.output);print(json.dumps(tables['output_summary'],indent=2))
