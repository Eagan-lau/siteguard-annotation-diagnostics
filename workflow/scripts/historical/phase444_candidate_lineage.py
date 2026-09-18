#!/usr/bin/env python3
"""Frozen candidate-lineage calculation; query labels are evaluation-only."""
import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict, Counter
from pathlib import Path

LEVELS = ('EC_L3','EC_L4','EXACT_RHEA')
FIELDS = ('ec_l3','ec_l4','canonical_rhea')
TRUTH = ('ec_l3_json','ec_l4_json','canonical_rhea_json')
TARGETS = ('same_ec_l3','same_ec_l4','same_exact_rhea')
KS = (10,20,50,100)
STATES = ('NO_DOCUMENTED_TRUTH','NO_TRAIN_LIBRARY_SUPPORT',
          'LIBRARY_SUPPORTED_UNION50_MISS','UNION50_POSITIVE_SCORED_SUBSET_MISS',
          'SCORED_SUBSET_POSITIVE')

def classify(t,l,r,s):
    if s and not r: raise ValueError('Scored candidate not in retrieval universe')
    if r and not l: raise ValueError('Retrieved candidate not in library')
    if l and not t: raise ValueError('Support without documented truth')
    return STATES[0 if not t else 1 if not l else 2 if not r else 3 if not s else 4]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,obj):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def table(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--contract',type=Path)
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:
        assert [classify(*x) for x in [(0,0,0,0),(1,0,0,0),(1,1,0,0),(1,1,1,0),(1,1,1,1)]]==list(STATES)
        for x in [(1,0,1,0),(1,1,0,1),(0,1,0,0)]:
            try:classify(*x)
            except ValueError:pass
            else:raise AssertionError(x)
        assert min(21,51)>20 and min(21,51)<=50
        print('PHASE444_SELF_TEST_PASS');return
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    r=args.root.resolve();c=json.loads(args.contract.read_text());out=r/c['output_dir']
    assert platform.python_version()=='3.11.15'
    assert str(r)==c['resolved_root'] and c['k']==list(KS)
    checks=[]
    def check(name,ok,detail=''):
        checks.append({'check':name,'pass':bool(ok),'detail':str(detail)})
        if not ok:raise AssertionError(name+': '+str(detail))
    def identity(stage):
        for item in c['inputs']:
            p=r/item['path'];check(stage+': '+item['path'],p.stat().st_size==item['bytes'] and digest(p)==item['sha256'])
    identity('input_before')
    out.mkdir(exist_ok=False)
    dump(out/'reservation.json',{'status':'STARTED','contract_sha256':digest(args.contract)})
    print('PHASE444_INPUTS_PASS',flush=True)
    try:
        pred=pd.read_parquet(r/'results/phase12/end_to_end_predictions.parquet')
        check('primary_cohort_unique',pred.query_protein_id.is_unique and len(pred)==27639)
        ids=sorted(pred.query_protein_id);idx={q:i for i,q in enumerate(ids)};n=len(ids)
        truth=pd.read_parquet(r/'data/interim/phase04/retrieval_query_truth.parquet')
        check('truth_unique',truth.protein_id.is_unique)
        all_test=truth.loc[truth['split'].eq('test')]
        truth=truth.set_index('protein_id').loc[ids]
        check('primary_queries_test',truth['split'].eq('test').all())
        split=pd.read_parquet(r/'data/splits/split_sequence.parquet',columns=['protein_id','split','cluster_id_30'])
        check('split_unique',split.protein_id.is_unique)
        split=split.set_index('protein_id');sp=split['split'].to_dict();cl=split.cluster_id_30.to_dict()
        check('truth_split_agreement',all(sp[q]=='test' and cl[q]==truth.loc[q,'cluster_id_30'] for q in ids))
        train_clusters=set(split.loc[split['split'].eq('train'),'cluster_id_30'])
        check('train_test_clusters_disjoint',not train_clusters.intersection(truth.cluster_id_30))
        labels=[[set(map(str,json.loads(truth.loc[q,f]))) for f in TRUTH] for q in ids]
        lib=pd.read_parquet(r/'data/reference/activity_reference_library.parquet',columns=['activity_id','reference_protein_id',*FIELDS,'canonical_ec','reference_partition'])
        check('activity_keys_unique',lib.activity_id.is_unique)
        check('library_train_only',lib.reference_partition.eq('train').all() and all(sp.get(q)=='train' for q in lib.reference_protein_id.unique()))
        ref=defaultdict(lambda:[set(),set(),set()]);acts={}
        for row in lib.itertuples(index=False):
            vals=[getattr(row,f) for f in FIELDS];acts[row.activity_id]=(row.reference_protein_id,vals)
            for j,v in enumerate(vals):
                if pd.notna(v) and str(v):ref[row.reference_protein_id][j].add(str(v))
        glob=[set().union(*(x[j] for x in ref.values())) for j in range(3)]
        documented=np.array([[bool(s) for s in lab] for lab in labels])
        supported=np.array([[bool(lab[j]&glob[j]) for j in range(3)] for lab in labels])
        ec_mismatch=int((lib.ec_l4.fillna('')!=lib.canonical_ec.fillna('')).sum())
        mins=np.full((n,3,2),101,dtype=np.int16);pairs50=[set() for _ in ids]
        raw_counts={};cand_counts=np.zeros((n,2),dtype=np.int32)
        for m,mod in enumerate(('mmseqs','foldseek')):
            path=r/f'data/processed/{mod}_candidates.parquet';rows=0
            cols=['query_protein_id','reference_protein_id','modality_rank','query_partition','reference_partition','reference_cluster_id_30']
            last=defaultdict(set)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=cols):
                d=batch.to_pydict()
                for q,rp,k,qs,rs,rc in zip(*(d[x] for x in cols)):
                    if q not in idx:continue
                    i=idx[q];rows+=1
                    assert qs=='test' and rs=='train' and sp[rp]=='train' and rc==cl[rp] and cl[q]!=rc and q!=rp
                    assert 1<=k<=100 and rc not in last[q]
                    last[q].add(rc);cand_counts[i,m]+=1
                    if k<=50:pairs50[i].add(rp)
                    if rp in ref:
                        for j in range(3):
                            if labels[i][j]&ref[rp][j]:mins[i,j,m]=min(mins[i,j,m],k)
            raw_counts[mod]=rows
            check(mod+'_cluster_dedup_and_partition',True,rows)
            print('PHASE444_RETRIEVAL_DONE',mod,rows,flush=True)
        scored=np.zeros((n,3),dtype=bool);pair_counts=np.zeros(n,dtype=np.int32)
        cols=['pair_set','query_protein_id','reference_protein_id','reference_activity_id','query_split_expected',*TARGETS,'sample_weight']
        frame=pd.read_parquet(r/'results/phase11/pairwise_predictions.parquet',columns=cols)
        frame=frame.loc[frame.query_split_expected.eq('test')]
        check('phase11_test_query_cohort_exact',set(frame.query_protein_id)==set(ids))
        check('scored_activity_pairs_unique',not frame.duplicated(['query_protein_id','reference_protein_id','reference_activity_id']).any())
        check('scored_population_only',frame.pair_set.eq('population_atlas').all())
        for row in frame.itertuples(index=False):
            i=idx[row.query_protein_id];rp,vals=acts[row.reference_activity_id]
            assert rp==row.reference_protein_id and rp in pairs50[i]
            pair_counts[i]+=1
            for j,t in enumerate(TARGETS):
                flag=pd.notna(vals[j]) and str(vals[j]) in labels[i][j]
                assert bool(getattr(row,t))==flag
                scored[i,j]|=flag
        check('every_scored_pair_in_union50_and_truth_recomputed',True,len(frame))
        p=pred.set_index('query_protein_id').loc[ids]
        for j,lev in enumerate(LEVELS):
            check('phase12_oracle_reproduced_'+lev,np.array_equal(scored[:,j],p['oracle_candidate_available_'+lev].to_numpy(bool)))
        union=mins.min(axis=2)
        check('scored_subset_of_union50',np.all(~scored|(union<=50)))
        check('retrieved_subset_of_library',np.all((union>100)|supported))
        ledger=[];metrics=[];part=[]
        for j,lev in enumerate(LEVELS):
            states=[classify(documented[i,j],supported[i,j],union[i,j]<=50,scored[i,j]) for i in range(n)]
            for i,q in enumerate(ids):
                row={'query_protein_id':q,'cluster_id_30':cl[q],'level':lev,'documented_truth':bool(documented[i,j]),
                     'library_supported':bool(supported[i,j]),'mmseqs_first_positive_rank':int(mins[i,j,0]),
                     'foldseek_first_positive_rank':int(mins[i,j,1]),'scored_available':bool(scored[i,j]),
                     'scored_activity_pairs':int(pair_counts[i]),'union50_reference_candidates':len(pairs50[i]),'state':states[i]}
                for k in KS:
                    for mod,a in [('mmseqs',mins[:,j,0]),('foldseek',mins[:,j,1]),('union',union[:,j])]:row[f'{mod}_at_{k}']=bool(a[i]<=k)
                ledger.append(row)
            counts=Counter(states)
            for state in STATES:part.append({'level':lev,'state':state,'queries':counts[state],'denominator_all':n,'fraction_all':counts[state]/n})
            for kind,a in [('library',supported[:,j]),('scored_subset',scored[:,j])]+[(f'{mod}_at_{k}',arr<=k) for k in KS for mod,arr in [('mmseqs',mins[:,j,0]),('foldseek',mins[:,j,1]),('union',union[:,j])]]:
                for denom,mask in [('all_phase12_queries',np.ones(n,bool)),('documented_truth',documented[:,j]),('library_supported_truth',supported[:,j])]:
                    den=int(mask.sum());num=int((a&mask).sum())
                    metrics.append({'level':lev,'stage':kind,'denominator':denom,'positive_queries':num,'eligible_queries':den,'availability':num/den if den else None})
        ledger_path=out/'phase444_query_lineage.parquet'
        with ledger_path.open('xb') as f:pq.write_table(pa.Table.from_pylist(ledger),f,compression='zstd')
        table(out/'phase444_availability.tsv',metrics);table(out/'phase444_partition.tsv',part)
        cluster_ids,inv=np.unique([cl[q] for q in ids],return_inverse=True);ng=len(cluster_ids)
        cluster_n=np.bincount(inv);rng=np.random.default_rng(c['bootstrap_seed']);b=c['bootstrap_replicates']
        deltas=np.stack([union[:,j].__le__(50).astype(int)-scored[:,j].astype(int) for j in range(3)]+[(union[:,j]<=100).astype(int)-(union[:,j]<=50).astype(int) for j in range(3)],axis=1)
        sums=np.stack([np.bincount(inv,weights=deltas[:,j],minlength=ng) for j in range(6)],axis=1)
        boot=np.zeros((b,6))
        for start in range(0,b,25):
            ix=rng.integers(0,ng,size=(min(25,b-start),ng));boot[start:start+len(ix)]=sums[ix].sum(axis=1)/cluster_n[ix].sum(axis=1)[:,None]
        cis=[]
        for z in range(6):
            lo,hi=np.quantile(boot[:,z],[.025,.975]);cis.append({'level':LEVELS[z%3],'contrast':'union50_minus_scored' if z<3 else 'union100_minus_union50','additional_positive_queries':int(deltas[:,z].sum()),'all_queries':n,'difference_fraction':float(deltas[:,z].mean()),'ci95_low':float(lo),'ci95_high':float(hi),'clusters':ng,'replicates':b,'seed':c['bootstrap_seed']})
        table(out/'phase444_paired_cluster_bootstrap.tsv',cis)
        identity('input_after')
        summary={'status':'PRODUCER_PASS_DESCRIPTIVE_ONLY','cohort_n':n,'phase04_all_test_queries':len(all_test),
                 'phase04_test_queries_not_in_phase12':len(set(all_test.protein_id)-set(ids)),
                 'library_activity_rows':len(lib),'library_reference_proteins':len(ref),
                 'canonical_ec_vs_ec_l4_mismatch_rows':ec_mismatch,'scored_test_pairs':len(frame),
                 'sample_weight_min':float(frame.sample_weight.min()),'sample_weight_max':float(frame.sample_weight.max()),
                 'retrieval_cohort_rows':raw_counts,'partition':part,'contrasts':cis,
                 'missing_rank_sentinel':101,'checks_passed':len(checks),
                 'caveat':'Availability is an evaluation oracle, not predictive precision or 95% safe coverage. No biochemical negatives inferred.'}
        table(out/'phase444_producer_checks.tsv',checks);dump(out/'phase444_producer_summary.json',summary)
        print(json.dumps(summary,allow_nan=False),flush=True)
    except Exception as e:
        dump(out/'FAILED_producer.json',{'error_type':type(e).__name__,'error':str(e),'checks':checks})
        raise

if __name__=='__main__':main()
