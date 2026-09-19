"""Frozen, post-hoc utility comparison; no fitting and no inference."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ['judge', 'hit_ec4_logit', 'hit_ec4_softmax', 'hit_ec4_margin', 'hit_native_level_logit']
Q = [0.10, 0.25, 0.50, 0.75]
SEED = 20260819


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def write_table(p, rows):
    pd.DataFrame(rows).to_csv(p, sep='\t', index=False, mode='x', float_format='%.17g')


def boolean(s):
    v=s.astype(str).str.lower()
    assert v.isin(['true','false','1','0']).all(), 'Invalid boolean'
    return v.isin(['true','1'])


def tie_info(score, y, order, k):
    boundary = score[order[k-1]]
    above = score > boundary
    tied = score == boundary
    m, slots = int(tied.sum()), k-int(above.sum())
    a, good = int(y[above].sum()), int(y[tied].sum())
    lo = a+max(0,slots-(m-good))
    hi = a+min(slots,good)
    return dict(boundary_score=float(boundary), boundary_tie_size=m,
                boundary_tie_slots=slots, tie_precision_min=lo/k,
                tie_precision_max=hi/k, tie_precision_expected=(a+slots*good/m)/k)


def bootstrap(y, cluster_index, orders, reps):
    assert len(y)>0 and len(y)==len(cluster_index), 'Empty or mismatched sample'
    nc = int(cluster_index.max())+1
    rng = np.random.default_rng(SEED)
    values = np.empty((reps,len(Q),len(METHODS)))
    counts = np.empty(reps,dtype=np.int64)
    for b in range(reps):
        w = np.bincount(rng.integers(0,nc,nc),minlength=nc)[cluster_index]
        n = int(w.sum()); counts[b] = n
        ks = np.floor(np.asarray(Q)*n).astype(int)
        assert (ks>0).all()
        for mi, method in enumerate(METHODS):
            order=orders[method]; wo=w[order]
            cumulative=np.cumsum(wo)
            j=np.searchsorted(cumulative,ks,side='left')
            correct=np.cumsum(wo*y[order])
            # Remove copies past the quota from its final ranked query.
            values[b,:,mi]=(correct[j]-(cumulative[j]-ks)*y[order[j]])/ks
    return values,counts


def self_test():
    y=np.array([1,0,1,0],dtype=int); score=np.array([.9,.9,.8,.7]); order=np.arange(4)
    t=tie_info(score,y,order,1)
    assert t['tie_precision_min']==0 and t['tie_precision_max']==1 and t['tie_precision_expected']==.5
    # Weighted boundary can cut through copies of one query; independently expand.
    w=np.array([2,3,0,1]); k=3; c=np.cumsum(w); j=np.searchsorted(c,k)
    got=(np.cumsum(w*y)[j]-(c[j]-k)*y[j])/k
    assert got==np.repeat(y,w)[:k].mean()==2/3
    try:
        bootstrap(np.array([]),np.array([],dtype=int),{},1)
    except AssertionError:
        pass
    else:
        raise AssertionError('empty sample accepted')
    assert not np.isfinite(np.array([1.0,np.nan])).all()
    assert len(set(['q','q']))!=2
    try:
        boolean(pd.Series(['unknown']))
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid boolean accepted')
    try:
        bootstrap(np.ones(1),np.array([0]),{m:np.array([0]) for m in METHODS},1)
    except AssertionError:
        pass
    else:
        raise AssertionError('zero quota accepted')
    assert hashlib.sha256(b'original').hexdigest()!=hashlib.sha256(b'tampered').hexdigest()
    print('PHASE448_SELF_TEST_PASS')


def verify_inputs(root, c):
    assert str(root.resolve())==c['resolved_root']
    assert sys.version_info[:3]==(3,11,15)
    assert c['budgets']==Q and c['seed']==SEED and c['methods']==METHODS
    for x in c['inputs']:
        p=root/x['path']
        assert p.stat().st_size==x['bytes'] and digest(p)==x['sha256'],x['path']


def load_cohort(root, phase):
    base=root/f'results/phase{phase}'
    j=pd.read_parquet(base/'external_blind_predictions.parquet')
    b=pd.read_parquet(base/'external_blind_single_tool_predictions.parquet')
    b=b[b.method.eq('HIT_EC')].copy()
    e=pd.read_parquet(base/'external_predictions.parquet')
    t=pd.read_csv(base/'external_canonicalized_truth_audit.tsv',sep='\t',dtype=str).rename(columns={'query_id':'query_protein_id'})
    h=pd.read_csv(root/f'data/interim/phase{phase}_hit_ec/phase{phase}_hit_ec_predictions.tsv',sep='\t',dtype={'query_id':str,'ec3_top1':str,'ec4_top1':str})
    keys=['query_protein_id','annotation_level']
    for frame in [j,b,e,t]:
        assert not frame.duplicated(keys).any()
        assert frame[keys].notna().all().all()
    assert not h.query_id.duplicated().any()
    kj=set(map(tuple,j[keys].to_numpy()))
    assert all(set(map(tuple,x[keys].to_numpy()))==kj for x in [b,e,t])
    assert set(h.query_id)==set(j.query_protein_id)
    for x in [b,e]:
        z=j.merge(x,on=keys,validate='one_to_one',suffixes=('_j','_x'))
        for col in ['candidate_label','query_cluster_id_30']:
            assert z[col+'_j'].eq(z[col+'_x']).all(),col
    ee=j.merge(e,on=keys,validate='one_to_one',suffixes=('_j','_e'))
    assert np.array_equal(ee.selection_score_j.to_numpy(),ee.selection_score_e.to_numpy())
    d=j.rename(columns={'selection_score':'judge'}).merge(h,left_on='query_protein_id',right_on='query_id',validate='many_to_one')
    d=d.merge(t,on=keys,validate='one_to_one')
    assert d.query_cluster_id_30.eq(d.external_cluster_id_30).all()
    assert d.query_cluster_id_30.notna().all() and d.candidate_label.notna().all()
    d['eligible']=boolean(d.label_eligible)
    d['frozen_accepted']=boolean(d.accepted)
    d['correct']=d.eligible & d.candidate_label.eq(d.truth_label)
    zz=d[keys+['correct','eligible','truth_label']].merge(e[keys+['correct','label_eligible','truth_label']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert boolean(zz.correct_old).eq(zz.correct_new).all()
    assert boolean(zz.label_eligible).eq(zz.eligible).all()
    assert zz.loc[zz.eligible,'truth_label_new'].eq(zz.loc[zz.eligible,'truth_label_old']).all()
    d['hit_ec4_logit']=d.ec4_top1_logit
    d['hit_ec4_softmax']=d.ec4_top1_softmax
    d['hit_ec4_margin']=d.ec4_softmax_margin12
    d['hit_native_level_logit']=np.where(d.annotation_level.eq('EC_L3'),d.ec3_top1_logit,d.ec4_top1_logit)
    assert np.isfinite(d[METHODS].to_numpy(dtype=float)).all()
    d['tie_key']=d.query_protein_id.map(lambda s:hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest())
    d['native_ec3_head_matches_candidate']=np.where(d.annotation_level.eq('EC_L3'),d.candidate_label.eq(d.ec3_top1),True)
    # Reproduce old frozen endpoints as a provenance/QC check, never as quota selection.
    old=pd.read_csv(base/'external_operating_points.tsv',sep='\t')
    frozen=[]
    for level,x in d[d.eligible].groupby('annotation_level'):
        a=x[x.frozen_accepted]; r=old[old.annotation_level.eq(level)&old.system.eq('HIT_ANCHORED_EVIDENCEJUDGE_V2')].iloc[0]
        assert (len(x),len(a),int(a.correct.sum()))==(int(r.total_eligible_queries),int(r.accepted_queries),int(r.correct))
        frozen.append(dict(phase=phase,level=level,n=len(x),accepted=len(a),correct=int(a.correct.sum()),precision=float(a.correct.mean()),coverage=len(a)/len(x)))
    return d,frozen


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path);p.add_argument('--contract',type=Path)
    p.add_argument('--self-test',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.self_test:
        self_test();return
    root=a.root;c=json.loads(a.contract.read_text());verify_inputs(root,c)
    out=root/c['output_dir'];assert not out.exists()
    if a.preflight:
        for phase in [31,33]:
            lock=json.loads((root/f'models/phase{phase}_external_inference_lock.json').read_text())
            assert lock['status']=='PASS' and not lock['truth_opened']
        print('PHASE448_PREFLIGHT_PASS');return
    out.mkdir(exist_ok=False)
    write_json(out/'reservation.json',{'phase':448,'started_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.getenv('SLURM_JOB_ID'),'status':'FORMAL_SINGLE_ATTEMPT','contract_sha256':digest(a.contract)})
    grids=[];curves=[];contrasts=[];ledger=[];exclusions=[];cohort_qc=[];normalized=[];frozen=[];query_sets={}
    for phase in [31,33]:
        d,fr=load_cohort(root,phase);frozen.extend(fr);query_sets[phase]=set(d.query_protein_id)
        for row in d[~d.eligible].to_dict('records'):
            exclusions.append({k:row[k] for k in ['query_protein_id','annotation_level','truth_canonicalization_status']}|{'phase':phase,'reason':'INELIGIBLE_FROZEN_CANONICAL_TRUTH'})
        for level,g in d[d.eligible].groupby('annotation_level',sort=True):
            g=g.sort_values('query_protein_id').reset_index(drop=True);n=len(g)
            clusters=sorted(g.query_cluster_id_30.unique());ci=np.array([clusters.index(x) for x in g.query_cluster_id_30])
            y=g.correct.to_numpy(dtype=int);score={m:g[m].to_numpy(dtype=float) for m in METHODS}
            # Rankings are exclusively score + truth-free deterministic key.
            orders={m:g.sort_values([m,'tie_key','query_protein_id'],ascending=[False,True,True]).index.to_numpy() for m in METHODS}
            values,counts=bootstrap(y,ci,orders,c['bootstrap_replicates'])
            with (out/f'bootstrap_phase{phase}_{level}.npz').open('xb') as f:
                np.savez_compressed(f,precision=values,resampled_n=counts)
            group=dict(phase=phase,level=level,n=n,n_clusters=len(clusters))
            cohort_qc.append(group|{'raw_queries':int(d[d.annotation_level.eq(level)].shape[0]),'excluded_truth':int(d[d.annotation_level.eq(level)&~d.eligible].shape[0]),'raw_correct':int(y.sum()),'raw_precision':float(y.mean()),'native_ec3_head_mismatches':int((~g.native_ec3_head_matches_candidate).sum()),'unique_scores':{m:int(g[m].nunique()) for m in METHODS}})
            for row in g[['query_protein_id','annotation_level','query_cluster_id_30','candidate_label','truth_label','correct','tie_key','native_ec3_head_matches_candidate']+METHODS].to_dict('records'):
                normalized.append({'phase':phase}|row)
            for mi,m in enumerate(METHODS):
                order=orders[m];cs=np.cumsum(y[order])
                for k in range(1,n+1):
                    curves.append(group|dict(method=m,accepted=k,correct=int(cs[k-1]),coverage=k/n,precision=float(cs[k-1]/k)))
                for qi,q in enumerate(Q):
                    k=math.floor(q*n);accepted=order[:k];lo,hi=np.quantile(values[:,qi,mi],[.025,.975])
                    grids.append(group|dict(method=m,budget=q,accepted=k,accepted_clusters=int(np.unique(ci[accepted]).size),coverage=k/n,correct=int(y[accepted].sum()),disagreement=int(k-y[accepted].sum()),precision=float(y[accepted].mean()),precision_ci_low=float(lo),precision_ci_high=float(hi))|tie_info(score[m],y,order,k))
            for qi,q in enumerate(Q):
                k=math.floor(q*n);sj=set(orders['judge'][:k]);pj=float(y[list(sj)].mean())
                for mi,m in enumerate(METHODS[1:],1):
                    sb=set(orders[m][:k]);pb=float(y[list(sb)].mean());added=sorted(sj-sb);removed=sorted(sb-sj)
                    diff=100*(values[:,qi,0]-values[:,qi,mi]);lo,hi=np.quantile(diff,[.025,.975])
                    contrasts.append(group|dict(baseline=m,primary=m=='hit_ec4_logit',budget=q,accepted=k,coverage=k/n,judge_precision=pj,hit_precision=pb,delta_precision_pp=100*(pj-pb),delta_ci_low_pp=float(lo),delta_ci_high_pp=float(hi),net_correct=int(y[added].sum()-y[removed].sum()),added_correct=int(y[added].sum()),new_disagreement=len(added)-int(y[added].sum()),lost_correct=int(y[removed].sum()),avoided_disagreement=len(removed)-int(y[removed].sum()),both_accepted=len(sj&sb),neither=n-len(sj|sb)))
                    if m=='hit_ec4_logit':
                        for i,row in g.iterrows():
                            status='both' if i in sj&sb else 'judge_only' if i in sj else 'hit_only' if i in sb else 'neither'
                            ledger.append(dict(phase=phase,level=level,budget=q,query_id=row.query_protein_id,cluster_id=row.query_cluster_id_30,candidate_label=row.candidate_label,documented_truth=row.truth_label,documented_correct=bool(y[i]),judge_score=float(score['judge'][i]),hit_score=float(score[m][i]),judge_accepted=i in sj,hit_accepted=i in sb,decision=status,interpretation='unchanged_candidate_not_label_correction'))
            print(f'COMPLETE phase{phase} {level} N={n} clusters={len(clusters)}',flush=True)
    verify_inputs(root,c)
    for name,rows in [('grid_all_methods.tsv',grids),('full_precision_coverage_curves.tsv',curves),('paired_utility_comparisons.tsv',contrasts),('per_query_decisions.tsv',ledger),('exclusions.tsv',exclusions),('normalized_evaluation_records.tsv',normalized),('frozen_endpoint_reproduction.tsv',frozen)]:
        write_table(out/name,rows)
    write_json(out/'cohort_qc.json',{'cohorts':cohort_qc,'cross_cohort_query_id_overlap':len(query_sets[31]&query_sets[33]),'overlap_note':'Query-ID check is not a sequence-overlap or training-leakage re-audit; no pooled inference.','excluded_missing_scores':0,'candidate_mismatches':0,'truth_usage':'evaluation_only','post_hoc':True})
    write_json(out/'producer_terminal.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','phase':448,'post_hoc':True,'new_fit':False,'threshold_changes':False,'python':sys.version,'software':{x:importlib.metadata.version(x) for x in ['numpy','pandas','pyarrow','matplotlib']},'git_commit':None,'git_note':'Local authoritative tree is not a Git checkout; source identities are SHA256-locked.','finished_utc':datetime.now(timezone.utc).isoformat(),'inputs_unchanged':True,'grid_rows':len(grids),'comparison_rows':len(contrasts),'ledger_rows':len(ledger)})
    print('PHASE448_PRODUCER_COMPLETE')


if __name__=='__main__':
    main()
