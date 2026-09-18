#!/usr/bin/env python3
"""Freeze complete predictions, then evaluate without refitting any policy."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib
from scipy.stats import beta

LEVELS=['EC_L3','EC_L4','EXACT_RHEA']
AK=['query_protein_id','reference_protein_id','reference_activity_id']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def js(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def tsv(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def cluster_precision(accepted,correct,cluster,seed):
    names,inv=np.unique(cluster[accepted],return_inverse=True)
    if not len(names):return None,None,0
    n=np.bincount(inv);s=np.bincount(inv,weights=correct[accepted]);rng=np.random.default_rng(seed);draws=[]
    for start in range(0,10000,50):
        ix=rng.integers(0,len(n),size=(50,len(n)));draws.extend((s[ix].sum(axis=1)/n[ix].sum(axis=1)).tolist())
    bounds=np.quantile(draws,[.025,.975]);return float(bounds[0]),float(bounds[1]),len(names)

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--contract',type=Path,required=True);args=a.parse_args()
    r=args.root.resolve();c=json.loads(args.contract.read_text());out=r/c['output_dir'];assert sys.version_info[:3]==(3,11,15)
    try:
        dr=json.loads((out/'DEEP_INFERENCE_PASS.json').read_text());fr=json.loads((out/'FEATURES_AND_TREE_PASS.json').read_text())
        assert sha(out/'full_deep_predictions.npy')==dr['prediction_sha256'] and sha(out/'full_tree_predictions.npy')==fr['tree_scores_sha256']
        deep=np.load(out/'full_deep_predictions.npy');tree=np.load(out/'full_tree_predictions.npy')
        pred=pl.read_parquet(out/'full_activity_features.parquet',columns=AK+['full_row','legacy_position'])
        assert pred.height==len(deep)==len(tree) and pred['full_row'].to_list()==list(range(len(deep)))
        old=pl.read_parquet(r/'results/phase11/pairwise_predictions.parquet',columns=AK+['query_split_expected']+[f'score_siteguard_{l}' for l in LEVELS]).filter(pl.col('query_split_expected')=='test')
        mapping=pred.select(AK+['full_row']).join(old,on=AK,how='inner',validate='1:1').sort('full_row');assert mapping.height==291730
        cfg=json.loads((r/'models/phase11/siteguard_model_config.json').read_text());prob=np.empty_like(deep);blenderr={}
        for j,l in enumerate(LEVELS):
            alpha=cfg['deep_blend_weights'][l];raw=(alpha*deep[:,j]+(1-alpha)*tree[:,j]).astype(np.float32)
            ix=mapping['full_row'].to_numpy();orig=mapping['score_siteguard_'+l].to_numpy();err=float(np.max(np.abs(raw[ix]-orig)));assert err<=5e-6,(l,err);blenderr[l]=err
            raw[ix]=orig
            cal=joblib.load(r/f'models/phase12/isotonic_{l}.joblib');prob[:,j]=cal.predict(raw).astype(np.float32)
            pred=pred.with_columns(pl.Series('raw_'+l,raw),pl.Series('probability_'+l,prob[:,j]))
        with (out/'full_frozen_predictions.parquet').open('xb') as f:pred.write_parquet(f,compression='zstd')
        js(out/'PREDICTIONS_FROZEN_BEFORE_QUERY_LABEL_JOIN.json',{'status':'PREDICTIONS_FROZEN','rows':pred.height,'sha256':sha(out/'full_frozen_predictions.parquet'),'blend_replay_max_abs':blenderr,'query_function_labels_read_in_scoring':False,'policy':'frozen calibrators/thresholds; legacy-first ties; no fitting'})
        print('PREDICTIONS_FROZEN_BEFORE_QUERY_LABEL_JOIN',pred.height,flush=True)
        # Outcome-only data starts here, after the prediction identity is recorded.
        truth=pl.read_parquet(r/'data/splits/split_sequence.parquet',columns=['protein_id','split','cluster_id_30','ec_l3_json','ec_l4_json','canonical_rhea_json']).filter(pl.col('split')=='test').rename({'protein_id':'query_protein_id'})
        truth=truth.with_columns([pl.col(f).str.json_decode(pl.List(pl.String)).alias('truth_'+l) for f,l in zip(['ec_l3_json','ec_l4_json','canonical_rhea_json'],LEVELS)])
        lib=pl.read_parquet(r/'data/reference/activity_reference_library.parquet',columns=['activity_id','reference_protein_id','ec_l3','ec_l4','canonical_rhea']).rename({'activity_id':'reference_activity_id'})
        labelled=pred.join(lib,on=['reference_activity_id','reference_protein_id'],validate='m:1').join(truth.select(['query_protein_id','cluster_id_30']+['truth_'+l for l in LEVELS]),on='query_protein_id',validate='m:1')
        assert labelled.height==pred.height
        labelled=labelled.with_columns([pl.col('truth_'+l).list.contains(pl.col(f)).fill_null(False).alias('correct_'+l) for l,f in zip(LEVELS,['ec_l3','ec_l4','canonical_rhea'])])
        oldfrozen=pl.read_parquet(r/'results/phase12/end_to_end_predictions.parquet').sort('query_protein_id')
        rows=[]
        for cohort in ['sampled','full']:
            use=labelled.filter(pl.col('legacy_position').is_not_null()) if cohort=='sampled' else labelled
            for l in LEVELS:
                oracle=use.group_by('query_protein_id').agg(pl.col('correct_'+l).any().alias('candidate_available'),pl.len().alias('candidate_activities'))
                # Sort score descending, then original full-row order: identical idxmax tie convention.
                top=use.sort(['query_protein_id','probability_'+l,'full_row'],descending=[False,True,False]).unique('query_protein_id',keep='first',maintain_order=True).sort('query_protein_id')
                top=top.select('query_protein_id','cluster_id_30','reference_protein_id','reference_activity_id',pl.col('probability_'+l).alias('probability'),pl.col('correct_'+l).alias('correct')).join(oracle,on='query_protein_id').sort('query_protein_id')
                assert top.height==27639
                if cohort=='sampled':
                    assert top['query_protein_id'].to_list()==oldfrozen['query_protein_id'].to_list()
                    assert top['reference_activity_id'].to_list()==oldfrozen['top_reference_activity_'+l].to_list()
                    assert top['correct'].to_list()==oldfrozen['top_correct_'+l].to_list()
                    assert top['probability'].to_list()==oldfrozen['top_probability_'+l].to_list()
                    assert top['candidate_available'].to_list()==oldfrozen['oracle_candidate_available_'+l].to_list()
                for x in top.to_dicts():rows.append(dict(cohort=cohort,level=l,**x))
        ledger=pl.DataFrame(rows)
        with (out/'phase445_query_results.parquet').open('xb') as f:ledger.write_parquet(f,compression='zstd')
        thresholds=pd.read_csv(r/'results/phase12/abstention_thresholds.tsv',sep='\t');metrics=[];contrasts=[]
        for l in LEVELS:
            full=ledger.filter((pl.col('level')==l)&(pl.col('cohort')=='full')).sort('query_protein_id')
            sampled=ledger.filter((pl.col('level')==l)&(pl.col('cohort')=='sampled')).sort('query_protein_id')
            assert full['query_protein_id'].to_list()==sampled['query_protein_id'].to_list()
            clusters=full['cluster_id_30'].to_numpy();names,inv=np.unique(clusters,return_inverse=True);n=len(clusters);group_n=np.bincount(inv)
            for target in [.9,.95]:
                th=float(thresholds.loc[thresholds.annotation_level.eq(l)&thresholds.target_precision.eq(target),'threshold'].iloc[0]);vectors=[]
                for cohort,data in [('sampled',sampled),('full',full)]:
                    accepted=data['probability'].to_numpy()>=th;correct=data['correct'].to_numpy();na=int(accepted.sum());ns=int((accepted&correct).sum());prec=ns/na if na else None
                    low,high,ncl=cluster_precision(accepted,correct,clusters,c['seed'])
                    cp=float(beta.ppf(.025,ns,na-ns+1)) if ns else (0.0 if na else None)
                    qualifies=na>=50 and ncl>=20 and prec>=.95 and low>=.95
                    metrics.append({'level':l,'cohort':cohort,'validation_target':target,'threshold':th,'queries':n,'candidate_available':int(data['candidate_available'].sum()),'candidate_availability':float(data['candidate_available'].mean()),'top1_correct':int(correct.sum()),'top1_accuracy':float(correct.mean()),'accepted':na,'accepted_correct':ns,'coverage':na/n,'precision':prec,'accepted_correct_fraction':ns/n,'accepted_disagreement_fraction':(na-ns)/n,'accepted_clusters':ncl,'cluster_precision_ci95_low':low,'cluster_precision_ci95_high':high,'query_exact_ci95_low':cp,'meets_descriptive_95_rule':bool(qualifies),'rule_qualified_95_coverage':na/n if qualifies else 0.0,'new_external_certification':False})
                    vectors.extend([accepted.astype(int),(accepted&correct).astype(int)])
                ds=np.stack([vectors[2]-vectors[0],vectors[3]-vectors[1]],axis=1)
                totals=np.stack([np.bincount(inv,weights=ds[:,j],minlength=len(names)) for j in range(2)],axis=1)
                rng=np.random.default_rng(c['seed']);boot=[]
                for start in range(0,2000,25):
                    ix=rng.integers(0,len(names),size=(25,len(names)));boot.extend(totals[ix].sum(axis=1)/group_n[ix].sum(axis=1)[:,None])
                boot=np.array(boot)
                for j,name in enumerate(['coverage_full_minus_sampled','accepted_correct_fraction_full_minus_sampled']):
                    lo,hi=np.quantile(boot[:,j],[.025,.975]);contrasts.append({'level':l,'validation_target':target,'contrast':name,'difference':float(ds[:,j].mean()),'ci95_low':float(lo),'ci95_high':float(hi),'clusters':len(names),'replicates':2000})
        tsv(out/'phase445_metrics.tsv',metrics);tsv(out/'phase445_paired_contrasts.tsv',contrasts)
        identities=[]
        for x in c['inputs']:
            p=Path(x['path']) if x.get('absolute') else r/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p);identities.append(x)
        js(out/'phase445_producer_summary.json',{'status':'PRODUCER_PASS_DESCRIPTIVE_ONLY','queries':27639,'activity_candidates':pred.height,'sampled_activity_candidates':291730,'metrics':metrics,'contrasts':contrasts,'inputs_unchanged':True,'no_threshold_refit':True,'no_new_model':True,'no_external_EvidenceJudge_endpoint_changed':True})
        print(json.dumps({'status':'PRODUCER_PASS_DESCRIPTIVE_ONLY','activity_candidates':pred.height,'metrics':metrics}),flush=True)
    except Exception as e:
        js(out/'FAILED_evaluation.json',{'type':type(e).__name__,'error':str(e)});raise

if __name__=='__main__':main()
