#!/usr/bin/env python3
"""Truth-free complete candidate features and frozen tree inference."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def save(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def mod(p):
    s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def writepq(frame,p):
    with p.open('xb') as f:frame.write_parquet(f,compression='zstd')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--contract',type=Path,required=True);a=ap.parse_args()
    import numpy as np,pandas as pd,polars as pl,lightgbm as lgb
    r=a.root.resolve();c=json.loads(a.contract.read_text());out=r/c['output_dir']
    assert str(r)==c['resolved_root'] and sys.version_info[:3]==(3,11,15)
    for x in c['inputs']:
        p=Path(x['path']) if x.get('absolute') else r/x['path']
        assert p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
    out.mkdir(exist_ok=False)
    save(out/'reservation.json',{'status':'STARTED','contract_sha256':sha(a.contract)})
    try:
        key=['query_protein_id','reference_protein_id'];ak=key+['reference_activity_id']
        old=pl.read_parquet(r/'results/phase11/pairwise_predictions.parquet',columns=ak+['query_split_expected']+[f'score_{m}_{l}' for m in ['lightgbm_global','deep_global','siteguard'] for l in ['EC_L3','EC_L4','EXACT_RHEA']]).filter(pl.col('query_split_expected')=='test').with_row_index('legacy_position')
        ids=pl.read_parquet(r/'results/phase12/end_to_end_predictions.parquet',columns=['query_protein_id'])
        assert ids.height==27639 and ids.unique().height==27639 and set(old['query_protein_id'])==set(ids['query_protein_id'])
        pairs=pl.scan_parquet(r/'data/interim/phase05/retrieval_pair_universe.parquet').join(ids.lazy(),on='query_protein_id',how='semi').collect()
        assert pairs.unique(key).height==pairs.height and pairs['query_partition'].eq('test').all()
        assert pairs['best_retrieval_rank'].max()<=50
        split=pl.read_parquet(r/'data/splits/split_sequence.parquet',columns=['protein_id','split','cluster_id_30'])
        q=split.rename({'protein_id':'query_protein_id','split':'qsplit','cluster_id_30':'qcl'})
        rr=split.rename({'protein_id':'reference_protein_id','split':'rsplit','cluster_id_30':'rcl'})
        qc=pairs.select(key).join(q,on='query_protein_id').join(rr,on='reference_protein_id')
        assert qc.height==pairs.height and qc['qsplit'].eq('test').all() and qc['rsplit'].eq('train').all() and (qc['qcl']!=qc['rcl']).all()
        cached=pl.read_parquet(r/'data/interim/phase06/global_pair_features.parquet').join(pairs.select(key),on=key,how='semi')
        missing=pairs.join(cached.select(key),on=key,how='anti').rename({'query_partition':'query_split_expected'})
        g=mod(r/'scripts/phase06_build_global_features.py');lookup=mod(r/'scripts/phase06_prepare_direct_alignments.py')
        missing=g.attach_direct_alignments(r,missing)
        proteins=g.build_protein_metadata(r)
        available=set(proteins.loc[proteins.has_structure,'protein_id'])
        direct_counts={}
        for modality,prefix,feature in [('mmseqs','sequence','sequence_identity'),('foldseek','structure','foldseek_identity')]:
            need=missing.filter(pl.col(feature).is_null()).select(key)
            if modality=='foldseek':need=need.filter(pl.col(key[0]).is_in(list(available))&pl.col(key[1]).is_in(list(available)))
            direct_counts[modality]=need.height
            if not need.height:continue
            dbdir=r/'databases'/modality
            qdb=dbdir/('query_seq' if modality=='mmseqs' else 'query_structure')
            rdb=dbdir/('reference_seq' if modality=='mmseqs' else 'reference_structure')
            qm=lookup.sequence_lookup(Path(str(qdb)+'.lookup')) if modality=='mmseqs' else lookup.structure_lookup(qdb)
            rm=lookup.sequence_lookup(Path(str(rdb)+'.lookup')) if modality=='mmseqs' else lookup.structure_lookup(rdb)
            keyed=lookup.attach_keys(need,qm,rm)
            assert keyed['query_db_key'].null_count()==0 and keyed['target_db_key'].null_count()==0
            d=out/('new_direct_'+modality);d.mkdir(exist_ok=False)
            inp=d/'known_pairs.tsv'
            keyed.select('query_db_key','target_db_key',pl.lit(2000).alias('prefilter_score'),pl.lit(0).alias('diagonal')).sort(['query_db_key','target_db_key']).write_csv(inp,separator='\t',include_header=False)
            tool=c['alignment_binaries'][modality];pref=d/'pref';aln=d/'aln';hits=d/'hits.tsv';threads=str(c['cpus'])
            commands=[[tool,'tsv2db',str(inp),str(pref),'--output-dbtype','7','-v','1'],
                      [tool,'align' if modality=='mmseqs' else 'structurealign',str(qdb),str(rdb),str(pref),str(aln)]+([] if modality=='mmseqs' else ['--alignment-type','2'])+['--alignment-mode','3','-e','1e100','--threads',threads],
                      [tool,'convertalis',str(qdb),str(rdb),str(aln),str(hits),'--format-output','query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits']]
            for z,cmd in enumerate(commands):
                with (d/f'command_{z}.log').open('xb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            cols=['query','target','fident','alnlen','qlen','tlen','qcov','tcov','evalue','bits']
            hit=pl.read_csv(hits,separator='\t',has_header=False,new_columns=cols,schema_overrides={'query':pl.String,'target':pl.String,**{x:pl.Float32 for x in ['fident','qcov','tcov','bits']}})
            if modality=='foldseek':hit=hit.with_columns(pl.col('query').str.extract(r'^AF-(.+)-F[0-9]+-model_v[0-9]+$',1),pl.col('target').str.extract(r'^AF-(.+)-F[0-9]+-model_v[0-9]+$',1))
            hit=hit.rename({'query':key[0],'target':key[1]})
            assert hit.height==need.height and hit.unique(key).height==need.height and need.join(hit.select(key),on=key,how='anti').height==0
            mappings={'fident':feature,'qcov':('sequence_query_coverage' if modality=='mmseqs' else 'foldseek_query_coverage'),'tcov':('sequence_reference_coverage' if modality=='mmseqs' else 'foldseek_reference_coverage'),'bits':('sequence_bitscore' if modality=='mmseqs' else 'foldseek_bitscore')}
            hit=hit.select(key+[pl.col(x).alias('new_'+dest) for x,dest in mappings.items()])
            missing=missing.join(hit,on=key,how='left',validate='1:1').with_columns([pl.coalesce([dest,'new_'+dest]).cast(pl.Float32).alias(dest) for dest in mappings.values()]).drop(['new_'+x for x in mappings.values()])
            print('NEW_DIRECT_COMPLETE',modality,need.height,flush=True)
        assert missing['sequence_identity'].null_count()==0
        ms=missing.filter(pl.col(key[0]).is_in(list(available))&pl.col(key[1]).is_in(list(available)))
        assert ms['foldseek_identity'].null_count()==0
        emb=np.load(r/'data/interim/phase06/esm2_t33_v4_embeddings.npy',mmap_mode='r')
        eindex=pd.read_csv(r/'data/interim/phase06/esm2_t33_v4_index.tsv',sep='\t',dtype={'protein_id':str})
        added=g.compute_dense_pair_features(missing,proteins,emb,eindex)
        pair_features=pl.concat([cached,added],how='vertical_relaxed')
        assert pair_features.height==pairs.height and pair_features.unique(key).height==pairs.height
        writepq(pair_features,out/'full_protein_pair_features.parquet')
        lib=pl.read_parquet(r/'data/reference/activity_reference_library.parquet',columns=['activity_id','reference_protein_id','ec_l3','canonical_rhea','reference_partition']).filter(pl.col('reference_partition')=='train').rename({'activity_id':'reference_activity_id'})
        activities=pairs.select(key).join(lib,on='reference_protein_id',how='inner').drop('reference_partition')
        features=activities.join(pair_features,on=key,how='left',validate='m:1').with_columns(pl.col('ec_l3').str.extract(r'^([0-9]+)',1).cast(pl.UInt8,strict=False).alias('reference_ec_l1')).join(g.reaction_features(r),on='canonical_rhea',how='left',validate='m:1').with_columns(pl.col('reference_reaction_available').fill_null(False)).drop(['ec_l3','canonical_rhea'])
        features=features.join(old.select(ak+['legacy_position']),on=ak,how='left',validate='1:1').sort(['legacy_position',*ak],nulls_last=True).with_row_index('full_row')
        assert features.unique(ak).height==features.height and features['legacy_position'].drop_nulls().n_unique()==old.height
        writepq(features,out/'full_activity_features.parquet')
        pre=json.loads((r/'data/interim/phase11/preprocessing.json').read_text())
        rawcols=pre['numeric_columns']+pre['categorical_columns']
        assert set(rawcols)==set(features.columns)-set(ak+['query_split','legacy_position','full_row'])
        assert not any('ground_truth' in x or x in ['same_ec_l3','same_ec_l4','same_exact_rhea','canonical_rhea'] for x in rawcols)
        pdf=features.select(rawcols).to_pandas()
        cats=pd.get_dummies(pdf[pre['categorical_columns']].fillna('MISSING').astype(str),prefix=pre['categorical_columns'],dtype=np.float32).reindex(columns=pre['categorical_dummy_columns'],fill_value=0)
        numeric=pdf[pre['numeric_columns']].apply(pd.to_numeric,errors='coerce').astype(np.float32)
        tree_x=np.column_stack([numeric.to_numpy(np.float32),cats.to_numpy(np.float32)])
        num=numeric.to_numpy(np.float32);median=np.array([pre['numeric_medians'][k] for k in pre['numeric_columns']],np.float32)
        mean=np.array([pre['numeric_means'][k] for k in pre['numeric_columns']],np.float32);scale=np.array([pre['numeric_scales'][k] for k in pre['numeric_columns']],np.float32)
        num=np.where(np.isnan(num),median,num);deep_x=np.column_stack([(num-mean)/scale,cats.to_numpy(np.float32)])
        meta=pl.read_parquet(r/'data/interim/phase11/evaluation_metadata.parquet',columns=ak+['query_split_expected']).with_row_index('eval_row').filter(pl.col('query_split_expected')=='test')
        replay=features.select(ak+['full_row']).join(meta,on=ak,how='inner',validate='1:1').sort('eval_row')
        assert replay.height==old.height
        old_x=np.load(r/'data/interim/phase11/eval_X.npy',mmap_mode='r')[replay['eval_row'].to_numpy()]
        new_x=deep_x[replay['full_row'].to_numpy()]
        err=float(np.max(np.abs(old_x-new_x)));assert np.allclose(old_x,new_x,rtol=2e-6,atol=2e-6),('feature_replay',err)
        with (out/'full_deep_X.npy').open('xb') as f:np.save(f,deep_x)
        old_join=features.select(ak+['full_row']).join(old,on=ak,how='inner',validate='1:1').sort('full_row')
        tree=np.empty((len(tree_x),3),np.float32);tree_err={}
        config=json.loads((r/'models/phase10/baseline_config.json').read_text());assert config['feature_columns']==pre['model_columns']
        for j,l in enumerate(['EC_L3','EC_L4','EXACT_RHEA']):
            model=lgb.Booster(model_file=str(r/f'models/phase10/lightgbm_global_{l}.txt'))
            tree[:,j]=model.predict(tree_x,num_threads=c['cpus']).astype(np.float32)
            error=float(np.max(np.abs(tree[old_join['full_row'].to_numpy(),j]-old_join['score_lightgbm_global_'+l].to_numpy())))
            tree_err[l]=error;assert error<=2e-6,(l,error)
        with (out/'full_tree_predictions.npy').open('xb') as f:np.save(f,tree)
        writepq(old_join.select(['full_row']+[f'score_deep_global_{l}' for l in ['EC_L3','EC_L4','EXACT_RHEA']]),out/'deep_replay_index.parquet')
        # Plain arrays let the original HPC PyTorch environment run without pandas/Arrow.
        with (out/'deep_replay_rows.npy').open('xb') as f:np.save(f,old_join['full_row'].to_numpy())
        with (out/'deep_replay_expected.npy').open('xb') as f:np.save(f,old_join.select([f'score_deep_global_{l}' for l in ['EC_L3','EC_L4','EXACT_RHEA']]).to_numpy())
        summary={'status':'FEATURES_AND_TREE_PASS','queries':ids.height,'protein_pairs':pairs.height,'cached_protein_pairs':cached.height,'new_protein_pairs':missing.height,'new_direct_alignment_counts':direct_counts,'activity_candidates':features.height,'legacy_scored_rows':old.height,'feature_replay_max_abs':err,'tree_replay_max_abs':tree_err,'query_function_labels_read':False,'feature_matrix_sha256':sha(out/'full_deep_X.npy'),'tree_scores_sha256':sha(out/'full_tree_predictions.npy')}
        save(out/'FEATURES_AND_TREE_PASS.json',summary);print(json.dumps(summary),flush=True)
    except Exception as e:
        save(out/'FAILED_prepare.json',{'type':type(e).__name__,'error':str(e)});raise

if __name__=='__main__':main()
