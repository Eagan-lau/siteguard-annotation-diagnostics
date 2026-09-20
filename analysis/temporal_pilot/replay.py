"""Reproduce fixed-score temporal selection and paired outcomes. Python + NumPy."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

LEVELS=('EC_L3','EC_L4','EXACT_RHEA')
FIELDS={'EC_L3':'ec_l3','EC_L4':'ec_l4','EXACT_RHEA':'canonical_rhea'}

def read_tsv(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def emit(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def write_tsv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def run(root,out):
 root=root.resolve();assert not out.exists();out.mkdir(parents=True)
 manifest=root/'MANIFEST.tsv'
 if manifest.exists():
  for r in read_tsv(manifest):assert hashlib.sha256((root/r['file']).read_bytes()).hexdigest()==r['sha256'],r['file']
 rows=read_tsv(root/'inputs/scored_candidates.tsv');features=read_tsv(root/'inputs/features.tsv')
 p=read_json(root/'inputs/preprocessing.json');x=np.load(root/'inputs/X_temporal.npy')
 cols=[]
 for n in p['numeric_columns']:
  a=np.array([float(r[n]) if r[n] else np.nan for r in features],dtype=np.float32)
  a[np.isnan(a)]=np.float32(p['numeric_medians'][n]);cols.append((a-np.float32(p['numeric_means'][n]))/np.float32(p['numeric_scales'][n]))
 for n in p['categorical_columns']:
  cols.extend(np.array([(r[n] or 'MISSING')==v for r in features],dtype=np.float32) for v in p['categorical_levels'][n])
 assert np.array_equal(np.column_stack(cols),x),'Trained input column order / preprocessing'
 raw=np.stack([np.load(root/f'inputs/raw_seed_{seed}.npy').astype(np.float64) for seed in [20260819,20260820,20260821]]).mean(axis=0).astype(np.float32)
 assert np.array_equal(raw,np.load(root/'inputs/raw_ensemble.npy'))
 calibrators=read_json(root/'inputs/calibrators.json')['levels'];threshold=read_json(root/'inputs/release_rule.json')['thresholds']
 calibrated={l:np.interp(raw[:,j],calibrators[l]['x_thresholds'],calibrators[l]['y_thresholds']).astype(np.float32) for j,l in enumerate(LEVELS)}
 for j,l in enumerate(LEVELS):
  assert np.array_equal(raw[:,j],np.array([r['raw_'+l] for r in rows],dtype=np.float32))
  assert np.array_equal(calibrated[l],np.array([r['calibrated_'+l] for r in rows],dtype=np.float32))
 truth={r['query_id']:r for r in read_json(root/'inputs/annotation_endpoints.json')}
 component={r['query_id']:int(r['component_id']) for r in read_tsv(root/'inputs/query_components.tsv')}
 saved={m:{r['query_id']:r for r in read_json(root/f'inputs/{m}_predictions.json')} for m in ['baseline','intervention']}
 results=[];exposure=[];outputs={m:[] for m in saved}
 for q in sorted(truth):
  ids=[i for i,r in enumerate(rows) if r['query_protein_id']==q]
  ec4=set(truth[q]['complete_ec']);ec3={'.'.join(e.split('.')[:3]) for e in ec4}
  endpoints={'EC_L3':ec3,'EC_L4':ec4,'EXACT_RHEA':set(truth[q]['canonical_rhea'])}
  ec4ids=[i for i in ids if rows[i]['ec_l4']]
  maximum=max(float(calibrated['EC_L4'][i]) for i in ec4ids)
  match=[i for i in ec4ids if rows[i]['ec_l4'] in ec4]
  best=max((float(calibrated['EC_L4'][i]) for i in match),default=None)
  exposure.append({'query_id':q,'retained_matching_activity_rows':len(match),'maximum_calibrated_ec4':maximum,'highest_matching_calibrated_ec4':best,'matching_label_in_maximum_set':best==maximum,'baseline_concordant':saved['baseline'][q]['top1_EC_L4'] in ec4,'diagnostic_state':'no_retained_match' if not match else 'concordant_selection' if saved['baseline'][q]['top1_EC_L4'] in ec4 else 'maximum_tie_loss' if best==maximum else 'strict_score_loss'})
  result={'query_id':q,'component_id':component[q]}
  for mode in saved:
   chosen={}
   for level in LEVELS:
    ix=np.array([i for i in ids if rows[i][FIELDS[level]]],dtype=int)
    if not len(ix):chosen[level]=None;continue
    maxima=ix[calibrated[level][ix]==calibrated[level][ix].max()]
    if mode=='intervention' and level=='EC_L4':maxima=maxima[raw[maxima,1]==raw[maxima,1].max()]
    chosen[level]=int(maxima.min())
   prediction={'query_id':q}
   for level in LEVELS:
    i=chosen[level];label=rows[i][FIELDS[level]] if i is not None else None
    assert saved[mode][q]['top1_row_'+level]==i and saved[mode][q]['top1_'+level]==label
    prediction['top1_'+level]=label;prediction['top1_row_'+level]=i
    if level in ('EC_L3','EC_L4'):result[mode+'_'+level.lower()]=int(label in endpoints[level])
   final=None;resolution='ABSTAIN'
   for level in reversed(LEVELS):
    i=chosen[level]
    if i is not None and calibrated[level][i]>=threshold[level] and (level=='EC_L3' or calibrated['EC_L3'][i]>=threshold['EC_L3']):final=i;resolution=level;break
   label=rows[final][FIELDS[resolution]] if final is not None else None
   assert (saved[mode][q]['final_row'],saved[mode][q]['final_resolution'],saved[mode][q]['final_label'])==(final,resolution,label)
   prediction.update(final_row=final,final_resolution=resolution,final_label=label);outputs[mode].append(prediction)
   accepted=resolution!='ABSTAIN';evaluable=accepted and (resolution!='EXACT_RHEA' or bool(endpoints[resolution]) and not truth[q]['rhea_ids_unmapped_to_frozen_reference_release'])
   result[mode+'_accepted']=int(accepted);result[mode+'_evaluable']=int(evaluable);result[mode+'_correct']=int(evaluable and label in endpoints[resolution]);result[mode+'_resolution']=resolution
  b,a=result['baseline_ec_l4'],result['intervention_ec_l4']
  result['paired_ec4_state']={(1,1):'both_concordant',(0,1):'rescued',(1,0):'newly_discordant',(0,0):'both_discordant'}[(b,a)]
  results.append(result)
 for mode,data in outputs.items():emit(out/(mode+'_predictions.json'),data)
 write_tsv(out/'paired_results.tsv',results)
 write_tsv(out/'diagnostic_exposure.tsv',exposure)
 original={r['query_id']:r for r in read_tsv(root/'results/temporal_paired_query_results.tsv')}
 for r in results:
  assert r['paired_ec4_state']==original[r['query_id']]['paired_ec4_state']
  for mode in saved:
   for level in ['ec3','ec4']:
    assert r[mode+'_'+level.replace('ec','ec_l')]==int(original[r['query_id']][mode+'_'+level+'_concordant']=='True')
   assert r[mode+'_resolution']==original[r['query_id']][mode+'_final_resolution']
 groups=sorted(set(component.values()));rng=np.random.default_rng(20260819);boot=[]
 for rep in range(2000):
  draw=rng.choice(groups,len(groups),replace=True);sample=[r for c in draw for r in results if r['component_id']==c]
  r={'replicate':rep,'records':len(sample)}
  for level in ['ec3','ec4']:
   key=level.replace('ec','ec_l');b=np.array([s['baseline_'+key] for s in sample],dtype=float);a=np.array([s['intervention_'+key] for s in sample],dtype=float)
   r['baseline_'+level]=float(b.mean());r['intervention_'+level]=float(a.mean());r['delta_'+level]=float((a-b).mean())
  for mode in saved:
   accepted=sum(s[mode+'_accepted'] for s in sample);ev=sum(s[mode+'_evaluable'] for s in sample);correct=sum(s[mode+'_correct'] for s in sample)
   r[mode+'_coverage']=accepted/len(sample);r[mode+'_concordant_coverage']=correct/len(sample);r[mode+'_precision']=correct/ev if ev else float('nan')
  boot.append(r)
 expected=read_tsv(root/'results/component_bootstrap.tsv')
 for a,b in zip(boot,expected):
  for key,v in a.items():assert np.isclose(v,float(b[key]) if b[key] else np.nan,rtol=0,atol=1e-12,equal_nan=True),(key,a['replicate'])
 write_tsv(out/'component_bootstrap.tsv',boot)
 summary=read_json(root/'results/summary.json')
 for level in ['ec3','ec4']:
  key=level.replace('ec','ec_l')
  for mode in saved:assert sum(r[mode+'_'+key] for r in results)==summary['top1'][level][mode+'_concordant']
 report={'status':'PASS','query_records':len(results),'components':len(groups),'prediction_records':sum(map(len,outputs.values())),'candidate_rows':len(rows),'bootstrap_replicates_reproduced':len(boot),'matrix_recomputed':True,'paired_outcomes_reproduced':True,'new_inference_or_model_training':False}
 emit(out/'reproduction_summary.json',report);print(json.dumps(report))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.data,args.output)
