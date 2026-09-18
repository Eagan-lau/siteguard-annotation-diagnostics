"""Set-output counts and paired component bootstrap, independent of model code."""
import numpy as np
import pandas as pd

COUNT_COLUMNS=['n','evaluable','emitted','hit','exact','tp','pred_evaluable','truth_labels','emitted_labels']
METRICS=['coverage','concordant_coverage','query_concordance','exact_set_concordance','micro_precision','micro_recall','micro_f1','mean_emitted_labels']

def counts(predicted,truth):
    p,t=set(predicted),set(truth);e=bool(t);tp=len(p&t) if e else 0
    return [1,int(e),int(bool(p)),int(tp>0),int(e and p==t),tp,len(p) if e else 0,len(t),len(p)]

def ratio(a,b):return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)!=0)

def measures(a):
    n,e,emit,hit,exact,tp,pred,truth,labels=np.moveaxis(np.asarray(a,dtype=float),-1,0)
    return np.stack([ratio(emit,n),ratio(hit,n),ratio(hit,e),ratio(exact,e),ratio(tp,pred),ratio(tp,truth),ratio(2*tp,pred+truth),ratio(labels,n)],axis=-1)

def ci(values):
    v=np.asarray(values);finite=np.isfinite(v)
    if not finite.all():return [np.nan,np.nan,int(finite.sum())]
    lo,hi=np.quantile(v,[.025,.975]);return [float(lo),float(hi),len(v)]

def analyze(table,replicates=2000,seed=20260819):
    summary=[];contrasts=[];draw_frames=[];bootstrap=[];censuses=[]
    for stratum in ['ALL','LE_1022','GT_1022']:
        frame=table if stratum=='ALL' else table.loc[table.length_stratum.eq(stratum)]
        components=sorted(frame.component_id.unique());k=len(components)
        rng=np.random.default_rng(seed)
        draws=rng.integers(0,k,size=(replicates,k))
        multiplicities=np.vstack([np.bincount(row,minlength=k) for row in draws])
        draw_frames.append(pd.DataFrame({'stratum':stratum,'replicate':np.repeat(np.arange(replicates),k),
                           'component_id':np.tile(components,replicates),'multiplicity':multiplicities.ravel()}))
        for weighting in ['protein','sequence']:
            by_method={};point_by_method={}
            for (method,level),group in frame.groupby(['method','level'],sort=True):
                numeric=group[COUNT_COLUMNS].astype(float).mul(group.node_weight if weighting=='sequence' else 1,axis=0)
                numeric['component_id']=group.component_id.to_numpy()
                comp=numeric.groupby('component_id',sort=True)[COUNT_COLUMNS].sum().reindex(components,fill_value=0)
                values=comp.to_numpy();total=values.sum(axis=0);sampled=measures(multiplicities@values);point=measures(total)
                by_method[(method,level)]=sampled;point_by_method[(method,level)]=point
                record={'stratum':stratum,'weighting':weighting,'method':method,'level':level,
                        'protein_records':len(group),'sequence_nodes':group.node_id.nunique(),'components':k}
                record.update({name:float(v) for name,v in zip(COUNT_COLUMNS,total)})
                for j,metric in enumerate(METRICS):
                    lo,hi,finite=ci(sampled[:,j]);record.update({metric:float(point[j]),metric+'_lower95':lo,metric+'_upper95':hi,metric+'_finite_draws':finite})
                summary.append(record)
                b=pd.DataFrame(sampled,columns=METRICS);b.insert(0,'replicate',np.arange(replicates))
                for name,value in [('stratum',stratum),('weighting',weighting),('method',method),('level',level)]:b[name]=value
                bootstrap.append(b)
                c=comp.reset_index()
                for name,value in [('stratum',stratum),('weighting',weighting),('method',method),('level',level)]:c[name]=value
                censuses.append(c)
            for level in ['EC_L3','EC_L4']:
                a=('SITEGUARD_TOP1',level);b=('CLEAN_TOP1',level)
                difference=by_method[a]-by_method[b];point=point_by_method[a]-point_by_method[b]
                r={'stratum':stratum,'weighting':weighting,'level':level,'contrast':'SITEGUARD_TOP1_MINUS_CLEAN_TOP1','components':k}
                for j,metric in enumerate(METRICS):
                    lo,hi,finite=ci(difference[:,j]);r.update({metric:float(point[j]),metric+'_lower95':lo,metric+'_upper95':hi,metric+'_finite_draws':finite})
                contrasts.append(r)
    return {'summary':pd.DataFrame(summary),'paired_contrasts':pd.DataFrame(contrasts),
            'component_draws':pd.concat(draw_frames,ignore_index=True),'bootstrap':pd.concat(bootstrap,ignore_index=True),
            'component_counts':pd.concat(censuses,ignore_index=True)}
