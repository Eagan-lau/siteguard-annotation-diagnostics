"""Describe frozen raw/calibrated selections; never choose or deploy a new rule.

Usage: python score_diagnostic.py S18_SOURCE NEW_OUTPUT
Requires NumPy and pandas. All candidate budgets and both EC endpoints are reported.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd

def analyse(source):
    c=pd.read_csv(source/'candidate_records.tsv.gz',sep='\t',float_precision='round_trip')
    o=pd.read_csv(source/'per_query_counts.tsv.gz',sep='\t')
    selected=pd.read_csv(source/'siteguard_selections.tsv',sep='\t').set_index(['protein_id','level'])
    primary=o[o.method.eq('SITEGUARD_TOP1')].set_index(['protein_id','level'])
    assert len(primary)==2232 and len(c)==209070
    refs=c.groupby(['query_protein_id','reference_protein_id']).rrf60_score.max().reset_index()
    refs=refs.sort_values(['query_protein_id','rrf60_score','reference_protein_id'],ascending=[True,False,True],kind='stable')
    refs['check_rank']=refs.groupby('query_protein_id').cumcount()+1
    c=c.merge(refs[['query_protein_id','reference_protein_id','check_rank']],validate='many_to_one')
    assert np.array_equal(c.check_rank,c.rrf_candidate_rank)
    rows=[]
    for pid,whole in c.groupby('query_protein_id',sort=True):
        whole=whole.sort_values('source_row')
        for level,col in [('EC_L3','ec_l3'),('EC_L4','ec_l4')]:
            target=primary.loc[(pid,level)];truth=set(json.loads(target.truth_labels_set))
            complete=whole[whole[col].notna()]
            for budget in ['10','25','50','ALL']:
                pool=complete if budget=='ALL' else complete[complete.check_rank.le(int(budget))]
                assert len(pool)>0
                raw=pool['raw_'+level].to_numpy();cal=pool['calibrated_'+level].to_numpy()
                assert np.isfinite(raw).all() and np.isfinite(cal).all()
                rawi=int(np.argmax(raw));cali=int(np.argmax(cal))
                # Independent index-of-first-maximum reconstruction.
                assert pool.index[rawi]==pool['raw_'+level].idxmax()
                assert pool.index[cali]==pool['calibrated_'+level].idxmax()
                rawtop=raw==raw.max();caltop=cal==cal.max();labels=pool[col].to_numpy()
                match=np.array([label in truth for label in labels])
                order=np.argsort(raw,kind='stable');monotone=int(np.sum(np.diff(cal[order])<0))
                assert monotone==0,(pid,level,budget)
                # Equal raw inputs must have equal frozen calibrated outputs.
                assert pool.groupby('raw_'+level)['calibrated_'+level].nunique().max()==1
                assert np.all(caltop[rawtop])
                rw=pool.iloc[rawi];cw=pool.iloc[cali]
                if budget=='50':
                    frozen=selected.loc[(pid,level)]
                    assert int(cw.source_row)==int(frozen.source_row)
                    assert cw[col]==frozen.predicted_label
                    assert bool(match[cali])==bool(target.hit)
                row={'protein_id':pid,'level':level,'budget':budget,'component_id':target.component_id,
                     'length_stratum':target.length_stratum,'candidate_rows':len(pool),
                     'candidate_labels':len(set(labels)),'matching_candidate':int(match.any()),
                     'raw_top_rows':int(rawtop.sum()),'raw_top_labels':len(set(labels[rawtop])),
                     'calibrated_top_rows':int(caltop.sum()),'calibrated_top_labels':len(set(labels[caltop])),
                     'raw_levels_in_calibrated_top':len(set(raw[caltop])),
                     'top_set_expanded':int(caltop.sum()>rawtop.sum()),
                     'raw_winner_row':int(rw.source_row),'calibrated_winner_row':int(cw.source_row),
                     'raw_winner_label':rw[col],'calibrated_winner_label':cw[col],
                     'winner_row_changed':int(rw.source_row!=cw.source_row),
                     'winner_label_changed':int(rw[col]!=cw[col]),
                     'raw_concordant':int(match[rawi]),'calibrated_concordant':int(match[cali]),
                     'raw_only_concordant':int(match[rawi] and not match[cali]),
                     'calibrated_only_concordant':int(match[cali] and not match[rawi]),
                     'calibrated_tie_loss':int(not match[cali] and (match & caltop).any()),
                     'raw_tie_loss':int(not match[rawi] and (match & rawtop).any()),
                     'monotonicity_violations':monotone}
                rows.append(row)
    per=pd.DataFrame(rows)
    assert len(per)==2232*4
    summaries=[]
    for (level,budget),g in per.groupby(['level','budget'],sort=False):
        sums=['matching_candidate','top_set_expanded','winner_row_changed','winner_label_changed',
              'raw_concordant','calibrated_concordant','raw_only_concordant','calibrated_only_concordant',
              'calibrated_tie_loss','raw_tie_loss','monotonicity_violations']
        r={'level':level,'budget':budget,'queries':len(g),'components':g.component_id.nunique()}
        r.update({s:int(g[s].sum()) for s in sums})
        r.update({'raw_multilabel_top':int(g.raw_top_labels.gt(1).sum()),
                  'calibrated_multilabel_top':int(g.calibrated_top_labels.gt(1).sum()),
                  'calibrated_tie_loss_raw_concordant':int((g.calibrated_tie_loss.eq(1)&g.raw_concordant.eq(1)).sum()),
                  'calibrated_tie_loss_raw_discordant':int((g.calibrated_tie_loss.eq(1)&g.raw_concordant.eq(0)).sum()),
                  'calibrated_tie_loss_expanded_top':int((g.calibrated_tie_loss.eq(1)&g.top_set_expanded.eq(1)).sum()),
                  'calibrated_tie_loss_unchanged_top':int((g.calibrated_tie_loss.eq(1)&g.top_set_expanded.eq(0)).sum())})
        assert r['calibrated_concordant']-r['raw_concordant']==r['calibrated_only_concordant']-r['raw_only_concordant']
        summaries.append(r)
    return per,pd.DataFrame(summaries)

def main():
    source=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists()
    per,summary=analyse(source);out.mkdir()
    per.to_csv(out/'per_query_scores.tsv',sep='\t',index=False)
    summary.to_csv(out/'score_summary.tsv',sep='\t',index=False)
    print(summary.to_json(orient='records',indent=2))

if __name__=='__main__':main()
