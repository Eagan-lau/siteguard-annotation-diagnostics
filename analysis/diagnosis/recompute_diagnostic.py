"""Reproduce fixed-prediction support/selection states from portable source tables.

Usage: python recompute_diagnostic.py SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY
No prediction is changed and no model is fitted.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

STATES = ['NO_LIBRARY_SUPPORT', 'ABSENT_FROM_SAVED_UNION', 'OUTSIDE_TOP50',
          'STRICT_SCORE_LOSS', 'CALIBRATED_TIE_LOSS', 'CONCORDANT_TOP1']

def diagnose(candidates, outcomes, selected, library):
    assert not candidates.source_row.duplicated().any()
    references = candidates.groupby(['query_protein_id', 'reference_protein_id'], sort=False).rrf60_score.max().reset_index()
    references['rrf60_score'] = references.rrf60_score.fillna(-np.inf)
    references = references.sort_values(['query_protein_id', 'rrf60_score', 'reference_protein_id'], ascending=[True,False,True], kind='stable')
    references['independent_rrf_rank'] = references.groupby('query_protein_id').cumcount()+1
    frame = candidates.merge(references[['query_protein_id','reference_protein_id','independent_rrf_rank']],
                             on=['query_protein_id','reference_protein_id'],validate='many_to_one')
    assert np.array_equal(frame.rrf_candidate_rank, frame.independent_rrf_rank)
    primary = outcomes[outcomes.method.eq('SITEGUARD_TOP1')].copy()
    winner_map = selected.set_index(['protein_id','level'])
    lib = {lev:set(library[library.level.eq(lev)].label) for lev in ['EC_L3','EC_L4']}
    query_map = {pid:g.sort_values('source_row') for pid,g in frame.groupby('query_protein_id',sort=False)}
    rows=[]
    for r in primary.itertuples():
        truth=set(json.loads(r.truth_labels_set))
        column='ec_l3' if r.level=='EC_L3' else 'ec_l4'
        score='calibrated_'+r.level
        all_rows=query_map[r.protein_id]
        usable=all_rows[all_rows[column].notna()]
        pool=usable[usable.independent_rrf_rank.le(50)]
        assert not pool.empty and truth
        available=pool[column].isin(truth)
        chosen=pool.loc[pool[score].idxmax()]
        frozen=winner_map.loc[(r.protein_id,r.level)]
        assert int(chosen.source_row)==int(frozen.source_row)
        assert chosen.reference_activity_id==frozen.reference_activity_id
        assert chosen[column]==frozen.predicted_label
        assert abs(float(chosen[score])-float(frozen.calibrated_score))<1e-7
        is_correct=chosen[column] in truth
        assert is_correct==bool(r.hit)
        assert json.loads(r.predicted_labels)==[chosen[column]]
        has_library=bool(truth & lib[r.level])
        has_union=bool(usable[column].isin(truth).any())
        has_top50=bool(available.any())
        assert not has_union or has_library
        tied=pool[pool[score].eq(chosen[score])]
        matching_tie=bool(tied[column].isin(truth).any())
        if not has_library: state='NO_LIBRARY_SUPPORT'
        elif not has_union: state='ABSENT_FROM_SAVED_UNION'
        elif not has_top50: state='OUTSIDE_TOP50'
        elif is_correct: state='CONCORDANT_TOP1'
        elif matching_tie: state='CALIBRATED_TIE_LOSS'
        else: state='STRICT_SCORE_LOSS'
        good=pool[available]
        rows.append({'protein_id':r.protein_id,'node_id':r.node_id,'component_id':r.component_id,
          'sequence_length':r.sequence_length,'length_stratum':r.length_stratum,'level':r.level,
          'truth_labels':json.dumps(sorted(truth)), 'predicted_label':chosen[column],
          'state':state, 'library_match':int(has_library),'saved_union_match':int(has_union),
          'top50_match':int(has_top50),'siteguard_concordant':int(is_correct),
          'winner_source_row':int(chosen.source_row), 'winner_reference':chosen.reference_protein_id,
          'winner_activity':chosen.reference_activity_id,'winner_score':float(chosen[score]),
          'best_matching_score':float(good[score].max()) if len(good) else np.nan,
          'top_score_tied_activity_rows':len(tied),'top_score_tied_labels':tied[column].nunique(),
          'top_score_has_matching_label':int(matching_tie),
          'first_matching_rrf_rank':float(usable.loc[usable[column].isin(truth),'independent_rrf_rank'].min()),
          'top50_activity_rows':len(pool),'top50_reference_proteins':pool.reference_protein_id.nunique()})
    out=pd.DataFrame(rows)
    clean=outcomes[outcomes.method.eq('CLEAN_TOP1')][['protein_id','level','hit']].rename(columns={'hit':'clean_concordant'})
    out=out.merge(clean,on=['protein_id','level'],validate='one_to_one')
    assert len(out)==len(primary) and out.groupby(['protein_id','level']).size().eq(1).all()
    groups=[]
    for (level,component),g in out.groupby(['level','component_id']):
        assert g.length_stratum.nunique()==1
        row={'level':level,'component_id':int(component),'length_stratum':g.length_stratum.iloc[0],
             'proteins':len(g),'sequences':g.node_id.nunique(),
             'siteguard_concordant':int(g.siteguard_concordant.sum()),'clean_concordant':int(g.clean_concordant.sum())}
        row.update({s:int(g.state.eq(s).sum()) for s in STATES})
        groups.append(row)
    component=pd.DataFrame(groups)
    aggregates=[]
    for level,g in out.groupby('level'):
        masks={'ALL':np.ones(len(g),bool),'LE_1022':g.length_stratum.eq('LE_1022'),
               'GT_1022':g.length_stratum.eq('GT_1022')}
        largest=g.groupby('component_id').size().idxmax()
        masks['EXCLUDING_LARGEST_COMPONENT']=g.component_id.ne(largest)
        masks['GT_1022_EXCLUDING_LARGEST_COMPONENT']=g.length_stratum.eq('GT_1022')&g.component_id.ne(largest)
        for name,mask in masks.items():
            q=g.loc[mask]
            row={'level':level,'group':name,'proteins':len(q),'components':q.component_id.nunique(),
              'siteguard_concordant':int(q.siteguard_concordant.sum()),'clean_concordant':int(q.clean_concordant.sum()),
              'siteguard_percent':100*q.siteguard_concordant.mean(),'clean_percent':100*q.clean_concordant.mean()}
            row.update({s:int(q.state.eq(s).sum()) for s in STATES})
            assert sum(row[s] for s in STATES)==len(q)
            aggregates.append(row)
    return {'query_diagnosis':out,'component_diagnosis':component,'diagnostic_summary':pd.DataFrame(aggregates)}

def run(source, output):
    assert not output.exists()
    frames=diagnose(pd.read_csv(source/'candidate_records.tsv.gz',sep='\t'),
                    pd.read_csv(source/'per_query_counts.tsv.gz',sep='\t'),
                    pd.read_csv(source/'siteguard_selections.tsv',sep='\t'),
                    pd.read_csv(source/'train_library_labels.tsv',sep='\t'))
    output.mkdir()
    for name,frame in frames.items():frame.to_csv(output/(name+'.tsv'),sep='\t',index=False)
    print(frames['diagnostic_summary'].to_json(orient='records'))
    return frames

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
