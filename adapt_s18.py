"""Adapt S18 source tables to the four model-neutral TSV inputs.

python adapt_s18.py S18/source NEW_INPUT_DIRECTORY
Uses the standard library. No model, predictions or acceptance decisions are fitted.
"""
import csv,gzip,sys
from pathlib import Path
from diagnose import read

def compressed(path):
    with gzip.open(path,'rt',encoding='utf8',newline='') as f:yield from csv.DictReader(f,delimiter='\t')

def write(path,fields,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)

def main(source,out):
    out.mkdir(exist_ok=False)
    queries=[{'query_id':r['protein_id'],'endpoint':r['level'],'truth_labels':r['truth_labels_set']}
             for r in compressed(source/'per_query_counts.tsv.gz') if r['method']=='SITEGUARD_TOP1']
    write(out/'queries.tsv',['query_id','endpoint','truth_labels'],queries)
    write(out/'library.tsv',['endpoint','label'],({'endpoint':r['level'],'label':r['label']} for r in read(source/'train_library_labels.tsv')))
    write(out/'outputs.tsv',['query_id','endpoint','selected_candidate_id','accepted'],
          ({'query_id':r['protein_id'],'endpoint':r['level'],'selected_candidate_id':r['source_row'],'accepted':''}
           for r in read(source/'siteguard_selections.tsv')))
    def candidates():
        for r in compressed(source/'candidate_records.tsv.gz'):
            for level,label in [('EC_L3','ec_l3'),('EC_L4','ec_l4')]:
                if r[label]:yield {'query_id':r['query_protein_id'],'endpoint':level,'candidate_id':r['source_row'],
                                  'label':r[label],'retained':int(float(r['rrf_candidate_rank'])<=50),
                                  'score':r['calibrated_'+level]}
    write(out/'candidates.tsv',['query_id','endpoint','candidate_id','label','retained','score'],candidates())
    print(f'Adapted {len(queries)} query-endpoint records. Acceptance decisions are unavailable in this unthresholded comparison.')

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
