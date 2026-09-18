"""Recompute S17 Table from protein-level label sets, without models or GPUs."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    assert not args.output_dir.exists(),'Choose a new output directory'
    spec=importlib.util.spec_from_file_location('metrics_core',Path(__file__).with_name('metrics_core.py'))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    table=pd.read_csv(args.input_dir/'per_query_counts.tsv.gz',sep='\t')
    for col in ['predicted_labels','truth_labels_set']:table[col]=table[col].map(json.loads)
    assert len(table)==8928 and not table.duplicated(['protein_id','level','method']).any()
    original=table[core.COUNT_COLUMNS].to_numpy()
    recomputed=np.array([core.counts(r.predicted_labels,r.truth_labels_set) for r in table.itertuples()])
    assert np.array_equal(original,recomputed)
    result=core.analyze(table,replicates=2000,seed=20260819)
    for name in ['summary','paired_contrasts','component_counts']:
        expected=pd.read_csv(args.input_dir/(name+'.tsv'),sep='\t')
        actual=result[name]
        assert actual.shape==expected.shape and list(actual.columns)==list(expected.columns)
        for col in actual:
            if pd.api.types.is_numeric_dtype(actual[col]):
                np.testing.assert_allclose(actual[col],expected[col],rtol=1e-10,atol=1e-10,equal_nan=True)
            else:assert actual[col].tolist()==expected[col].tolist()
    args.output_dir.mkdir(parents=True)
    for name,frame in result.items():frame.to_csv(args.output_dir/(name+'.tsv.gz'),sep='\t',index=False)
    print('All protein counts, aggregate results and paired intervals reproduce within 1e-10 tolerance.')

if __name__=='__main__':main()
