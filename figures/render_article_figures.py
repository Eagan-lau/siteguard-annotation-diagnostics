"""Reproduce the current article figure numbering from supplied source tables."""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    # Source-bundle identifiers are stable even when article figure numbering changes.
    scratch=Path(tempfile.mkdtemp(prefix='siteguard_figures_'))
    base=scratch/'source_bundle_render'
    subprocess.run([sys.executable,str(HERE/'build_figures.py'),'--output-dir',str(base)],check=True)
    subprocess.run([sys.executable,str(HERE/'build_fig1.py'),'--output-dir',str(base)],check=True)
    mapping={'Fig1':'Fig1','Fig2':'Fig2','Fig3':'Fig3','Fig4':'Fig4',
             'Fig5':'Fig6','Fig6':'S12_Fig','Fig7':'S13_Fig'}
    for old,new in mapping.items():
        for ext in ['pdf','svg']:
            shutil.copy2(base/f'{old}.{ext}',out/f'{new}.{ext}')
    subprocess.run([sys.executable,str(HERE/'build_diagnostic_figure.py'),
        '--source-dir',str(HERE/'diagnostic_source'),'--output-dir',str(out)],check=True)
    (out/'figure_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf8')
    print('Current figures:',out)
    print('Intermediate rendered source bundles:',scratch)

if __name__=='__main__':main()
