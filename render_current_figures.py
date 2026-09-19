"""Render the six current main figures, S12-S14 and the graphical abstract."""
from pathlib import Path
import argparse,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=p.parse_args();out=args.output.resolve();data=args.data_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='siteguard_render_') as t:
        tmp=Path(t);source=tmp/'source_figures'
        subprocess.run([sys.executable,str(ROOT/'article_figures/render_article_figures.py'),'--output-dir',str(source)],cwd=tmp,check=True)
        for old,new in [('Fig1','Fig1'),('Fig2','Fig4'),('Fig3','Fig5'),('Fig4','Fig6'),('Fig5','Fig3'),('Fig6','Figure_S14'),('S12_Fig','Figure_S12'),('S13_Fig','Figure_S13')]:
            for ext in ['pdf','svg','png','tif','tiff']:
                f=source/f'{old}.{ext}'
                if f.exists():shutil.copy2(f,out/f'{new}.{ext}')
        subprocess.run([sys.executable,str(ROOT/'analysis/native_workflows/render_figure2.py'),'--data-dir',str(data/'native_workflows')],cwd=tmp,check=True)
        for f in (tmp/'figure_output/figures').glob('Fig2.*'):shutil.copy2(f,out/f.name)
        subprocess.run([sys.executable,str(ROOT/'article_figures/build_graphical_abstract.py'),'--output',str(out)],cwd=tmp,check=True)
    assert all((out/f'Fig{i}.pdf').is_file() for i in range(1,7))
    print('Rendered current article figures and graphical abstract.')

if __name__=='__main__':main()
