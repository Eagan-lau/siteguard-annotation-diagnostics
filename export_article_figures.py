"""Export the article's vector artwork and supplied main-figure TIFFs."""
import argparse
from pathlib import Path
import shutil

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.data_dir/'article_figures'
    required=[source/'main'/f'Fig{i}.pdf' for i in range(1,7)]
    required += [source/'supplemental'/f'S{i}_Fig.pdf' for i in range(1,15)]
    for file in required:
        if not file.is_file(): raise FileNotFoundError(file)
    a.output.mkdir(parents=True,exist_ok=False)
    for file in required: shutil.copy2(file,a.output/file.name)
    for file in (source/'main').glob('*.tif'):shutil.copy2(file,a.output/file.name)
    print('Exported six main figures and fourteen supporting figures.')

if __name__=='__main__':main()
