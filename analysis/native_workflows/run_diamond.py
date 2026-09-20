"""Rerun the fixed DIAMOND demonstration in a new directory."""
import argparse,json,subprocess
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diamond',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();exe=args.diamond.resolve();source=args.source.resolve();out=args.output.resolve()
    version=subprocess.check_output([str(exe),'version'],text=True).strip()
    assert version=='diamond version 2.1.8',version
    assert (source/'references.fasta').is_file() and (source/'queries.fasta').is_file()
    out.mkdir(exist_ok=False)
    commands=[[str(exe),'makedb','--in',str(source/'references.fasta'),'--db',str(out/'reference'),'--threads','8'],
        [str(exe),'blastp','--query',str(source/'queries.fasta'),'--db',str(out/'reference'),'--out',str(out/'hits.tsv'),
        '--outfmt','6','qseqid','sseqid','pident','length','qstart','qend','sstart','send','evalue','bitscore','qlen','slen',
        '--very-sensitive','--evalue','0.001','--max-target-seqs','0','--max-hsps','1','--threads','8','--block-size','0.5','--index-chunks','4']]
    with (out/'configuration.json').open('x') as f:json.dump({'version':version,'commands':commands},f,indent=2)
    for command in commands:subprocess.run(command,check=True,cwd=out)
if __name__=='__main__':main()
