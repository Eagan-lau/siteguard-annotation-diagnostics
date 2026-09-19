"""Render the article's conceptual graphical abstract as vector PDF/SVG and TIFF."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

INK='#203642'; MUTED='#62737B'; TEAL='#16847E'; RUST='#C56E4B'; PALE='#EDF4F4'; GRID='#DCE5E7'

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(5.5,5.5),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,396),ylim=(396,0));ax.axis('off')
    def text(x,y,s,size=12,weight='normal',color=INK,ha='left'):
        return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va='center',linespacing=1.15)
    def arrow(a,b,color=MUTED,width=1.15):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=width,color=color))
    def dot(x,y,match=False,r=4.0,empty=False):
        ax.add_patch(Circle((x,y),r,facecolor='white' if empty else TEAL if match else RUST,edgecolor=TEAL if match else RUST,linewidth=1.2))
    text(20,27,'Diagnosing enzyme annotation',16,'bold')
    # A schematic enzyme and recorded reaction give biological context.
    verts=[(25,71),(21,46),(58,42),(59,65),(59,83),(34,96),(31,72),(29,57),(48,49),(65,60),(79,70),(67,91),(53,78)]
    codes=[MPath.MOVETO]+[MPath.CURVE4]*12
    ax.add_patch(PathPatch(MPath(verts,codes),fill=False,edgecolor=INK,lw=2.2,capstyle='round'))
    text(20,107,'Enzyme query',12)
    arrow((91,73),(124,73))
    dot(138,73,True,6);arrow((150,73),(176,73),TEAL);dot(190,73,True,6)
    text(127,107,'Recorded activity',12)
    ax.plot([232,232],[51,115],color=GRID,lw=1)
    text(246,64,'CLEAN · DIAMOND',12,'bold')
    text(246,83,'SiteGuard',12,'bold')
    text(246,106,'Workflow records',12,color=MUTED)
    # Each row depicts a distinct query-level failure, not a time series.
    text(157,139,'Library',12,'bold',ha='center')
    text(246,139,'Candidates',12,'bold',ha='center')
    text(345,139,'Output',12,'bold',ha='center')
    for y,label,lib,can in [(174,'Reference\ngap',False,False),(219,'Candidate\nloss',True,False),(264,'Selection\nloss',True,True)]:
        text(20,y,label,12,'bold')
        ax.add_patch(Rectangle((133,y-15),49,30,facecolor=PALE,edgecolor='none'))
        dot(146,y-5,lib);dot(169,y-5);dot(146,y+7);dot(169,y+7)
        arrow((188,y),(216,y))
        ax.add_patch(Rectangle((221,y-15),49,30,facecolor=PALE,edgecolor='none'))
        dot(233,y,can);dot(257,y)
        arrow((278,y),(323,y))
        dot(346,y,False,7)
        if y==219:
            ax.plot([204,204],[y-8,y+8],color=TEAL,lw=2)
        if y==264:
            ax.plot([333,359],[y+15,y+15],color=RUST,lw=1.5)
    dot(138,299,True,3.5);text(147,299,'Match',12)
    dot(229,299,False,3.5);text(238,299,'Other activity',12)
    ax.plot([20,376],[316,316],color=GRID,lw=1)
    text(20,338,'Target a change',13,'bold')
    arrow((147,338),(181,338))
    text(192,338,'Test paired outcomes',13,'bold')
    # Bidirectional changes explicitly distinguish exposure from repair benefit.
    dot(213,370,False,4);arrow((222,370),(242,370),TEAL);dot(251,370,True,4)
    dot(293,370,True,4);arrow((302,370),(322,370),RUST);dot(331,370,False,4)
    text(20,371,'Measure gains and costs',12,color=MUTED)
    for ext in ('pdf','svg','png','tiff'):
        kwargs={'dpi':300,'facecolor':'white'}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(out/f'Graphical_Abstract.{ext}',**kwargs)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('graphical_abstract'));main(p.parse_args().output)
