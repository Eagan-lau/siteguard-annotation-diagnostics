"""Native workflow diagnosis, using measured query records without refitting."""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]/'data/native_workflows'
OUT=Path('figure_output')
COL={'NO_LIBRARY_SUPPORT':'#A6AFB6','ABSENT_FROM_RETURNED_CANDIDATES':'#D5A347','RETENTION_LOSS':'#688FAB','SELECTION_LOSS':'#80659F'}
def main():
    rows=list(csv.DictReader((ROOT/'results/native_query_diagnostics.tsv').open(),delimiter='\t'))
    rows=[r for r in rows if r['level']=='EC_L4' and r['method'] in ['CLEAN_MAXSEP','DIAMOND_TOP_HIT_SET']]
    assert len(rows)==2232
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.6,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'svg.fonttype':'none','pdf.fonttype':42,'text.color':'#233541','axes.labelcolor':'#233541','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.5,7.0),facecolor='white')
    fig.text(.035,.971,'A',fontsize=12,weight='bold');fig.text(.083,.971,'Native workflow boundaries',fontsize=10,weight='bold')
    a=fig.add_axes([.12,.71,.85,.21]);a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    for y,label,texts,color in [(.77,'CLEAN',['5,242 EC centres\nAll distances scored','10 nearest centres\nCalling input','Maximum separation\nLabel set'],'#267A91'),(.24,'DIAMOND',['136,293 reference proteins\nFixed sequence search','All returned hits\nRetained for selection','Top-hit reference\nAll associated labels'],'#A56D39')]:
        a.text(-.10,y,label,ha='left',va='center',weight='bold',color=color)
        for x,t in zip([.26,.59,.91],texts):
            a.scatter([x],[y+.115],s=28,facecolor='white',edgecolor=color,linewidth=1.2,zorder=3)
            a.text(x,y-.02,t,ha='center',va='top',linespacing=1.5,fontsize=8)
        for x1,x2 in [(.28,.57),(.61,.89)]:a.annotate('',xy=(x2,y+.115),xytext=(x1,y+.115),arrowprops={'arrowstyle':'->','lw':.8,'color':color})
    b=fig.add_axes([.12,.46,.85,.17]);b.set_xlim(0,200);b.set_ylim(-.65,1.65)
    fig.text(.035,.661,'B',fontsize=12,weight='bold');fig.text(.083,.661,'Where EC-L4 disagreements occur',fontsize=10,weight='bold')
    sources=[]
    for y,m,l in [(1,'CLEAN_MAXSEP','CLEAN'),(0,'DIAMOND_TOP_HIT_SET','DIAMOND')]:
        sub=[r for r in rows if r['method']==m];left=0
        for st in COL:
            n=sum(r['state']==st for r in sub);sources.append({'panel':'B','method':m,'state':st,'records':n,'denominator':1116})
            if n:
                b.barh(y,n,left=left,height=.40,color=COL[st],edgecolor='white',linewidth=.8)
                b.text(left+n/2,y,str(n),ha='center',va='center',fontsize=8,weight='bold',color='white' if st in ['RETENTION_LOSS','SELECTION_LOSS'] else '#233541')
            left+=n
        assert left=={'CLEAN_MAXSEP':61,'DIAMOND_TOP_HIT_SET':178}[m]
        b.text(left+3,y,str(left),va='center',fontsize=8,weight='bold')
    b.set_yticks([1,0],['CLEAN','DIAMOND']);b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    b.set_xlabel('Discordant query records (common cohort: 1,116)');b.set_xticks([0,50,100,150,200]);b.grid(axis='x',lw=.4,color='#E5E9EC');b.set_axisbelow(True)
    labels=['No library label','No returned match','Outside calling input','Selection loss']
    handles=[Rectangle((0,0),1,1,facecolor=COL[k]) for k in COL]
    fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.075,.38),ncol=4,frameon=False,fontsize=7.1,handlelength=1.1,columnspacing=1.3)
    fig.text(.035,.337,'C',fontsize=12,weight='bold');fig.text(.083,.337,'Ranks of available matches in discordant outputs',fontsize=10,weight='bold')
    for j,m in enumerate(['CLEAN_MAXSEP','DIAMOND_TOP_HIT_SET']):
        ax=fig.add_axes([.12 if j==0 else .63,.085,.34,.20]);sub=[r for r in rows if r['method']==m and r['state']!='CONCORDANT_OUTPUT' and r['returned_match']=='1']
        ranks=np.sort([int(float(r['first_matching_rank'])) for r in sub]);assert len(ranks)==[50,88][j]
        ax.step(np.r_[1,ranks,max(ranks)*1.1],np.r_[0,np.arange(1,len(ranks)+1)/len(ranks)*100,100],where='post',lw=1.6,color=['#267A91','#A56D39'][j])
        ax.set_xscale('log');ax.set_ylim(0,105);ax.set_yticks([0,50,100]);ax.set_xlim(1,max(ranks)*1.15)
        ax.set_xlabel(['First matching EC-centre rank','First matching protein-hit rank'][j]);ax.set_title(['CLEAN: 50 discordant records','DIAMOND: 88 discordant records'][j],loc='left',fontsize=8.3,pad=10)
        threshold=[10,50][j];ax.axvline(threshold,color='#66737A',ls='--',lw=.8)
        ax.text(threshold,8,['Calling limit = 10','Rank 50 (no truncation)'][j],ha='left',va='bottom',rotation=90,fontsize=7,color='#52636D')
        ax.grid(axis='y',lw=.4,color='#E5E9EC');ax.set_axisbelow(True)
        if j==0:
            ax.set_ylabel('Cumulative fraction (%)')
        for r in sub:sources.append({'panel':'C','method':m,'state':r['state'],'protein_id':r['protein_id'],'first_matching_rank':r['first_matching_rank'],'denominator':len(sub)})
    d=OUT/'figures';d.mkdir(parents=True,exist_ok=True)
    fig.savefig(d/'Fig2.pdf');fig.savefig(d/'Fig2.svg');fig.savefig(d/'Fig2.png',dpi=180)
    fig.savefig(d/'Fig2.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
    sd=OUT/'figure_source_data';sd.mkdir(exist_ok=True)
    keys=['panel','method','state','records','denominator','protein_id','first_matching_rank']
    with (sd/'Fig2_source.tsv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(sources)
    print(json.dumps({'figure':str(d/'Fig2.pdf'),'rows':len(rows),'rank_subset_counts':[50,88],'scope':'EC-L4; 1116 records; 43 components'}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT)
    ROOT=parser.parse_args().data_dir.resolve()
    main()
