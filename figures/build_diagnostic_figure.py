"""Render EC-L4 diagnosis from saved per-query records; no model inference."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,
    'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'savefig.facecolor':'white','text.color':'#243642','axes.labelcolor':'#243642',
    'xtick.color':'#536370','ytick.color':'#243642'})
COLORS=['#7C8990','#86ADC3','#E3B266','#BD7251','#81749F']
STATES=['NO_LIBRARY_SUPPORT','ABSENT_FROM_SAVED_UNION','OUTSIDE_TOP50','STRICT_SCORE_LOSS','CALIBRATED_TIE_LOSS']

def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def values(source):
    q=[r for r in read(source/'query_diagnosis.tsv') if r['level']=='EC_L4']
    s=[r for r in read(source/'per_query_scores.tsv') if r['level']=='EC_L4' and r['budget']=='50']
    assert len(q)==len(s)==1116
    assert len({r['protein_id'] for r in q})==1116
    qmap={r['protein_id']:r for r in q}
    assert {r['protein_id'] for r in s}==set(qmap)
    for r in s:assert int(r['calibrated_concordant'])==int(qmap[r['protein_id']]['siteguard_concordant'])
    counts=Counter(r['state'] for r in q)
    failure=[counts[k] for k in STATES]
    assert failure==[67,17,1,59,60] and counts['CONCORDANT_TOP1']==912
    comps=[]
    for ident,n in [('335',36),('716',91)]:
        rows=[r for r in q if r['component_id']==ident]
        assert len(rows)==n
        c={k:sum(int(r[k]) for r in rows) for k in ['library_match','top50_match','siteguard_concordant','clean_concordant']}
        c.update({'component':ident,'n':n,'states':dict(Counter(r['state'] for r in rows))})
        comps.append(c)
    assert [list(c[k] for k in ['library_match','top50_match','siteguard_concordant','clean_concordant']) for c in comps]==[[0,0,0,36],[91,91,8,91]]
    paired=Counter((int(r['raw_concordant']),int(r['calibrated_concordant'])) for r in s)
    matrix=[[paired[(1,1)],paired[(1,0)]],[paired[(0,1)],paired[(0,0)]]]
    assert matrix==[[902,13],[10,191]]
    top=[sum(int(r[k])>1 for r in s) for k in ['raw_top_labels','calibrated_top_labels']]
    assert top==[1,97]
    tie=[r for r in s if int(r['calibrated_tie_loss'])]
    assert len(tie)==60 and sum(int(r['raw_concordant']) for r in tie)==13
    return {'queries':1116,'components':len({r['component_id'] for r in q}),
        'disagreements':204,'concordant':912,'states':dict(zip(STATES,failure)),
        'component_examples':comps,'paired_concordance_matrix':matrix,
        'multilabel_top_set_queries':dict(zip(['raw','calibrated'],top)),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.tsv')}}

def header(fig,y,letter,title,subtitle):
    fig.text(.035,y,letter,weight='bold',fontsize=12,va='top')
    fig.text(.075,y,title,weight='bold',fontsize=10,va='top')
    fig.text(.075,y-.029,subtitle,fontsize=8.5,color='#536370',va='top')

def render(source,out):
    v=values(source);out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(7.5,8.3))
    header(fig,.975,'A','EC-L4 disagreements occur at distinct stages',
        '1,116 screened records: 912 concordant selections and 204 disagreements')
    ax=fig.add_axes([.075,.810,.88,.08]);left=0
    for count,color in zip(v['states'].values(),COLORS):
        ax.barh(0,count,left=left,height=.55,color=color,edgecolor='white',linewidth=.7)
        if count>=10:ax.text(left+count/2,0,str(count),ha='center',va='center',fontsize=10,color='white' if color in [COLORS[0],COLORS[3],COLORS[4]] else '#243642')
        else:ax.annotate('1',xy=(left+count/2,-.275),xytext=(left+count/2,-.85),ha='center',va='top',fontsize=8.5,arrowprops={'arrowstyle':'-','color':'#536370','lw':.6})
        left+=count
    ax.set_xlim(0,204);ax.set_ylim(-1.1,.45);ax.axis('off')
    labels=['No TRAIN\nlibrary support','Absent from\nsaved union','Outside\nTop-50','Strict-score\nselection loss','Top-score-tie\nselection loss']
    for i,(label,col) in enumerate(zip(labels,COLORS)):
        x=.075+i*.177
        fig.add_artist(plt.Rectangle((x,.767),.012,.012,transform=fig.transFigure,facecolor=col,edgecolor='none'))
        fig.text(x+.018,.782,label,fontsize=8,va='top',linespacing=1.25)
    header(fig,.707,'B','Component examples identify different intervention targets',
        'Two post hoc illustrations; bars show fractions of each component')
    for j,c in enumerate(v['component_examples']):
        ax=fig.add_axes([.215+j*.435,.408,.29,.205])
        names=['TRAIN support','Top-50 support','SiteGuard','CLEAN']
        nums=[c[k] for k in ['library_match','top50_match','siteguard_concordant','clean_concordant']]
        y=np.arange(4)
        ax.barh(y,[100*n/c['n'] for n in nums],height=.55,color=['#C2D1D8','#90B1C2','#357CA1','#243642'])
        for yy,n in zip(y,nums):
            x=100*n/c['n'];ax.text(x-2 if x>55 else x+2,yy,f'{n}/{c["n"]}',ha='right' if x>55 else 'left',va='center',fontsize=8.5,color='white' if yy>=2 and x>55 else '#243642')
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks([0,50,100],['0','50','100%'])
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
        ax.set_title(f'Component {c["component"]}  (n = {c["n"]})',pad=9,fontsize=9)
    fig.text(.215,.371,'36 disagreements: no library support',fontsize=8)
    fig.text(.650,.371,'83 disagreements: 44 strict, 39 tied',fontsize=8)
    header(fig,.313,'C','Tied-score failures and score-only gains are different quantities',
        'Same 1,116 records and Top-50 candidates; original row-order ties retained')
    ax=fig.add_axes([.145,.087,.255,.12])
    ax.barh([0,1],[1,97],color=['#C2D1D8','#81749F'],height=.55)
    for y,n in enumerate([1,97]):ax.text(n+2,y,str(n),va='center',fontsize=9)
    ax.set_yticks([0,1],['Raw','Calibrated']);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xticks([0,50,100]);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Queries with multiple labels\ntied at the highest score',fontsize=8.5)
    ax=fig.add_axes([.665,.075,.245,.17])
    cmap=LinearSegmentedColormap.from_list('counts',['#F3F5F7','#357CA1'])
    mat=np.array(v['paired_concordance_matrix'])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=cmap(mat[i,j]/902),edgecolor='white',linewidth=2))
    ax.set_xlim(-.5,1.5);ax.set_ylim(1.5,-.5)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=11,color='white' if mat[i,j]>450 else '#243642')
    ax.set_xticks([0,1],['Concordant','Discordant'],fontsize=8)
    ax.set_yticks([0,1],['Concordant','Discordant'],fontsize=8)
    ax.xaxis.tick_top();ax.xaxis.set_label_position('top');ax.set_xlabel('Calibrated-score selection',labelpad=8,fontsize=8.5)
    ax.set_ylabel('Raw-score selection',fontsize=8.5,labelpad=6)
    ax.tick_params(length=0);ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for ext in ['pdf','svg']:fig.savefig(out/f'Fig5.{ext}',metadata={'Creator':'Matplotlib'})
    fig.savefig(out/'Fig5.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(out/'Fig5_preview.png',dpi=140)
    im=Image.open(out/'Fig5.tif')
    if im.mode!='RGB':im.convert('RGB').save(out/'Fig5.tif',dpi=(600,600),compression='tiff_lzw')
    (out/'Fig5_values.json').write_text(json.dumps(v,indent=2),encoding='utf8')
    plt.close(fig)
    return v

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(render(a.source_dir,a.output_dir),indent=2))
