"""Render Figure 1 from the accompanying numerical source.

Run: python render_figure1.py --output-dir fig1_output
Dependencies: matplotlib, numpy, Pillow. All geometry is at final print size.
"""
from pathlib import Path
from io import BytesIO
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MP
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_data/Fig1/numerical_source.json'
W, H = 540, 620
INK='#253742'; MUTED='#5F6F7A'; LINE='#BCC7CD'; PALE='#EDF1F3'
BLUE='#306B93'; PURPLE='#635F9A'; TEAL='#227F79'; ORANGE='#BC7048'
TIER=['#306B93','#92B5C9','#D9E0E5']

plt.rcParams.update({'font.family':'Arial','font.size':8,'text.color':INK,
                    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                    'axes.unicode_minus':False,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
TEXT=[]

def txt(x,y,s,size=8,bold=False,color=INK,ha='left',va='center',rotation=0):
    a=ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
              color=color,ha=ha,va=va,rotation=rotation,linespacing=1.3,zorder=10)
    TEXT.append(a); return a

def line(x1,y1,x2,y2,c=LINE,lw=.75,ls='-',z=2):
    return ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=z)[0]

def box(x,y,w,h,fc='white',ec=LINE,lw=.75,rounding=0,z=3):
    patch=(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',
                        facecolor=fc,edgecolor=ec,lw=lw,zorder=z) if rounding else
           Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,lw=lw,zorder=z))
    ax.add_patch(patch); return patch

def dot(x,y,r=2.2,c=BLUE,fill=True,lw=.8):
    p=Circle((x,y),r,facecolor=c if fill else 'white',edgecolor=c,lw=lw,zorder=5)
    ax.add_patch(p);return p

def arrow(x1,y1,x2,y2,c=MUTED,lw=.9,ls='-',style='-|>'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style,color=c,lw=lw,linestyle=ls,
                                shrinkA=0,shrinkB=0,mutation_scale=7),zorder=4)

def curve(x1,y1,x2,y2,c=LINE,lw=.8):
    mid=(x1+x2)/2
    p=MP([(x1,y1),(mid,y1),(mid,y2),(x2,y2)],
         [MP.MOVETO,MP.CURVE4,MP.CURVE4,MP.CURVE4])
    ax.add_patch(PathPatch(p,facecolor='none',edgecolor=c,lw=lw,zorder=1))

def panel(letter,title,x,y):
    txt(x,y,letter,12,True);txt(x+20,y,title,10,True)

def glyph(kind,x,y,c=BLUE):
    if kind=='seq':
        for i,h in enumerate([2,5,3,6,3,5]):
            line(x+i*2.4,y-h/2,x+i*2.4,y+h/2,c,1.6)
    elif kind=='reaction':
        dot(x+1,y,2,c,False);arrow(x+5,y,x+12,y,c,.7);dot(x+16,y,2,c,False)
    elif kind=='chemical':
        angles=np.arange(7)*np.pi/3
        xx=x+8+5*np.cos(angles);yy=y+5*np.sin(angles)
        ax.plot(xx,yy,c=c,lw=.8)
    elif kind=='domain':
        line(x,y,x+17,y,c,.7)
        box(x+1,y-3,5,6,c,c);box(x+10,y-3,5,6,'white',c)
    elif kind=='site':
        for i,(dx,dy) in enumerate([(0,3),(4,-2),(8,2),(13,-3),(17,1)]):
            dot(x+dx,y+dy,2,c,i==2)
    elif kind=='structure':
        xs=np.array([0,2,5,9,11,8,4,7,12,16]);ys=np.array([1,-4,-2,3,0,-5,-3,5,4,-1])
        ax.plot(x+xs,y+ys,c=c,lw=1)

def panel_a(data):
    panel('A','Integrated evidence and annotation provenance',18,21)
    rows=[('Swiss-Prot','Protein + evidence','seq',BLUE),
          ('Rhea / ENZYME','Reaction + EC','reaction',BLUE),
          ('ChEBI','Chemical participants','chemical',BLUE),
          ('Pfam / CATH','Global context','domain',MUTED),
          ('M-CSA / UniProt sites','Catalytic residues','site',MUTED),
          ('PDB / SIFTS / AFDB','Structure mapping','structure',MUTED)]
    for i,(name,role,kind,c) in enumerate(rows):
        y=48+i*22
        glyph(kind,22,y,c);txt(48,y-3,name,8.1,True)
        txt(48,y+7,role,8,color=MUTED)
        curve(181,y,230,99,c='#B5C4CD',lw=.65)
    arrow(230,99,243,99,BLUE)
    box(246,58,109,82,fc='#F5F8FA',ec='#90AABC',rounding=3)
    txt(300.5,73,'Activity record',9,True,ha='center')
    line(256,85,345,85,'#C8D4DC',.65)
    for j,(label,c) in enumerate([('Protein',BLUE),('EC / reaction',TEAL),('Mapped context',MUTED)]):
        dot(258,97+15*j,2,c);txt(267,97+15*j,label,8)
    arrow(359,99,376,99,BLUE)
    e=data['evidence'];txt(384,46,f'{e["denominator"]:,}',14,True)
    txt(384,62,'activity records',8.5)
    for i,(key,label,c) in enumerate(zip(['gold','silver','other_or_unattributed'],['Gold','Silver','Other'],TIER)):
        y=82+31*i;val=e[key];pct=100*val/e['denominator']
        txt(384,y,f'{label}  {val:,}',8.2,True)
        txt(522,y,f'{pct:.1f}%',8.2,ha='right',color=MUTED)
        box(384,y+8,138,5,PALE,PALE,0)
        box(384,y+8,138*pct/100,5,c,c,0)
    txt(384,170,'Provenance, not accuracy',8,color=MUTED)
    line(18,185,522,185,PALE,.9)

def panel_b():
    panel('B','Functional resolution',18,203)
    # Prefix depth, with actual branching rather than a one-to-one EC/Rhea arrow.
    ys=[234,260,286,314]
    for i,y in enumerate(ys):
        c=MUTED if i<2 else [BLUE,PURPLE][i-2]
        if i<3:arrow(56,y+8,56,ys[i+1]-8,LINE,.7)
        for j in range(4):
            x=26+j*15
            box(x,y-7,12,14,c if j<=i else 'white',c,.6)
            txt(x+6,y,'abcd'[j] if j<=i else '-',8,color='white' if j<=i else c,ha='center')
        txt(95,y,f'EC-L{i+1}',8.6,i>=2,c)
    # A labelled bipartite crosswalk. Nodes are symbolic EC entries / reactions,
    # not actual accessions; undirected edges denote associations, not prediction.
    txt(162,232,'EC-L4',8.5,True,PURPLE,ha='center')
    txt(237,232,'Exact Rhea',8.8,True,TEAL,ha='center')
    line(136,245,136,331,PALE,.7)
    for y1,y2 in [(269,254),(269,289),(308,289),(308,324)]:
        curve(180,y1,211,y2,c='#8EAAA6',lw=1.05)
    for label,y in [('EC i',269),('EC j',308)]:
        box(144,y-10,36,20,fc='#F1F0F7',ec=PURPLE,lw=.85,rounding=2)
        txt(162,y,label,8.5,True,PURPLE,ha='center')
    for label,y in [('Reaction 1',254),('Reaction 2',289),('Reaction 3',324)]:
        box(211,y-10,52,20,fc='#F0F7F5',ec=TEAL,lw=.85,rounding=2)
        txt(237,y,label,8,color=TEAL,ha='center')
    txt(203,344,'Many-to-many (schematic)',8,color=MUTED,ha='center')
    txt(26,359,'EC-L3 and EC-L4 are depths, not EC classes.',8,color=MUTED)
    # Small underlines distinguish the three evaluated endpoints without a legend.
    line(95,297,131,297,BLUE,1.3);line(95,325,131,325,PURPLE,1.3)
    line(212,241,262,241,TEAL,1.3)

def panel_c(data):
    panel('C','Historical split and audit',283,203)
    c=data['cohort'];s=data['historical_sequence_comparison']
    txt(283,226,f'{c["eligible_proteins"]:,} eligible proteins',10,True)
    txt(283,241,f'{c["clusters"]:,} clusters; split before pairs',8,color=MUTED)
    start=283
    for key,col in zip(['train','validation','test'],[BLUE,'#91B2C5',MUTED]):
        w=239*c[key]/c['eligible_proteins'];box(start,253,w,11,col,'white',.5);start+=w
    for x,key,label,col in zip([283,377,459],['train','validation','test'],['Train','Validation','Test'],[BLUE,'#91B2C5',MUTED]):
        box(x,271,7,7,col,col,0)
        txt(x+12,275,label,8.2,True);txt(x+12,289,f'{c[key]:,}',8.2)
    txt(283,311,'Detected TRAIN neighbours',8.5,True)
    # Explicit quantitative ticks. Denominator is the named partition, not sequence identity.
    x0,x1=329,437;axis_y=356
    for i,(key,label) in enumerate([('validation','Val.'),('test','Test')]):
        r=s[key];y=328+i*15;value=100*r['with_training_neighbour']/r['queries']
        txt(283,y,label,8)
        box(x0,y-3.5,x1-x0,7,PALE,PALE,0)
        box(x0,y-3.5,(x1-x0)*value/30,7,BLUE,BLUE,0)
        txt(446,y,f'{value:.2f}%',8.2,True)
        txt(486,y,f'{r["exact_overlaps"]} exact',8,color=MUTED)
    for tick in [0,10,20,30]:
        xx=x0+(x1-x0)*tick/30
        line(xx,axis_y-4,xx,axis_y-1,LINE,.5);txt(xx,axis_y+4,str(tick),8,color=MUTED,ha='center')
    txt(383,374,'% of queries',8,color=MUTED,ha='center')
    txt(283,387,'>30% identity; both coverages \u226570%',8,color=MUTED)
    line(18,398,522,398,PALE,.9)

def candidate_grid(cx,cy,sampled=False):
    # Schematic 4x3 array, identical geometry in both arms; not empirical counts.
    for r in range(3):
        for col in range(4):
            retained=(r,col) in [(0,0),(1,2),(2,3)]
            color=BLUE if not sampled or retained else '#DCE3E8'
            dot(cx+7*col,cy+7*r,1.8,color,True,.5)

def panel_d(data):
    panel('D','Candidate processing and retrospective diagnosis',18,411)
    c=data['cohort'];txt(522,411,f'{c["common_queries"]:,} common queries',8.3,True,ha='right')
    # Fixed shared inputs, two candidate policies, same prediction function/rule.
    txt(22,437,'Top-50 per search',8.5,True)
    txt(22,451,'MMseqs2 + Foldseek',8,color=MUTED)
    candidate_grid(46,468)
    txt(22,498,'Union + activity',8,color=MUTED);txt(22,510,'expansion',8,color=MUTED)
    line(75,475,105,475,BLUE);line(105,454,105,500,BLUE)
    arrow(105,454,134,454,BLUE);arrow(105,500,134,500,BLUE)
    candidate_grid(147,447,True);candidate_grid(147,493,False)
    txt(183,454,'Sampled',8.5,True);txt(183,500,'Complete',8.5,True)
    txt(278,435,'Fixed scorer + rule',8.5,True,ha='center')
    # Shared styling and positional alignment encode the locked intervention.
    box(237,444,87,70,'#F4F7F9','#CFD9E0',.65,3)
    for y in [454,500]:
        arrow(229,y,249,y,BLUE)
        dot(260,y,8,BLUE,False);txt(260,y,'f',10,True,BLUE,ha='center')
        arrow(269,y,286,y,BLUE)
        dot(299,y,8,BLUE,False);txt(299,y,'T',9,True,BLUE,ha='center')
        arrow(308,y,343,y,BLUE)
        box(349,y-6,11,12,'white',BLUE,.9)
        line(352,y-2,357,y-2,BLUE,.7);line(352,y+2,357,y+2,BLUE,.7)
    txt(354,477,'Outputs',8,ha='center',color=MUTED)
    line(365,454,390,454,BLUE);line(365,500,390,500,BLUE)
    line(390,454,390,500,BLUE);arrow(390,477,412,477,BLUE)
    txt(463,473,'Paired outcomes',8.6,True,ha='center')
    txt(463,489,'Concordance + coverage',8,color=MUTED,ha='center')
    txt(463,445,'Recorded query labels',8,color=TEAL,ha='center')
    arrow(463,453,463,461,TEAL,.8,'--')
    txt(18,532,'Sampled-set availability checks',8.7,True)
    txt(522,532,'First failed check assigns the state',8,color=MUTED,ha='right')
    positions=[53,159,265,371,487]
    for i,(x,label,fail) in enumerate(zip(positions[:-1],['Annotation','Library','Retrieval','Scored set'],['Not evaluable','No reference','Retrieval miss','Sampling loss'])):
        txt(x,551,label,8.3,True,ha='center')
        dot(x,568,3,BLUE,False)
        arrow(x+5,568,positions[i+1]-8,568,BLUE,.8)
        arrow(x,573,x,587,MUTED,.75)
        box(x-39,591,78,17,'#F0F3F5','none',0,2)
        txt(x,599,fail,8,ha='center',color=MUTED)
    dot(487,568,4,TEAL)
    txt(487,587,'Match retained',8.3,True,TEAL,ha='center')
    # Test membership accounts for the complete common cohort; retain omitted count in caption.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    out=p.parse_args().output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    d=json.loads(SOURCE.read_text(encoding='utf8'))
    e,c=d['evidence'],d['cohort']
    assert e['gold']+e['silver']+e['other_or_unattributed']==e['denominator']
    assert c['train']+c['validation']+c['test']==c['eligible_proteins']
    assert c['common_queries']+c['test_outside_common']==c['test']
    for name,row in d['historical_sequence_comparison'].items():
        if isinstance(row,dict):assert row['exact_overlaps']<=row['with_training_neighbour']<=row['queries']
    panel_a(d);panel_b();panel_c(d);panel_d(d)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in TEXT:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds)
    dest=out/'Fig1'
    fig.savefig(dest.with_suffix('.pdf'),metadata={'Title':'Resource integration and query-level diagnosis of enzyme annotation','Author':'Yugeng Liu et al.'})
    fig.savefig(dest.with_suffix('.svg'))
    fig.savefig(dest.with_suffix('.png'),dpi=180)
    buf=BytesIO();fig.savefig(buf,format='png',dpi=300);buf.seek(0)
    with Image.open(buf) as im:im.convert('RGB').save(dest.with_suffix('.tif'),dpi=(300,300),compression='tiff_lzw')
    plt.close(fig)

if __name__=='__main__':main()
