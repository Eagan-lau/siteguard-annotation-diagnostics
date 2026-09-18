"""Render Fig 1 at 7.5 by 8.67 inches without reducing label size."""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
BLUE, TEAL, GRAY, INK, LIGHT = '#306B93', '#227F79', '#596874', '#243543', '#E2E7EB'
plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data=json.loads((HERE/'Fig1/numerical_source.json').read_text())
    evidence,cohort=data['evidence'],data['cohort']
    assert sum(evidence[k] for k in ['gold','silver','other_or_unattributed'])==evidence['denominator']
    assert sum(cohort[k] for k in ['train','validation','test'])==cohort['eligible_proteins']
    fig=plt.figure(figsize=(7.5,624/72));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,540),ylim=(624,0));ax.axis('off')
    def text(s,x,y,size=8,bold=False,color=INK,ha='left'):
        return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',color=color,ha=ha,va='baseline')
    def line(x1,y1,x2,y2,color=GRAY,lw=.7):ax.plot([x1,x2],[y1,y2],color=color,lw=lw)
    def box(x,y,w,h,color):ax.add_patch(Rectangle((x,y),w,h,color=color,linewidth=0))
    def dot(x,y,color=BLUE,r=2.5):ax.plot(x,y,'o',ms=r*2,color=color,markeredgewidth=0)
    def arrow(x1,y1,x2,y2,color=GRAY):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':color,'lw':.8,'shrinkA':0,'shrinkB':0,'mutation_scale':7})
    def panel(letter,title,x,y):text(letter,x,y,12,True);text(title,x+19,y,10,True)
    panel('A','Resources linked to catalytic activity records',18,23)
    resources=[('Swiss-Prot','Protein + evidence'),('Rhea / ENZYME','Reaction + EC'),('ChEBI','Chemical participants'),('Pfam / CATH','Global context'),('M-CSA / UniProt sites','Catalytic residues'),('PDB / SIFTS / AFDB','Mapped structures')]
    for i,(name,role) in enumerate(resources):
        y=46+i*17;color=BLUE if i<3 else GRAY
        dot(23,y-3,color);text(name,32,y,8.2,i<2);line(145,y-3,164,y-3,color);text(role,175,y,8.2,color=GRAY)
    text(f'{evidence["denominator"]:,}',329,48,12,True);text('activity records',329,62,8.5)
    start=329
    for key,label,color in [('gold','Gold',BLUE),('silver','Silver','#92B5C9'),('other_or_unattributed','Other',LIGHT)]:
        width=190*evidence[key]/evidence['denominator'];box(start,74,width,17,color);start+=width
    for i,(key,label,color) in enumerate([('gold','Gold',BLUE),('silver','Silver','#92B5C9'),('other_or_unattributed','Other',LIGHT)]):
        y=107+i*15;box(329,y-7,7,7,color);text(f'{label}  {100*evidence[key]/evidence["denominator"]:.1f}%',341,y)
    text('Provenance tiers',418,137,8,color=GRAY)
    panel('B','Documented resolution',18,165)
    for j,(label,pattern) in enumerate(zip(['EC-L1','EC-L2','EC-L3','EC-L4'],['a.-.-.-','a.b.-.-','a.b.c.-','a.b.c.d'])):
        y=194+j*23;color=BLUE if j>=2 else GRAY
        for k in range(4):
            ax.add_patch(Rectangle((22+k*11,y-8),8,8,facecolor=color if k<=j else 'white',edgecolor=color,lw=.5))
        text(label,77,y,8.3,j>=2,color);text(pattern,132,y,8.3,color=color)
    arrow(181,260,207,260,TEAL);text('Exact Rhea',213,257,8.5,True,TEAL);text('cross-map',213,271)
    text('EC-L3 and EC-L4: annotation depths',22,299,8,color=GRAY)
    text('EC and Rhea: many-to-many mapping',22,315,8,color=GRAY)
    text('Measured endpoints',22,347,8.5,True)
    for j,(label,color) in enumerate([('EC-L3',BLUE),('EC-L4','#635F9A'),('Exact Rhea',TEAL)]):
        dot(26+85*j,363,color,2.5);text(label,34+85*j,366,8.2,color=color)
    panel('C','Historical split and sequence audit',301,165)
    text(f'{cohort["eligible_proteins"]:,} proteins',301,187,9,True)
    text('8,401 clusters; nominal 30% setting',301,202)
    start=301
    for key,color in [('train',BLUE),('validation','#92B5C9'),('test',GRAY)]:
        width=218*cohort[key]/cohort['eligible_proteins'];box(start,213,width,12,color);start+=width
    for i,(key,label,color) in enumerate([('train','Train',BLUE),('validation','Validation','#92B5C9'),('test','Test',GRAY)]):
        y=243+i*15;box(301,y-7,7,7,color);text(f'{label}  {cohort[key]:,}',313,y)
    text('Queries with TRAIN neighbour (%)',301,293,8.5,True)
    sequence=data['historical_sequence_comparison']
    for i,(label,key) in enumerate([('Val.','validation'),('Test','test')]):
        row=sequence[key];num,denom,exact=row['with_training_neighbour'],row['queries'],row['exact_overlaps']
        y=310+i*17;value=100*num/denom
        text(label,301,y);box(326,y-7,81,7,LIGHT);box(326,y-7,81*value/30,7,BLUE)
        text(f'{value:.2f}%',414,y);text(f'{exact} exact',468,y,color=GRAY)
    text('Bar scale: 0–30% of queries',301,342,8,color=GRAY)
    text('Match: >30% identity; both coverages ≥70%',301,357,8,color=GRAY)
    text('27,639 scored; 32 omitted by sampling',301,377,8.5,True)
    panel('D','Paired candidate processing and evaluation',18,401)
    for row in range(4):
        for col in range(4):dot(28+col*8,435+row*8,BLUE,2)
    text('Top-50 union',20,481,8.4,True);text('MMseqs2 + Foldseek',20,495)
    line(66,446,94,446);line(94,427,94,479)
    for y,sampled,label in [(427,True,'Historical subset'),(479,False,'Complete set')]:
        arrow(94,y,115,y)
        for row in range(2):
            for col in range(4):dot(126+col*8,y-4+row*8,BLUE if not sampled or (row+col)%3==0 else LIGHT,2)
        text(label,116,y+20)
        arrow(159,y,201,y,BLUE);text('f',215,y+3,12,True,BLUE)
        arrow(226,y,258,y,BLUE);text('T',270,y+3,11,True,BLUE);arrow(282,y,366,y,BLUE)
    text('Same model + rule',202,456,8.3,True)
    line(366,427,366,479,BLUE);arrow(366,453,392,453,BLUE)
    text('Fixed predictions',402,442,8.5,True);text('+ evaluation labels',402,459,8.2,color=TEAL)
    text('Query labels excluded from prediction inputs',202,508,8,color=GRAY)
    checks=[('Labels','Not evaluable'),('Library','No reference'),('Retrieval','Retrieval miss'),('Scoring','Sampling loss')]
    for i,(top,bottom) in enumerate(checks):
        x=55+i*108;dot(x,556,BLUE,3.2);text(top,x,540,8.5,True,ha='center')
        arrow(x+7,556,x+100,556,BLUE);arrow(x,564,x,590)
        text(bottom,x,607,8,ha='center',color=GRAY)
    dot(496,556,BLUE,3.2);text('Match',496,582,8.5,True,BLUE,ha='center');text('retained',496,597,8.5,False,BLUE,ha='center')
    fig.canvas.draw()
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_text():
            bounds=label.get_window_extent(fig.canvas.get_renderer());canvas=fig.bbox
            assert bounds.x0>=0 and bounds.x1<=canvas.width and bounds.y0>=0 and bounds.y1<=canvas.height,label.get_text()
    for ext in ['pdf','svg']:fig.savefig(out/f'Fig1.{ext}')
    fig.savefig(out/'Fig1.png',dpi=160);plt.close(fig)
    print(json.dumps({'figure':1,'width_in':7.5,'height_in':624/72,'font_min_pt':8,'output':str(out)}))


if __name__=='__main__':main()
