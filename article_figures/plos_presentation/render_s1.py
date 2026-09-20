"""Render the historical partitioning schematic from its documented counts."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
INK='#233541'; BLUE='#306B93'; MUTED='#596874'; ORANGE='#C57545'
def save_s1(out, values):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED})
    f=plt.figure(figsize=(7.5,6.65),facecolor='white')
    f.text(.04,.967,'A',fontsize=12,weight='bold')
    f.text(.085,.967,'Historical protein split before pair construction',fontsize=10,weight='bold')
    f.text(.085,.927,f"{values['historical_split']['total_protein_records']:,} proteins assigned by historical sequence clusters",fontsize=8)
    a=f.add_axes([.18,.72,.75,.17]); a.set_xlim(0,160000)
    vals=[values['historical_split'][r] for r in ['training','validation','test']]
    assert sum(vals)==values['historical_split']['total_protein_records']
    a.barh([2,1,0],vals,height=.53,color=[BLUE,'#7DA4BB',ORANGE])
    a.set_yticks([2,1,0],['Training','Validation','Test'])
    a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
    a.set_xticks([0,50000,100000,150000],['0','50,000','100,000','150,000'])
    a.set_xlabel('Protein records');a.grid(axis='x',lw=.4,color='#E4E8EC');a.set_axisbelow(True)
    for y,n in zip([2,1,0],vals):
        a.text(n+2500,y,f'{n:,} ({100*n/sum(vals):.1f}%)',va='center',fontsize=8)
    a.set_xlim(0,188000)
    f.text(.085,.642,'The historical cluster split is distinct from the later sequence-isolated reconstruction.',fontsize=8)
    f.text(.04,.592,'B',fontsize=12,weight='bold')
    f.text(.085,.592,'Prediction and evaluation use separate information',fontsize=10,weight='bold')
    b=f.add_axes([.085,.355,.86,.19]);b.axis('off')
    b.add_patch(Rectangle((0,.03),.40,.94,facecolor='#EEF3F7',edgecolor='none'))
    b.add_patch(Rectangle((.60,.03),.40,.94,facecolor='#F7F0EB',edgecolor='none'))
    b.text(.025,.81,'Prediction inputs',weight='bold',fontsize=9,color=BLUE)
    b.text(.625,.81,'Evaluation labels',weight='bold',fontsize=9,color=ORANGE)
    b.text(.025,.60,'Sequence evidence\nStructural evidence\nTool confidence and fixed covariates',va='top',linespacing=1.8)
    b.text(.625,.60,'Query EC annotations\nQuery Rhea annotations\nConcordance with documented activity',va='top',linespacing=1.8)
    b.annotate('',xy=(.59,.47),xytext=(.41,.47),arrowprops={'arrowstyle':'->','lw':1,'color':MUTED})
    b.text(.50,.71,'Join after\npredictions\nare fixed',ha='center',va='center',fontsize=7.8,linespacing=1.4)
    f.text(.04,.297,'C',fontsize=12,weight='bold')
    f.text(.085,.297,'Descriptive and optimization pairs have different roles',fontsize=10,weight='bold')
    c=f.add_axes([.31,.135,.63,.105]);c.set_xlim(0,1600000)
    c.barh([1,0],[values['pair_populations'][r] for r in ['descriptive','optimization']],height=.45,color=[BLUE,'#7DA4BB'])
    c.set_yticks([1,0],['Descriptive population','Optimization sample'])
    c.tick_params(axis='y',length=0);c.spines['left'].set_visible(False)
    c.set_xticks([0,500000,1000000,1500000],['0','0.5','1.0','1.5']);c.set_xlabel('Pair records (million)')
    for y,n in zip([1,0],[values['pair_populations'][r] for r in ['descriptive','optimization']]):c.text(n-30000,y,f'{n:,}',ha='right',va='center',color='white',weight='bold')
    f.text(.085,.043,f"Difficult-case universe: {values['pair_populations']['difficult_case']:,} pairs     Matched comparison: {values['pair_populations']['matched_per_arm']:,} pairs per arm",fontsize=8)
    for ext in ['pdf','svg'] : f.savefig(out/f'Figure_S1.{ext}')
    plt.close(f)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'inputs/s1_layout_values.json')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    save_s1(args.output_dir,json.loads(args.input.read_text(encoding='utf8')))
