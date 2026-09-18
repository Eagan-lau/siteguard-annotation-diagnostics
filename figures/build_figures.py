"""Render enzyme-annotation Figures 2–7 from supplied numerical sources.

Usage: python build_figures.py --output-dir figures_output
Requires NumPy, Matplotlib and pypdf (or the bundled distribution).
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
P = argparse.ArgumentParser(description=__doc__)
P.add_argument('--source-public-code', type=Path)
P.add_argument('--output-dir', type=Path, required=True)
args = P.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
DATA = OUT / 'source_data'
DATA.mkdir()
LEVELS = ['EC_L3', 'EC_L4', 'EXACT_RHEA']
LABELS = ['EC-L3', 'EC-L4', 'Exact Rhea']
COLORS = ['#306B93', '#635F9A', '#227F79']
INK, MUTED, GRID = '#243543', '#596874', '#E2E7EB'
BLUE, ORANGE, LIGHT = COLORS[0], '#C57545', '#F0F3F5'
CHECKS, PLOTTED, INPUTS = [], [], {}

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'axes.edgecolor': MUTED,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'legend.frameon': False, 'legend.fontsize': 7.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'hatch.linewidth': .55,
})

def check(condition, label):
    if not condition:
        raise AssertionError(label)
    CHECKS.append(label)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stage_sources():
    if args.source_public_code:
        source = args.source_public_code.resolve()
        required = {
            2:['phase444_partition.tsv','phase444_availability.tsv','phase444_paired_cluster_bootstrap.tsv'],
            3:['phase445_metrics.tsv','paired_transition_sampling_loss_outcomes.tsv','paired_transition_paired_bootstrap.tsv'],
            5:['native_states.tsv','native_metrics.tsv','native_output_set_sizes.tsv'],
            6:['plotted_values.json','Figure3_matched_analysis.tsv'],
            7:['evidencejudge_full_precision_coverage_curves.tsv','evidencejudge_paired_utility_comparisons.tsv'],
        }
        for old, new in [(2,2),(3,3),(6,5),(4,6),(5,7)]:
            src = source / 'figures' / 'historical' / f'Fig{old}' / 'source_data'
            dest = DATA / f'Fig{new}'
            dest.mkdir()
            for name in required[new]: shutil.copy2(src/name,dest/name)
        seq = source / 'source_data' / 'sequence_rebuild'
        dest = DATA / 'Fig4'
        dest.mkdir()
        for name in ['Fig4A_role_counts.tsv', 'Fig4B_budget_diagnostics.tsv',
                     'Fig4C_primary_endpoints.tsv', 'Fig4D_final_resolution_counts.tsv',
                     'retest_cluster_bootstrap.tsv']:
            shutil.copy2(seq / name, dest / name)
        vend = source / 'vendor' / 'pypdf-6.10.0-purepy.zip'
        if vend.exists():
            shutil.copy2(vend, OUT / vend.name)
        for p in DATA.rglob('*'):
            if not p.is_file():
                continue
            rel = p.relative_to(DATA)
            cur = int(rel.parts[0][3:])
            old = {2:2,3:3,5:6,6:4,7:5}.get(cur)
            original = (seq / rel.name if cur == 4 else
                        source / 'figures' / 'historical' / f'Fig{old}' / 'source_data' / rel.name)
            check(sha(p) == sha(original), 'Input copy SHA-256: ' + str(rel))
    else:
        for p in (HERE / 'source_data').iterdir():
            shutil.copytree(p, DATA / p.name)
        vend = HERE / 'pypdf-6.10.0-purepy.zip'
        if vend.exists(): shutil.copy2(vend, OUT / vend.name)
        manifest = json.loads((HERE / 'input_manifest.json').read_text())
        for rel, meta in manifest.items():
            check(sha(DATA / rel) == meta['sha256'], 'Portable source identity: ' + rel)
    for p in sorted(DATA.rglob('*')):
        if p.is_file():
            INPUTS[p.relative_to(DATA).as_posix()] = {'bytes':p.stat().st_size,'sha256':sha(p)}
    for name in ['build_figures.py','DESIGN.md','FIGURE_LEGENDS.md','README.md']:
        if (HERE / name).exists(): shutil.copy2(HERE / name, OUT / name)

def table(fig, name):
    path = DATA / f'Fig{fig}' / name
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def one(rows, **match):
    found = [r for r in rows if all(r[k] == str(v) for k,v in match.items())]
    check(len(found) == 1, 'Unique source row: ' + repr(match))
    return found[0]

def record(fig, panel, row, **extra):
    PLOTTED.append(dict(figure=fig, panel=panel, source_row=row, **extra))

def newfig(h):
    # Print scale: 183 mm wide. All font sizes are final printed point sizes.
    return plt.figure(figsize=(183/25.4, h), facecolor='white')

def heading(fig, letter, title, y, x=.025):
    fig.text(x, y, letter, fontsize=12, fontweight='bold', va='top')
    fig.text(x+.04, y-.001, title, fontsize=10, fontweight='bold', va='top')

def axes(fig, bounds, grid='y'):
    ax=fig.add_axes(bounds)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.55)
        ax.set_axisbelow(True)
    return ax

def export(fig, number):
    for label in fig.findobj(matplotlib.text.Text):
        label.set_fontfamily('Arial')
        label.set_fontsize(max(8, label.get_fontsize()))
    fig.savefig(OUT / f'Fig{number}.pdf')
    fig.savefig(OUT / f'Fig{number}.svg')
    fig.savefig(OUT / f'Fig{number}.png', dpi=190)
    plt.close(fig)

def fig2():
    part=table(2,'phase444_partition.tsv')
    av=table(2,'phase444_availability.tsv')
    ci=table(2,'phase444_paired_cluster_bootstrap.tsv')
    f=newfig(7.2)
    heading(f,'A','Where documented candidate support is lost',.982)
    state_names=['No truth','No reference','Retrieval miss','Sampling loss','Retained']
    states=['NO_DOCUMENTED_TRUTH','NO_TRAIN_LIBRARY_SUPPORT','LIBRARY_SUPPORTED_UNION50_MISS',
            'UNION50_POSITIVE_SCORED_SUBSET_MISS','SCORED_SUBSET_POSITIVE']
    fills=['#DCE1E5','#7A8791','#C6D8E4',ORANGE,BLUE]
    f.legend([Patch(facecolor=c) for c in fills], state_names,
             loc='upper center',bbox_to_anchor=(.51,.942),ncol=5,columnspacing=1.1,handlelength=1)
    ax=axes(f,[.15,.735,.81,.16],grid=None)
    for j, lev in enumerate(LEVELS):
        left=0; total=0
        for k,(key,co) in enumerate(zip(states,fills)):
            r=one(part,level=lev,state=key); n=int(r['queries']); total+=n
            width=n/27639*100
            ax.barh(2-j,width,left=left,height=.58,color=co,edgecolor='white',linewidth=.6)
            if width>9:
                ax.text(left+width/2,2-j,f'{width:.1f}%',ha='center',va='center',fontsize=8,
                        color='white' if k in [1,3,4] else INK)
            left+=width;record(2,'A',r)
        check(total==27639,'Fig2 state closure '+lev)
    ax.set_yticks([2,1,0],LABELS);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Queries (%)  |  N = 27,639 at each resolution')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    heading(f,'B','Retrieval depth and the historical scoring subset',.65)
    for j,lev in enumerate(LEVELS):
        ax=axes(f,[.09+j*.305,.375,.25,.18])
        vals=[]
        for k in [10,20,50,100]:
            r=one(av,level=lev,stage=f'union_at_{k}',denominator='all_phase12_queries')
            vals.append(float(r['availability'])*100);record(2,'B',r)
        ax.plot([10,20,50,100],vals,color=COLORS[j],marker=['o','s','^'][j],ms=3.5,lw=1.5)
        for stage,style,color in [('library','--',MUTED),('scored_subset',':',ORANGE)]:
            r=one(av,level=lev,stage=stage,denominator='all_phase12_queries')
            v=100*float(r['availability'])
            ax.axhline(v,color=color,ls=style,lw=1)
            record(2,'B',r)
        ax.set(xlim=(0,105),ylim=(0,105),xticks=[10,50,100],yticks=[0,25,50,75,100])
        ax.set_xlabel('Top-k union');ax.set_title(LABELS[j],color=COLORS[j],fontweight='bold')
        if j==0: ax.set_ylabel('Matching candidate available (%)')
        else: ax.set_yticklabels([])
    f.legend([Line2D([0],[0],color=INK,lw=1.5),Line2D([0],[0],color=MUTED,ls='--'),
              Line2D([0],[0],color=ORANGE,ls=':')],['Retrieval union','TRAIN library','Scored subset'],
              loc='upper center',bbox_to_anchor=(.53,.626),ncol=3,handlelength=2)
    heading(f,'C','Restoring scoring support versus extending retrieval',.29)
    ax=axes(f,[.33,.095,.62,.137],grid=None)
    values=np.zeros((2,3)); refs=[]
    for i,key in enumerate(['union50_minus_scored','union100_minus_union50']):
        rr=[]
        for j,lev in enumerate(LEVELS):
            r=one(ci,level=lev,contrast=key);rr.append(r)
            values[i,j]=100*float(r['difference_fraction'])
            check(np.isclose(int(r['additional_positive_queries'])/27639*100,values[i,j]),'Fig2 paired delta '+lev+key)
            record(2,'C',r)
        refs.append(rr)
    cmap=LinearSegmentedColormap.from_list('gain',['#F3F6F8',BLUE])
    ax.pcolormesh(np.arange(4),np.arange(3),values,cmap=cmap,vmin=0,vmax=40,
                  edgecolors='white',linewidth=2)
    ax.set(xlim=(0,3),ylim=(2,0));ax.set_xticks(np.arange(3)+.5,LABELS)
    ax.xaxis.tick_top();ax.set_yticks([.5,1.5],['Top-50 minus scored','Top-100 minus Top-50'])
    ax.tick_params(length=0);[s.set_visible(False) for s in ax.spines.values()]
    for i in range(2):
        for j in range(3):
            r=refs[i][j];v=values[i,j];lo=100*float(r['ci95_low']);hi=100*float(r['ci95_high'])
            ax.text(j+.5,i+.47,f'{v:+.2f}',ha='center',va='center',fontsize=10,fontweight='bold',
                    color='white' if v>20 else INK)
            ax.text(j+.5,i+.78,f'[{lo:.2f}, {hi:.2f}]',ha='center',va='center',fontsize=7,
                    color='white' if v>20 else INK)
    f.text(.33,.052,'Gain in percentage points [95% paired-cluster interval]',fontsize=7.5)
    f.text(.33,.027,'2,000 saved resamples; 1,220 query clusters',fontsize=7,color=MUTED)
    export(f,2)

def ribbon(ax,x0,y0,x1,y1,h,color,alpha=.3):
    dx=(x1-x0)*.42
    p=MPath([(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),
             (x1,y1+h),(x1-dx,y1+h),(x0+dx,y0+h),(x0,y0+h),(x0,y0)],
            [MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
             MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY])
    ax.add_patch(PathPatch(p,facecolor=color,edgecolor='none',alpha=alpha))

def fig3():
    metrics=table(3,'phase445_metrics.tsv'); ci=table(3,'paired_transition_paired_bootstrap.tsv')
    outcomes=table(3,'paired_transition_sampling_loss_outcomes.tsv')
    f=newfig(8.2)
    heading(f,'A','Coverage and precision under the unchanged policy',.982)
    f.legend([Line2D([0],[0],color=MUTED,marker='o',mfc='white',ls='none'),
              Line2D([0],[0],color=MUTED,marker='o',ls='none')],['Sampled','Complete'],
             loc='upper right',bbox_to_anchor=(.98,.95),ncol=2)
    for j,lev in enumerate(LEVELS):
        ax=axes(f,[.095+j*.305,.705,.248,.19],grid='both')
        a=one(metrics,level=lev,cohort='sampled',validation_target='0.95')
        b=one(metrics,level=lev,cohort='full',validation_target='0.95')
        for r in [a,b]:
            check(int(r['queries'])==27639,'Fig3 common denominator '+lev+r['cohort'])
            for key,den,num in [('coverage',27639,int(r['accepted'])),
                                 ('precision',int(r['accepted']),int(r['accepted_correct'])),
                                 ('accepted_correct_fraction',27639,int(r['accepted_correct']))]:
                check(np.isclose(float(r[key]),num/den),'Fig3 '+key+lev+r['cohort'])
            record(3,'A',r)
        xy=[(100*float(r['coverage']),100*float(r['precision'])) for r in [a,b]]
        ax.annotate('',xy=xy[1],xytext=xy[0],arrowprops={'arrowstyle':'->','color':COLORS[j],'lw':1.7})
        for k,(x,y) in enumerate(xy):
            ax.plot(x,y,'o',color=COLORS[j],mfc='white' if k==0 else COLORS[j],ms=5)
            off=(4,6) if k==0 else (4,-16)
            if j==0: off=(3,6) if k==0 else (-4,-15)
            ax.annotate(f'{x:.2f}, {y:.2f}',(x,y),xytext=off,textcoords='offset points',
                        ha='right' if k==1 and j==0 else 'left',fontsize=6.8,color=COLORS[j])
        ax.set(xlim=(0,62),ylim=(75,100),xticks=[0,20,40,60],yticks=[75,85,95,100])
        ax.set_xlabel('Accepted coverage (%)')
        if j==0:ax.set_ylabel('Accepted precision (%)')
        else:ax.set_yticklabels([])
        ax.set_title(LABELS[j],fontweight='bold',color=COLORS[j])
        f.text(.095+j*.305,.646,
               f'Concordant coverage: {100*float(a["accepted_correct_fraction"]):.2f} → '
               f'{100*float(b["accepted_correct_fraction"]):.2f}%',fontsize=7)
    heading(f,'B','What happened to the restored exact-Rhea queries?',.602)
    ax=axes(f,[.055,.31,.9,.24],grid=None);ax.axis('off');ax.set(xlim=(0,1),ylim=(1,0))
    keys=['TOP1_CONCORDANT_ACCEPTED','TOP1_CONCORDANT_REJECTED',
          'CANDIDATE_PRESENT_TOP1_DISAGREEMENT_ACCEPTED','CANDIDATE_PRESENT_TOP1_DISAGREEMENT_REJECTED']
    rr=[one(outcomes,level='EXACT_RHEA',target='0.95',full_state=k) for k in keys]
    ns=[int(r['queries']) for r in rr]
    check(ns==[2327,4126,98,2531] and sum(ns)==9082,'Fig3 joint restored cohort closure')
    scale=.72/9082; heights=[n*scale for n in ns]; x0,x1,x2=.06,.41,.74;w=.012
    root=.16; ym=[.09,.69];yr=[.08,.30,.69,.755]
    match=ns[0]+ns[1]
    ribbon(ax,x0+w,root,x1,ym[0],match*scale,BLUE)
    ribbon(ax,x0+w,root+match*scale,x1,ym[1],(9082-match)*scale,ORANGE)
    ax.add_patch(Rectangle((x0,root),w,.72,color=MUTED))
    for j,total in enumerate([match,9082-match]):
        ax.add_patch(Rectangle((x1,ym[j]),w,total*scale,color=[BLUE,ORANGE][j]))
    for i,(r,n,h) in enumerate(zip(rr,ns,heights)):
        col=BLUE if i<2 else ORANGE
        start=ym[i//2]+(0 if i%2==0 else heights[i-1])
        ribbon(ax,x1+w,start,x2,yr[i],h,col,.42 if i%2==0 else .18)
        ax.add_patch(Rectangle((x2,yr[i]),w,h,facecolor=col if i%2==0 else 'white',edgecolor=col,lw=.7))
        ax.text(x2+.028,yr[i]+h/2,('Accepted' if i%2==0 else 'Not accepted')+f'\n{n:,} ({n/9082*100:.1f}%)',
                va='center',fontsize=7.5,color=INK)
        record(3,'B',r,derived_height=h)
    ax.text(x0,.07,'9,082 queries',ha='center',fontsize=8,fontweight='bold')
    ax.text(x1,.03,'Top-1 concordant: 6,453',ha='center',fontsize=8,color=BLUE)
    ax.text(x1,.65,'Top-1 disagreement: 2,629',ha='center',fontsize=8,color=ORANGE)
    heading(f,'C','Paired benefit and cost of complete candidate scoring',.268)
    specs=[('accepted_concordant_fraction_delta','Concordant accepted / all'),
           ('accepted_disagreement_fraction_delta','Discordant accepted / all'),
           ('accepted_precision_delta','Accepted precision')]
    for j,(key,label) in enumerate(specs):
        ax=axes(f,[.095+j*.305,.075,.248,.137])
        for i,lev in enumerate(LEVELS):
            r=one(ci,level=lev,target='0.95',scope='ALL_QUERIES',metric=key)
            v,lo,hi=[100*float(r[k]) for k in ['estimate','ci95_low','ci95_high']]
            check(int(r['replicates'])==2000 and int(r['cohort_clusters'])==1220,'Fig3 saved bootstrap metadata')
            ax.bar(i,v,color=COLORS[i],width=.55,alpha=.85)
            ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor=INK,lw=.8,capsize=2)
            ax.text(i,hi+1.5 if v>=0 else lo-1.5,f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=7)
            record(3,'C',r)
        ax.axhline(0,color=MUTED,lw=.7);ax.set(ylim=(-20,40),xticks=range(3),xticklabels=['L3','L4','Rhea'])
        ax.set_title(label,fontsize=8)
        if j==0:ax.set_ylabel('Difference (pp)')
        else:ax.set_yticklabels([])
    f.text(.095,.017,'Error bars: pointwise 95% intervals, 2,000 paired resamples of 1,220 query clusters.',fontsize=7)
    export(f,3)

def fig4():
    roles=table(4,'Fig4A_role_counts.tsv');budget=table(4,'Fig4B_budget_diagnostics.tsv')
    endpoints=table(4,'Fig4C_primary_endpoints.tsv');resolution=table(4,'Fig4D_final_resolution_counts.tsv')
    boot=table(4,'retest_cluster_bootstrap.tsv')
    check(len(boot)==2000,'Fig4 exactly 2,000 saved bootstrap estimates')
    f=newfig(8.2)
    heading(f,'A','Whole sequence components assigned to five data roles',.982)
    ax=axes(f,[.065,.844,.90,.086],grid=None);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    total=sum(int(r['proteins']) for r in roles); left=0
    rolecols=['#306B93','#7099B4','#99B5C7','#BDD0DC',ORANGE]
    purposes=['Model fitting','Early stopping','Isotonic fitting','Rule selection','Frozen evaluation']
    for j,(r,co) in enumerate(zip(roles,rolecols)):
        frac=int(r['proteins'])/total
        ax.add_patch(Rectangle((left,.55),frac,.25,facecolor=co,edgecolor='white',lw=.9))
        dest=.075+j*.207
        ax.plot([left+frac/2,dest],[.53,.26],color=co,lw=.65)
        ax.text(dest,.17,r['role'],ha='center',va='center',fontweight='bold',fontsize=8)
        left+=frac;record(4,'A',r,derived_fraction=frac)
    f.text(.065,.94,f'{total:,} proteins; {sum(int(r["components"]) for r in roles):,} components',fontsize=7.5)
    for j,r in enumerate(roles):
        x=.133+j*.186
        f.text(x,.834,f'{int(r["proteins"]):,} proteins',ha='center',fontsize=8)
        f.text(x,.814,f'{int(r["components"]):,} components',ha='center',fontsize=7,color=MUTED)
        f.text(x,.794,purposes[j],ha='center',fontsize=7)
    f.text(.065,.765,'0 qualifying cross-role edges in the specified sequence searches',fontsize=8)
    heading(f,'B','More candidate support does not ensure better top-1 selection',.724)
    f.legend([Line2D([0],[0],color=BLUE,marker='o',lw=1.2),
              Line2D([0],[0],color=ORANGE,marker='s',ls='--',lw=1.2)],
             ['Matching candidate available','Top-1 prediction concordant'],
             loc='upper center',bbox_to_anchor=(.52,.696),ncol=2)
    for j,lev in enumerate(LEVELS):
        ax=axes(f,[.09+j*.305,.478,.25,.151])
        rr=[one(budget,level=lev,budget=b) for b in ['10','25','50','ALL']]
        ax.axvspan(1.85,2.15,color=LIGHT)
        for key,co,mark,ls in [('oracle_fraction',BLUE,'o','-'),
                               ('top1_concordant_fraction_all_queries',ORANGE,'s','--')]:
            ax.plot(range(4),[100*float(r[key]) for r in rr],marker=mark,ls=ls,color=co,ms=3,lw=1.25)
        for r in rr:check(int(r['queries_total'])==20637,'Fig4 budget denominator');record(4,'B',r)
        ax.set(xticks=range(4),xticklabels=['10','25','50','ALL'],ylim=(0,100),yticks=[0,25,50,75,100])
        ax.set_title(LABELS[j],fontsize=8.5,color=COLORS[j],fontweight='bold');ax.set_xlabel('Candidate budget')
        if j==0:ax.set_ylabel('RETEST queries (%)')
        else:ax.set_yticklabels([])
    heading(f,'C','Sampling uncertainty across 192 sequence components',.409)
    names=['Coverage','Accepted precision','Concordant coverage']
    bins=np.arange(0,102,2)
    max_hist=max(np.histogram([100*float(b[r['metric']]) for b in boot],bins=bins)[0].max() for r in endpoints)
    histogram_ceiling=int(np.ceil(max_hist/50)*50)
    for j,(r,name) in enumerate(zip(endpoints,names)):
        ax=axes(f,[.09+j*.305,.23,.25,.105])
        v,lo,hi=[100*float(r[k]) for k in ['estimate','lower_95','upper_95']]
        values=np.array([float(b[r['metric']])*100 for b in boot])
        check(np.allclose(np.quantile(values,[.025,.975]),[lo,hi],atol=1e-10),'Fig4 original CI from saved draws '+name)
        check(np.isclose(float(r['numerator'])/float(r['denominator'])*100,v),'Fig4 point arithmetic '+name)
        ax.axvspan(lo,hi,color='#E7EDF2',zorder=0)
        hist_values,_,_=ax.hist(values,bins=bins,histtype='stepfilled',color=BLUE,alpha=.6,lw=.5)
        check(hist_values.sum()==2000 and hist_values.max()<=histogram_ceiling,'Fig4 histogram not clipped '+name)
        ax.axvline(v,color=INK,lw=1.2);ax.set(xlim=(0,100),xticks=[0,50,100],ylim=(0,histogram_ceiling),yticks=[0,100,200])
        ax.set_title(name,fontsize=8.5);ax.set_xlabel('Estimate (%)')
        if j==0:ax.set_ylabel('Resamples')
        else:ax.set_yticklabels([])
        f.text(.09+j*.305,.373,f'{v:.2f}%  [95% CI {lo:.2f}–{hi:.2f}]',fontsize=7)
        record(4,'C',r,saved_bootstrap_column=r['metric'],histogram_bin_width_pp=2)
    heading(f,'D','Accepted outputs by reported resolution',.161)
    ax=axes(f,[.14,.055,.52,.078],grid='x')
    total=0
    for j,lev in enumerate(LEVELS):
        r=one(resolution,resolution=lev);left=0;total+=int(r['outputs'])
        check(sum(int(r[k]) for k in ['concordant','discordant','unevaluable'])==int(r['outputs']),'Fig4 resolution closure '+lev)
        for key,co in [('concordant',BLUE),('discordant',ORANGE),('unevaluable',MUTED)]:
            n=int(r[key]);ax.barh(2-j,n,left=left,height=.52,color=co);left+=n
        ax.text(left+170,2-j,f'{int(r["concordant"]):,}/{int(r["evaluable"]):,}',va='center',fontsize=7)
        record(4,'D',r)
    check(total==11686,'Fig4 hierarchical output closure')
    ax.set(yticks=[2,1,0],yticklabels=LABELS,xlim=(0,14000),xticks=[0,5000,10000],xticklabels=['0','5,000','10,000'])
    ax.set_xlabel('Accepted queries; labels = concordant / evaluable',fontsize=7)
    f.legend([Patch(color=co) for co in [BLUE,ORANGE,MUTED]],['Concordant','Disagreement','Unevaluable'],
             loc='center left',bbox_to_anchor=(.735,.088),fontsize=7,handlelength=1)
    f.text(.74,.031,'8,951 abstentions\n2 unevaluable Rhea outputs',fontsize=7,color=MUTED)
    export(f,4)

def fig5():
    states=table(5,'native_states.tsv');metrics=table(5,'native_metrics.tsv');sizes=table(5,'native_output_set_sizes.tsv')
    f=newfig(7.45)
    heading(f,'A','Native transfer: support, retrieval and rank-1 selection',.982)
    f.text(.065,.94,'Aggregate nested counts; N = 27,639. This workflow has no sampling stage.',fontsize=7.5)
    for j,lev in enumerate(LEVELS):
        truth=27639-int(one(states,level=lev,state='NO_DOCUMENTED_TRUTH')['queries'])
        lib=truth-int(one(states,level=lev,state='NO_TRAIN_LIBRARY_SUPPORT')['queries'])
        top=int(one(states,level=lev,state='NATIVE50_MATCH_RETAINED')['queries'])
        match=int(one(metrics,level=lev,metric='query_any_match_evaluable')['numerator'])
        vals=[truth,truth-lib,lib-top,top-match,match]
        check(truth-sum(vals[1:4])==match and min(vals)>=0,'Fig5 nested waterfall closure '+lev)
        check(one(states,level=lev,state='POST_RETRIEVAL_SAMPLING_LOSS')['applicable']=='False','Fig5 no fictitious sampling stage')
        ax=axes(f,[.085+j*.307,.65,.26,.222])
        lower=[0,lib,top,match,0];co=[COLORS[j],ORANGE,ORANGE,ORANGE,COLORS[j]]
        ax.bar(range(5),vals,bottom=lower,color=co,width=.67,edgecolor='white',lw=.5)
        for i,(v,b) in enumerate(zip(vals,lower)):
            offset=2300 if i==1 else 550
            ax.text(i,b+v+offset,('−' if i in [1,2,3] else '')+f'{v:,}',ha='center',fontsize=6.5)
        for i,y in enumerate([truth,lib,top,match]):ax.plot([i+.34,i+.66],[y,y],color=MUTED,lw=.55)
        ax.set(ylim=(0,32000),xticks=range(5),xticklabels=['Truth','Ref.\nloss','Retr.\nloss','Rank-1\nloss','Match'],
               yticks=[0,10000,20000,30000],yticklabels=['0','10k','20k','30k'])
        ax.tick_params(axis='x',labelsize=6.5)
        ax.set_title(LABELS[j],fontweight='bold',color=COLORS[j])
        if j==0:ax.set_ylabel('Queries')
        else:ax.set_yticklabels([])
        record(5,'A',{'level':lev,'truth':truth,'library':lib,'top50':top,'rank1_match':match},derived_stage_losses=vals[1:4])
    heading(f,'B','How many distinct labels does the reference emit?',.564)
    f.text(.065,.524,'Discrete distributions; log query-count axis; each panel contains all 27,639 queries.',fontsize=7.5)
    for j,lev in enumerate(LEVELS):
        ax=axes(f,[.085+j*.307,.322,.26,.156])
        rr=[r for r in sizes if r['level']==lev]
        check(sum(int(r['queries']) for r in rr)==27639,'Fig5 histogram closure '+lev)
        x=[int(r['predicted_label_count']) for r in rr]; n=[int(r['queries']) for r in rr]
        # A disclosed 0.5 log-axis baseline makes one-query bins visible.
        ax.bar(x,np.array(n)-.5,bottom=.5,width=.8,color=COLORS[j],edgecolor='white',linewidth=.2)
        ax.set(yscale='log',ylim=(.5,40000),xlim=(-.75,20.75),xticks=[0,5,10,15,20],yticks=[1,100,10000],yticklabels=['1','100','10k'])
        ax.minorticks_off()
        ax.set_title(LABELS[j],fontweight='bold',color=COLORS[j],fontsize=8.5)
        ax.set_xlabel('Emitted label count')
        if j==0:ax.set_ylabel('Queries (log scale)')
        else:ax.set_yticklabels([])
        for r in rr:record(5,'B',r)
    heading(f,'C','Three exact-Rhea proportions, three distinct denominators',.235)
    ax=axes(f,[.55,.062,.39,.119],grid=None)
    specs=[('emission_coverage_all','Queries with an output','all queries'),
           ('query_any_match_evaluable','Query any-match','evaluable queries'),
           ('micro_concordance_precision','Matched emitted labels','emitted labels on evaluable queries')]
    for i,(key,name,unit) in enumerate(specs):
        r=one(metrics,level='EXACT_RHEA',metric=key);v=100*float(r['value']);y=2-i
        check(np.isclose(float(r['numerator'])/float(r['denominator'])*100,v),'Fig5 denominator '+key)
        ax.barh(y,100,color=LIGHT,height=.52)
        ax.barh(y,v,color=COLORS[2],height=.52)
        ax.text(v-2,y,f'{v:.2f}%',ha='right',va='center',color='white',fontsize=8)
        f.text(.065,.169-i*.043,name,fontsize=8,fontweight='bold')
        f.text(.065,.153-i*.043,f'{int(r["numerator"]):,}/{int(r["denominator"]):,} {unit}',fontsize=6.9,color=MUTED)
        record(5,'C',r)
    ax.set(xlim=(0,100),ylim=(-.5,2.5),yticks=[],xticks=[0,50,100],xticklabels=['0%','50%','100%'])
    ax.spines['left'].set_visible(False)
    export(f,5)

def fig6():
    pv=json.loads((DATA/'Fig6/plotted_values.json').read_text())
    matched=table(6,'Figure3_matched_analysis.tsv')
    paired=table(6,'matched_pair_instances.tsv')
    check(len(paired)==297 and len({r['match_id'] for r in paired})==297,'Fig6 original 297 unique matched records')
    f=newfig(8.1)
    heading(f,'A','Mapped catalytic-site records define the analyzed subset',.982)
    ax=axes(f,[.20,.844,.61,.075],grid=None)
    for i,r in enumerate([r for r in pv if r['panel']=='A']):
        v=100*r['rate'];ax.barh(1-i,100,color=LIGHT,height=.48)
        ax.barh(1-i,v,color=BLUE,height=.48)
        ax.text(v/2,1-i,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=8)
        ax.text(102,1-i,f'{r["numerator"]:,}/{r["denominator"]:,}',va='center',fontsize=8)
        check(np.isclose(r['numerator']/r['denominator'],r['rate']),'Fig6 mapping arithmetic')
        record(6,'A',r)
    ax.set(xlim=(0,100),yticks=[1,0],yticklabels=['M-CSA','Swiss-Prot'],xticks=[0,50,100],xticklabels=['0%','50%','100%'])
    ax.spines['left'].set_visible(False)
    f.text(.20,.799,'Mapped / all test source records; source records can overlap.',fontsize=7.5)
    heading(f,'B','Increment beyond global homology',.756)
    ax=axes(f,[.11,.568,.355,.129])
    for i,lev in enumerate(LEVELS):
        r=next(r for r in pv if r['panel']=='B' and r['level']==lev)
        v,lo,hi=[r[k] for k in ['estimate','low','high']]
        ax.bar(i,v,color=COLORS[i],width=.57)
        ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor=INK,capsize=3,lw=.9)
        record(6,'B',r)
    ax.axhline(0,color=MUTED,lw=.7);ax.set(xticks=range(3),xticklabels=LABELS,ylim=(-.1,.18),yticks=[-.1,0,.1])
    ax.set_ylabel('AUPRC difference')
    f.text(.55,.688,'2,837 shared test pair records',fontsize=8,fontweight='bold')
    f.text(.55,.668,'Global + four local scores\nversus global-only model',fontsize=8,linespacing=1.5,va='top')
    f.text(.55,.61,'95% intervals: 300 paired\nquery-cluster resamples',fontsize=7.5,color=MUTED,linespacing=1.5,va='top')
    heading(f,'C','Paired outcomes at high and low local similarity',.507)
    f.text(.065,.47,'297 matched sets per resolution; cells show matched-set counts.',fontsize=7.5)
    cmap=LinearSegmentedColormap.from_list('paired',['#F5F7F9',BLUE])
    for j,lev in enumerate(LEVELS):
        r=one(matched,annotation_level=lev);n=int(r['matched_pairs'])
        high=int(round(n*float(r['local_high_concordance'])))
        low=int(round(n*float(r['local_low_concordance'])))
        ho=int(r['discordant_high_positive']);lo=int(r['discordant_low_positive'])
        # Count the original pair outcomes directly; summaries are cross-checks only.
        cells=np.zeros((2,2),dtype=int)
        for pair in paired:
            low_value,high_value=int(pair['low_'+lev]),int(pair['high_'+lev])
            check(low_value in (0,1) and high_value in (0,1),'Binary paired outcomes '+lev+' '+pair['match_id'])
            cells[low_value,high_value]+=1
        check(cells[:,1].sum()==high and cells[1,:].sum()==low,'Fig6 direct marginal counts '+lev)
        check(cells[0,1]==ho and cells[1,0]==lo,'Fig6 direct discordant counts '+lev)
        check(cells.sum()==n and cells.min()>=0,'Fig6 paired contingency closure '+lev)
        ax=axes(f,[.13+j*.3,.316,.18,.097],grid=None)
        ax.pcolormesh(np.arange(3),np.arange(3),cells,cmap=cmap,vmin=0,vmax=297,edgecolors='white',linewidth=1.2)
        ax.set(xlim=(0,2),ylim=(2,0),xticks=[.5,1.5],xticklabels=['No','Yes'],yticks=[.5,1.5],yticklabels=['No','Yes'])
        ax.tick_params(length=0);[s.set_visible(False) for s in ax.spines.values()]
        for row in range(2):
            for col in range(2):ax.text(col+.5,row+.5,str(cells[row,col]),ha='center',va='center',fontsize=10,color='white' if cells[row,col]>150 else INK)
        ax.set_title(LABELS[j],color=COLORS[j],fontweight='bold',fontsize=8.5)
        ax.set_xlabel('High similarity: match?',fontsize=7)
        if j==0:ax.set_ylabel('Low similarity: match?',fontsize=7)
        v=100*float(r['matched_risk_difference']);lower=100*float(r['risk_difference_ci_lower']);upper=100*float(r['risk_difference_ci_upper'])
        f.text(.09+j*.30,.25,f'Δ {v:+.2f} pp [{lower:.2f}, {upper:.2f}]',fontsize=7)
        f.text(.09+j*.30,.229,f'McNemar q = {float(r["mcnemar_fdr_bh"]):.4f}',fontsize=7,color=MUTED)
        record(6,'C',r,derived_contingency=cells.tolist())
    heading(f,'D','Family effects are heterogeneous',.193)
    ax=axes(f,[.115,.065,.405,.089],grid='y')
    for j,lev in enumerate(LEVELS):
        rr=[r for r in pv if r['panel']=='D_family' and r['level']==lev]
        check(len(rr)==38,'Fig6 all family effects retained '+lev)
        values=np.sort([r['value'] for r in rr]);probs=np.arange(1,39)/38*100
        ax.step(np.r_[values[0],values],np.r_[0,probs],where='post',color=COLORS[j],ls=['-','--',':'][j],lw=1.25)
        for r in rr:record(6,'D',r)
        sm=next(r for r in pv if r['panel']=='D_summary' and r['level']==lev)
        f.add_artist(Line2D([.53,.555],[.148-j*.025,.148-j*.025],transform=f.transFigure,
                            color=COLORS[j],ls=['-','--',':'][j],lw=1.25))
        f.text(.56,.144-j*.025,LABELS[j]+f'  {sm["estimate"]:+.3f} [{sm["low"]:+.3f}, {sm["high"]:+.3f}]',fontsize=7,color=COLORS[j])
        record(6,'D',sm)
    ax.axvline(0,color=MUTED,lw=.65);ax.set(xlim=(-.32,.2),ylim=(0,100),yticks=[0,50,100],xticks=[-.3,-.15,0,.15])
    ax.set_xlabel('Log-loss reduction with local scores',fontsize=7)
    ax.set_ylabel('Families (%)',fontsize=7)
    f.text(.56,.172,'Equal-family mean [95% interval]',fontsize=7,fontweight='bold')
    f.text(.56,.05,'38 families; at most 8 sequence clusters each\n(family-claim support criterion: 10).',fontsize=8,linespacing=1.5)
    export(f,6)

def fig7():
    curves=table(7,'evidencejudge_full_precision_coverage_curves.tsv')
    allcomp=table(7,'evidencejudge_paired_utility_comparisons.tsv')
    comp=[r for r in allcomp if r['primary']=='True']
    groups=[('31','EC_L3'),('31','EC_L4'),('33','EC_L3'),('33','EC_L4')]
    cohorts={'31':'2019–2021','33':'2009–2015'}
    names=[cohorts[p]+' | '+{'EC_L3':'EC-L3','EC_L4':'EC-L4'}[l] for p,l in groups]
    check(len(comp)==16,'Fig7 all 16 prespecified matched-quota contrasts')
    f=newfig(8.35)
    heading(f,'A','EvidenceJudge and the generating candidate score',.982)
    f.legend([Line2D([0],[0],color=BLUE),Line2D([0],[0],color=ORANGE,ls='--')],
             ['EvidenceJudge','Raw EC4 logit'],loc='upper center',bbox_to_anchor=(.53,.95),ncol=2)
    for j,(phase,lev) in enumerate(groups):
        ax=axes(f,[.095+(j%2)*.465,.745-(j//2)*.172,.392,.106])
        for method,col,ls in [('judge',BLUE,'-'),('hit_ec4_logit',ORANGE,'--')]:
            rr=sorted([r for r in curves if r['phase']==phase and r['level']==lev and r['method']==method],key=lambda r:int(r['accepted']))
            check([int(r['accepted']) for r in rr]==list(range(1,int(rr[0]['n'])+1)),'Fig7 complete trajectory '+phase+lev+method)
            xx=[100*float(r['coverage']) for r in rr]; yy=[100*float(r['precision']) for r in rr]
            check(min(yy)>=80,'Fig7 focused precision domain contains curve')
            ax.plot(xx,yy,color=col,ls=ls,lw=1)
            for r in rr:record(7,'A',r)
        ax.set(xlim=(0,100),ylim=(79.5,100.5),xticks=[0,50,100],yticks=[80,90,100])
        ax.set_title(names[j]+f'   n={rr[0]["n"]}; {rr[0]["n_clusters"]} clusters',loc='left',fontsize=7.4)
        if j%2==0:ax.set_ylabel('Precision (%)')
        if j>=2:ax.set_xlabel('Coverage (%)')
    heading(f,'B','Matched-quota precision differences',.49)
    f.text(.065,.452,'EvidenceJudge minus raw EC4 logit (pp); cells retain pointwise 95% cluster intervals.',fontsize=7.1)
    vals=np.zeros((4,4));refs=[]
    for i,(phase,lev) in enumerate(groups):
        rr=[]
        for j,b in enumerate([.1,.25,.5,.75]):
            r=next(r for r in comp if r['phase']==phase and r['level']==lev and float(r['budget'])==b)
            rr.append(r); vals[i,j]=float(r['delta_precision_pp'])
            check(np.isclose(vals[i,j],100*int(r['net_correct'])/int(r['accepted'])),'Fig7 quota delta arithmetic')
            record(7,'B',r)
        refs.append(rr)
    ax=axes(f,[.26,.26,.69,.153],grid=None)
    cmap=LinearSegmentedColormap.from_list('signed',[ORANGE,'#FAFAFA',BLUE])
    ax.pcolormesh(np.arange(5),np.arange(5),vals,cmap=cmap,norm=TwoSlopeNorm(0,vmin=-6,vmax=6),edgecolors='white',linewidth=1.5)
    ax.set(xlim=(0,4),ylim=(4,0),xticks=np.arange(4)+.5,xticklabels=['10%','25%','50%','75%'],yticks=np.arange(4)+.5,yticklabels=names)
    ax.xaxis.tick_top();ax.tick_params(length=0,labelsize=7);[s.set_visible(False) for s in ax.spines.values()]
    for i in range(4):
        for j in range(4):
            r=refs[i][j];v=vals[i,j];lo=float(r['delta_ci_low_pp']);hi=float(r['delta_ci_high_pp'])
            ax.text(j+.5,i+.4,f'{v:+.2f}',ha='center',va='center',fontsize=8,fontweight='bold')
            ax.text(j+.5,i+.77,f'[{lo:.2f}, {hi:.2f}]',ha='center',va='center',fontsize=6.4)
    heading(f,'C','Selection turnover at the 50% quota',.213)
    ax=axes(f,[.12,.103,.81,.075])
    for j,(phase,lev) in enumerate(groups):
        r=next(r for r in comp if r['phase']==phase and r['level']==lev and float(r['budget'])==.5)
        gain,lost,net=[int(r[k]) for k in ['added_correct','lost_correct','net_correct']]
        check(gain-lost==net,'Fig7 observed concordant turnover')
        ax.bar(j,gain,color=BLUE,width=.5)
        ax.bar(j,-lost,color=ORANGE,width=.5,hatch='///',edgecolor='white',lw=.3)
        ax.text(j,gain+4,f'+{gain}',ha='center',va='bottom',fontsize=7)
        ax.text(j,-lost-4,f'−{lost}',ha='center',va='top',fontsize=7)
        f.text(.204+j*.2025,.03,f'Net {net:+d}',ha='center',fontsize=8,fontweight='bold')
        record(7,'C',r)
    ax.axhline(0,color=MUTED,lw=.7)
    ax.set(ylim=(-100,100),yticks=[-60,0,60],xticks=range(4),xticklabels=[n.replace(' | ','\n') for n in names])
    ax.tick_params(axis='x',labelsize=7);ax.set_ylabel('Output count',fontsize=7)
    f.legend([Patch(facecolor=BLUE),Patch(facecolor=ORANGE,hatch='///',edgecolor='white')],['Gained','Displaced'],
             loc='upper right',bbox_to_anchor=(.97,.206),ncol=2,fontsize=7)
    export(f,7)

def finish():
    for rel, meta in INPUTS.items():
        check(sha(DATA/rel)==meta['sha256'],'Source unchanged during rendering '+rel)
    if (OUT/'pypdf-6.10.0-purepy.zip').exists():sys.path.insert(0,str(OUT/'pypdf-6.10.0-purepy.zip'))
    from pypdf import PdfReader,PdfWriter
    writer=PdfWriter()
    pdfchecks=[]
    for n in range(2,8):
        path=OUT/f'Fig{n}.pdf';reader=PdfReader(path)
        check(len(reader.pages)==1,'Fig'+str(n)+' one-page vector PDF')
        check(len(reader.pages[0].images)==0,'Fig'+str(n)+' no raster image objects')
        writer.add_page(reader.pages[0])
        pdfchecks.append({'figure':n,'page_count':1,'raster_images':0,'bytes':path.stat().st_size,'sha256':sha(path)})
    writer.write(OUT/'Figures_2-7_redesigned.pdf')
    (OUT/'input_manifest.json').write_text(json.dumps(INPUTS,indent=2),encoding='utf-8')
    (OUT/'plotted_values.json').write_text(json.dumps(PLOTTED,indent=2),encoding='utf-8')
    (OUT/'numeric_validation.json').write_text(json.dumps({'status':'PASS','checks':CHECKS,'check_count':len(CHECKS),
        'new_training':False,'new_resampling':False,'visual_QA':'pending PDF rasterization',
        'figure_pdfs':pdfchecks,'plotted_source_records':len(PLOTTED)},indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'checks':len(CHECKS),'plotted_source_records':len(PLOTTED)}))

stage_sources()
for fun in [fig2,fig3,fig4,fig5,fig6,fig7]:fun()
finish()
