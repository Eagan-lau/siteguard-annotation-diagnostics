"""Render Figure 2 from the accompanying candidate-availability sources.

All dimensions and font sizes are final print sizes. Run with Python 3,
NumPy, Matplotlib and Pillow.
Run: python render_figure2.py --output-dir fig2_output
"""
from pathlib import Path
from io import BytesIO
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / 'source_data/Fig2'
WIDTH, HEIGHT = 7.5, 8.7
N = 27639
LEVELS = ['EC_L3', 'EC_L4', 'EXACT_RHEA']
LABELS = ['EC-L3', 'EC-L4', 'Exact Rhea']
COLORS = ['#306B93', '#635F9A', '#227F79']
INK, MUTED, GRID = '#253742', '#596874', '#E3E8EB'
BLUE, ORANGE = '#2F6F91', '#B96C43'
CHECKS, VALUES = [], []


def check(test, message):
    if not test:
        raise AssertionError(message)
    CHECKS.append(message)


def read_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def one(rows, **selectors):
    selected = [r for r in rows if all(r[k] == v for k, v in selectors.items())]
    check(len(selected) == 1, 'Unique row: ' + repr(selectors))
    return selected[0]


def record(panel, source, row, **extra):
    VALUES.append({'panel': panel, 'source': source, 'source_row': row, **extra})


def xy(x, y):
    return x / WIDTH, y / HEIGHT


def label(fig, x, y, text, **kwargs):
    return fig.text(*xy(x, y), text, **kwargs)


def heading(fig, letter, title, y):
    label(fig, .14, y, letter, fontsize=12, fontweight='bold', va='top')
    label(fig, .48, y, title, fontsize=10.5, fontweight='bold', va='top')


def axis(fig, x, y, w, h, grid=True):
    ax = fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', length=3, width=.65, labelsize=8)
    if grid:
        ax.yaxis.grid(True, color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    return ax


def render_a(fig, part):
    heading(fig, 'A', 'Where documented candidate support is lost', 8.53)
    states = ['NO_DOCUMENTED_TRUTH', 'NO_TRAIN_LIBRARY_SUPPORT',
              'LIBRARY_SUPPORTED_UNION50_MISS', 'UNION50_POSITIVE_SCORED_SUBSET_MISS',
              'SCORED_SUBSET_POSITIVE']
    fills = ['#DCE1E5', '#7A8791', '#C6D8E4', ORANGE, BLUE]
    names = ['No truth', 'No reference', 'Retrieval miss', 'Sampling loss', 'Retained']
    fig.legend([Patch(facecolor=c) for c in fills], names,
               loc='upper center', bbox_to_anchor=xy(3.83, 8.20), ncol=5,
               columnspacing=1.15, handlelength=1.1, handletextpad=.45)
    ax = axis(fig, 1.14, 6.78, 6.08, 1.12, grid=False)
    for j, lev in enumerate(LEVELS):
        left, total = 0., 0
        for k, (key, color) in enumerate(zip(states, fills)):
            row = one(part, level=lev, state=key)
            n = int(row['queries'])
            width = 100 * n / N
            total += n
            check(int(row['denominator_all']) == N, 'Panel A denominator: ' + lev + '/' + key)
            check(np.isclose(width, 100 * float(row['fraction_all'])), 'Panel A fraction: ' + lev + '/' + key)
            ax.barh(2-j, width, left=left, height=.59, color=color,
                    edgecolor='white', linewidth=.6)
            if width > 9:
                ax.text(left+width/2, 2-j, f'{width:.1f}%', ha='center', va='center',
                        fontsize=8, color='white' if k in [1, 3, 4] else INK)
            left += width
            record('A', 'phase444_partition.tsv', row, plotted_percentage=width)
        check(total == N, 'Panel A partition closes: ' + lev)
    ax.set(yticks=[2, 1, 0], yticklabels=LABELS, xlim=(0, 100),
           xticks=[0, 25, 50, 75, 100], ylim=(-.55, 2.55))
    ax.set_xlabel('Queries (%) | N = 27,639 at each resolution', labelpad=6)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8)


def render_b(fig, av):
    heading(fig, 'B', 'Retrieval depth and the historical scoring subset', 6.16)
    fig.legend([Line2D([0], [0], color=INK, lw=1.5),
                Line2D([0], [0], color=MUTED, ls='--', lw=1),
                Line2D([0], [0], color=ORANGE, ls=':', lw=1.2)],
               ['Retrieval union', 'TRAIN library', 'Scored subset'],
               loc='upper center', bbox_to_anchor=xy(3.92, 5.82), ncol=3,
               handlelength=2, columnspacing=2)
    for j, lev in enumerate(LEVELS):
        ax = axis(fig, .83+j*2.28, 3.93, 1.78, 1.38)
        vals = []
        for k in [10, 20, 50, 100]:
            row = one(av, level=lev, stage=f'union_at_{k}', denominator='all_phase12_queries')
            value = 100 * float(row['availability'])
            check(int(row['eligible_queries']) == N, 'Panel B denominator: ' + lev + '/' + str(k))
            check(np.isclose(value, 100 * int(row['positive_queries']) / N),
                  'Panel B numerator: ' + lev + '/' + str(k))
            vals.append(value)
            record('B', 'phase444_availability.tsv', row, plotted_percentage=value)
        ax.plot([10, 20, 50, 100], vals, color=COLORS[j], marker=['o', 's', '^'][j],
                markersize=3.8, linewidth=1.5)
        for stage, style, color in [('library', '--', MUTED), ('scored_subset', ':', ORANGE)]:
            row = one(av, level=lev, stage=stage, denominator='all_phase12_queries')
            value = 100 * float(row['availability'])
            check(np.isclose(value, 100 * int(row['positive_queries']) / N),
                  'Panel B reference-line numerator: ' + lev + '/' + stage)
            ax.axhline(value, color=color, linestyle=style, linewidth=1.1)
            record('B', 'phase444_availability.tsv', row, plotted_percentage=value)
        ax.set(xlim=(0, 105), ylim=(0, 105), xticks=[10, 50, 100], yticks=[0, 25, 50, 75, 100])
        ax.set_xlabel('Top-k union', labelpad=6)
        ax.set_title(LABELS[j], color=COLORS[j], fontsize=9, fontweight='bold', pad=8)
        if j == 0:
            ax.set_ylabel('Matching candidate available (%)', labelpad=7)
        else:
            ax.set_yticklabels([])


def render_c(fig, ci, av, part):
    heading(fig, 'C', 'Restoring scoring support versus extending retrieval', 3.18)
    fig.legend([Patch(facecolor=BLUE, edgecolor=BLUE),
                Patch(facecolor='white', edgecolor=ORANGE, linewidth=1.1)],
               ['Retain all Top-50 candidates', 'Extend retrieval to Top-100'],
               loc='upper left', bbox_to_anchor=xy(.75, 2.87), ncol=2,
               handlelength=1.25, columnspacing=1.8)
    ax = axis(fig, .81, .91, 3.56, 1.52)
    zoom = axis(fig, 5.47, .91, 1.77, 1.52)
    for plot in [ax, zoom]:
        plot.set_xticks(np.arange(3), LABELS)
        plot.tick_params(axis='x', length=0, pad=6)
    ax.set(xlim=(-.55, 2.55), ylim=(0, 46), yticks=[0, 10, 20, 30, 40])
    ax.set_ylabel('Matching-candidate availability gain (pp)', labelpad=8, fontsize=8)
    zoom.set(xlim=(-.6, 2.6), ylim=(0, 2.5), yticks=[0, .5, 1, 1.5, 2, 2.5],
             yticklabels=['0', '0.5', '1.0', '1.5', '2.0', '2.5'])
    zoom.set_ylabel('Gain (pp)', labelpad=6)
    zoom.set_title('Retrieval gain: expanded scale', fontsize=8, pad=10)
    check(len(ci) == 6, 'Panel C has all six paired contrasts')
    for j, lev in enumerate(LEVELS):
        for idx, contrast in enumerate(['union50_minus_scored', 'union100_minus_union50']):
            row = one(ci, level=lev, contrast=contrast)
            value = 100 * float(row['difference_fraction'])
            lo, hi = [100 * float(row[key]) for key in ['ci95_low', 'ci95_high']]
            check(int(row['all_queries']) == N and int(row['clusters']) == 1220 and
                  int(row['replicates']) == 2000, 'Panel C sampling metadata: ' + lev + '/' + contrast)
            check(np.isclose(value, 100 * int(row['additional_positive_queries']) / N),
                  'Panel C estimate arithmetic: ' + lev + '/' + contrast)
            check(lo <= value <= hi, 'Panel C asymmetric interval: ' + lev + '/' + contrast)
            union50 = int(one(av, level=lev, stage='union_at_50', denominator='all_phase12_queries')['positive_queries'])
            other_stage = 'scored_subset' if idx == 0 else 'union_at_100'
            other = int(one(av, level=lev, stage=other_stage, denominator='all_phase12_queries')['positive_queries'])
            expected = union50-other if idx == 0 else other-union50
            check(expected == int(row['additional_positive_queries']), 'Panel C matches B count difference: ' + lev + '/' + contrast)
            if idx == 0:
                state = one(part, level=lev, state='UNION50_POSITIVE_SCORED_SUBSET_MISS')
                check(expected == int(state['queries']), 'Panel C retention gain matches A loss: ' + lev)
            x = j + (-.17 if idx == 0 else .17)
            color, face = (BLUE, BLUE) if idx == 0 else (ORANGE, 'white')
            ax.bar(x, value, width=.30, color=face, edgecolor=color, linewidth=1.1, zorder=3)
            ax.errorbar(x, value, yerr=[[value-lo], [hi-value]], fmt='none',
                        ecolor=INK if idx == 0 else ORANGE, elinewidth=.9, capsize=3, capthick=.9, zorder=5)
            check(0 <= lo and hi < 46, 'Panel C main interval in range: ' + lev + '/' + contrast)
            if idx == 0:
                ax.text(x, hi+1.45, f'+{value:.2f}', ha='center', va='bottom', fontsize=8.5)
            else:
                zoom.bar(j, value, width=.44, color='white', edgecolor=ORANGE, linewidth=1.1, zorder=3)
                zoom.errorbar(j, value, yerr=[[value-lo], [hi-value]], fmt='none',
                              ecolor=ORANGE, elinewidth=.9, capsize=3, capthick=.9, zorder=5)
                zoom.text(j, hi+.085, f'+{value:.2f}', ha='center', va='bottom', fontsize=8.5, color=ORANGE)
                check(0 <= lo and hi < 2.5, 'Panel C zoom interval in range: ' + lev)
            record('C', 'phase444_paired_cluster_bootstrap.tsv', row,
                   plotted_percentage_points=value, lower_95_pp=lo, upper_95_pp=hi,
                   repeated_in_zoom=(idx == 1))
    ax.add_patch(Rectangle((-.52, 0), 3.05, 2.5, fill=False, edgecolor='#8B979F',
                           linewidth=.7, linestyle=(0, (3, 2)), zorder=6))
    label(fig, .81, .43, 'Bars: paired differences; whiskers: pointwise 95% cluster-bootstrap intervals.', fontsize=8)
    label(fig, .81, .22, '27,639 queries; 1,220 clusters; 2,000 resamples. Right plot repeats gains on a 0-2.5 pp scale.', fontsize=8)


def verify_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    boxes = []
    for t in texts:
        box = t.get_window_extent(renderer)
        check(t.get_fontsize() >= 8, 'Font at least 8 pt: ' + t.get_text())
        check(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1,
              'Text within canvas: ' + t.get_text())
        boxes.append((t, box))
    overlaps = []
    for i, (left, a) in enumerate(boxes):
        for right, b in boxes[i+1:]:
            if a.overlaps(b):
                # Ignore same-position invisible axis offset labels (already filtered)
                # and require at least 0.7 pixels of actual intersection.
                dx = min(a.x1, b.x1) - max(a.x0, b.x0)
                dy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if dx > .7 and dy > .7:
                    overlaps.append([left.get_text(), right.get_text(), round(dx, 2), round(dy, 2)])
    check(not overlaps, 'No overlapping visible text: ' + repr(overlaps))
    return {'minimum_font_pt': min(t.get_fontsize() for t in texts),
            'visible_text_objects': len(texts), 'overlapping_text_pairs': overlaps,
            'canvas_inches': [WIDTH, HEIGHT]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    out = p.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8.5, 'legend.frameon': False,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED,
        'ytick.color': MUTED, 'axes.edgecolor': MUTED, 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white'})
    part = read_tsv(DATA / 'phase444_partition.tsv')
    av = read_tsv(DATA / 'phase444_availability.tsv')
    ci = read_tsv(DATA / 'phase444_paired_cluster_bootstrap.tsv')
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
    render_a(fig, part)
    render_b(fig, av)
    render_c(fig, ci, av, part)
    layout = verify_layout(fig)
    fig.savefig(out / 'Fig2.pdf', metadata={'Title': 'Candidate retention and retrieval support', 'Author': 'Yugeng Liu et al.'})
    fig.savefig(out / 'Fig2.svg')
    fig.savefig(out / 'Fig2.png', dpi=180)
    raster = BytesIO()
    fig.savefig(raster, format='png', dpi=300)
    raster.seek(0)
    with Image.open(raster) as im:
        im.convert('RGB').save(out / 'Fig2.tif', compression='tiff_lzw', dpi=(300, 300))
    plt.close(fig)
    check(len([r for r in VALUES if r['panel'] == 'A']) == 15, 'All 15 A source states retained')
    check(len([r for r in VALUES if r['panel'] == 'B']) == 18, 'All 18 B source values retained')
    check(len([r for r in VALUES if r['panel'] == 'C']) == 6, 'All six C source estimates and intervals retained')



if __name__ == '__main__':
    main()
