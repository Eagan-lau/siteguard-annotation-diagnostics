# Reproduce the article results

Extract the companion data ZIP in this repository so its top-level directory
is `data/`. Each command writes to a new output directory.
Use Python 3.12 in a virtual environment for the pinned analysis dependencies.
The standalone diagnostic itself supports Python 3.9 or later without packages.

```bash
python -m pip install -r requirements.txt
python adapt_s18.py data/diagnosis/source outputs/diagnostic_input
python diagnose.py --input outputs/diagnostic_input --output outputs/diagnosis
python analysis/diagnosis/recompute_diagnostic.py data/diagnosis/source outputs/component_diagnosis
python score_diagnostic.py data/diagnosis/source outputs/score_diagnostics
python analysis/clean_comparison/recompute_statistics.py --input-dir data/paired_clean --output-dir outputs/clean_comparison
python figures/render_article_figures.py --output-dir outputs/figures
```

The real-data adapter retains 2,232 query–endpoint records. The CLEAN comparison
uses 1,116 protein records and 43 sequence components. Its statistical entry
recalculates 2,000 paired component-bootstrap draws (seed 20260819).
The score analysis reports both EC endpoints at budgets 10, 25, 50 and all.

Figure rendering requires Arial for the original layout. Source-bundle names
are mapped by the renderer to the current six main figures and S12–S13.
The companion archive also supplies final S1–S11 vector figures and their
scientific source tables. Re-rendering a supplied figure and rerunning its
upstream model are separate operations.

The full upstream workflow uses distinct CPU-analysis and GPU-training
environments, documented under `workflow/`. These result-reproduction commands
require neither a GPU nor the third-party pretrained predictors.
