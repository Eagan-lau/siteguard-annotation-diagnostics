# Reproduce the article results

Extract the data archive beside this file to create `data/`. The core diagnostic uses Python 3.9 or later. The numerical analyses and plotting programs use Python 3.12 and the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python diagnose.py --example --output outputs/synthetic
python adapt_siteguard.py data/diagnosis/source outputs/diagnostic_input
python diagnose.py --input outputs/diagnostic_input --output outputs/diagnosis
python analysis/diagnosis/recompute_diagnostic.py data/diagnosis/source outputs/component_diagnosis
python score_diagnostic.py data/diagnosis/source outputs/score_diagnostics
python analysis/clean_comparison/recompute_statistics.py --input-dir data/paired_clean --output-dir outputs/clean_comparison
python analysis/native_workflows/test_diagnostic_core.py
python analysis/native_workflows/replay_native_diagnostics.py --input data/native_workflows/source --output outputs/native_workflows
python analysis/native_workflows/summarize_native.py --results data/native_workflows/results --output outputs/native_metrics.tsv
python analysis/temporal_pilot/replay.py --data data/temporal_pilot --output outputs/temporal_pilot
python export_article_figures.py --output outputs/article_figures
```

Use a new output directory for each entry. The SiteGuard adapter produces 2,232 query–endpoint states. The native adapters produce 6,696 query–endpoint–output-mode states. The temporal replay uses per-seed scores to reconstruct 20 predictions and 2,000 component resamples.

## Figures

`export_article_figures.py` exports the six main figures and 14 supporting figures in the article's vector layout, together with the supplied main-figure TIFFs. `article_figures/README.md` maps numerical plotting programs to those figures. The plotting programs regenerate the analytical charts; the archived PDFs provide the selected publication layouts.

## Training and resource acquisition

`workflow/` contains acquisition, feature construction, model fitting and evaluation implementations. Repeating these upstream calculations requires the provider databases, external pretrained models and experiment-specific feature tables described in `DATA_SOURCES.md` and `workflow/README.md`. The commands above replay the deposited results without training or a GPU.
