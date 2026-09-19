# Reproduce the article results

Extract the companion data ZIP here to obtain `data/`. Use new output directories. The Python-standard-library core diagnostic works with Python 3.9 or later. Use Python 3.12 and the pinned requirements for the full analysis and plotting environment.

```bash
python -m pip install -r requirements.txt
python diagnose.py --example --output outputs/synthetic
python adapt_s18.py data/diagnosis/source outputs/diagnostic_input
python diagnose.py --input outputs/diagnostic_input --output outputs/diagnosis
python analysis/diagnosis/recompute_diagnostic.py data/diagnosis/source outputs/component_diagnosis
python score_diagnostic.py data/diagnosis/source outputs/score_diagnostics
python analysis/clean_comparison/recompute_statistics.py --input-dir data/paired_clean --output-dir outputs/clean_comparison
python analysis/native_workflows/test_diagnostic_core.py
python analysis/native_workflows/replay_native_diagnostics.py --input data/native_workflows/source --output outputs/native_workflows
python analysis/native_workflows/summarize_native.py --results data/native_workflows/results --output outputs/native_metrics.tsv
python analysis/temporal_pilot/replay.py --data data/temporal_pilot --output outputs/temporal_pilot
python render_current_figures.py --output outputs/current_figures
```

The SiteGuard adapter reconstructs 2,232 query-endpoint states. The native CLEAN/DIAMOND adapter reconstructs 6,696 query-endpoint-output-mode states. The temporal replay reconstructs twenty paired predictions, trained feature order, frozen calibration and all 2,000 component draws from saved per-seed outputs. It does not rerun neural inference or resource acquisition.

Figure rendering uses Arial to preserve layout. `article_figures/` retains the original renderer labels; the top-level entry maps these to current Figures 1–6 and Figure S14, and renders the native Figure 2 plus the graphical abstract. Other supplemental figures and their underlying tables are supplied in `data/article_figures/` and the numerical data directories.

The complete scientific upstream source and CPU/GPU environment definitions are under `workflow/`. These result replays require no GPU, fitting or external pretrained predictor. Third-party resources are downloaded separately from the releases in DATA_SOURCES.md.
