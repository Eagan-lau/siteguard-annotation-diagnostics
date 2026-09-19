# Figure source data and reproduction

This directory contains the evaluated numerical inputs and source-bundle renderers. The current manuscript uses the mapping below. From the repository root run `python render_current_figures.py --output outputs/current_figures` to apply this mapping and add the native-workflow Figure 2 and graphical abstract.

## Usage

Use Python 3.11 or later with NumPy, Matplotlib and Pillow. The included pypdf distribution supports the source-bundle renderer. Install Arial for the authored font geometry.

```sh
python render_article_figures.py --output-dir figures_output
```

The output directory must be new. The command writes vector PDFs, editable SVGs and PNG previews for all eight figures, with TIFFs for Figures 1, 2 and 5. Sources are read relative to the extracted archive, and intermediate renders use a temporary directory.

## Numerical source mapping

| Article figure | Numerical source | Renderer |
| --- | --- | --- |
| Figure 1 | `source_data/Fig1/numerical_source.json` | `render_figure1.py` |
| Figure 4 | `source_data/Fig2/phase444_*.tsv` | `render_figure2.py` |
| Figure 5 | `source_data/Fig3/` | `build_figures.py`, source figure 3 |
| Figure 6 | `source_data/Fig4/` | `build_figures.py`, source figure 4 |
| Figure 3 | `diagnostic_source/` | `build_diagnostic_figure.py` |
| Figure S14 | `source_data/Fig5/` | `build_figures.py`, source figure 5 |
| Figure S12 | `source_data/Fig6/` | `build_figures.py`, source figure 6 |
| Figure S13 | `source_data/Fig7/` | `build_figures.py`, source figure 7 |

Figure 2 retains all six paired availability estimates and pointwise 95% intervals. Its right-hand panel C view repeats the three retrieval-extension estimates on a 0-2.5 percentage-point scale. Figure 5 uses the 1,116 EC-L4 records at a Top-50 budget in the saved diagnosis and score tables. Figure S12 includes the original 297 matched sets as Parquet and an equivalent tab-delimited file. `supporting_sources/S5/` contains the candidate multiplicity, retrieval-rank and cluster-weighting source tables for Figure S5.

`input_manifest.json` records the source-table identities used by the source-bundle renderer. `FILE_MANIFEST.tsv` records the SHA-256 hash and byte count of each archive member, excluding itself.

The internal source-bundle command retains its source numbering; the root command applies the manuscript numbering. Native Figure 2 uses `analysis/native_workflows/render_figure2.py` and `data/native_workflows/results/`. The graphical abstract is a conceptual schematic drawn by `build_graphical_abstract.py`.
