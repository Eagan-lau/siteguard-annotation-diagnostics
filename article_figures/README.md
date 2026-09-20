# Figure data and plotting programs

The article's final layouts are in the data archive under `data/article_figures/`. Run `python export_article_figures.py --output outputs/article_figures` from the repository root to export them.

| Article figure | Numerical data | Analytical plotting program |
| --- | --- | --- |
| Fig 1 | `source_data/Fig1/` | `render_figure1.py` |
| Fig 2 | `data/native_workflows/`; `plos_presentation/inputs/` | `plos_presentation/render_native_figure2.py` |
| Fig 3 | `diagnostic_source/`; `data/diagnosis/` | `build_diagnostic_figure.py` |
| Fig 4 | `source_data/Fig2/` | `render_figure2.py` |
| Fig 5 | `source_data/Fig3/` | `build_figures.py` |
| Fig 6 | `source_data/Fig4/` | `build_figures.py` |
| S1 Fig | `plos_presentation/inputs/s1_layout_values.json` | `plos_presentation/render_s1.py` |
| S5 Fig | `source_data/Fig5/` | `build_figures.py` |
| S11 Fig | `source_data/Fig6/` | `build_figures.py` |
| S12 Fig | `source_data/Fig7/` | `build_figures.py` |

The source-data directory names are identifiers used by the plotting programs. The table above gives their article numbering. Additional supporting-figure records are mapped in `data/article_figures/FIGURE_SOURCE_MAP.tsv` and `data/SUPPLEMENT_MAP.tsv`. The 297 local-similarity matched sets are supplied as TSV and Parquet in `source_data/Fig6/`.

Plotting requires the root dependencies and Arial. Use each script's `--help` for its output argument. Analytical chart rendering and export of the selected publication artwork are separate commands.
