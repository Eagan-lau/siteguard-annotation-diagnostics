# SiteGuard annotation diagnostics

Software for **Diagnosing reference support and selection failures in enzyme annotation**.

The diagnostic traces a query's documented activities through a reference library, candidate set and selected outputs. It identifies missing reference support, candidate loss and selection disagreement. Adapters cover SiteGuard candidate scores, CLEAN EC-centre distances and DIAMOND protein hits.

## Try the diagnostic

Python 3.9 or later is sufficient for the core and synthetic example.

```bash
python diagnose.py --example --output demo_output
python -m unittest -v test_diagnose.py
```

To analyze another workflow, supply the four tables described in [INPUT_FORMAT.md](INPUT_FORMAT.md). [REPRODUCE.md](REPRODUCE.md) gives commands for the article's datasets and figures.

## Files

| Location | Contents |
| --- | --- |
| `example/` | Ten synthetic queries and expected diagnostic states |
| `analysis/` | Native workflow adapters, paired comparisons, component diagnosis and temporal replay |
| `article_figures/` | Numerical figure inputs and plotting programs |
| `workflow/` | Resource acquisition, features, training and evaluation source |

Extract the [data archive](https://doi.org/10.5281/zenodo.22859607) here to obtain `data/`. It contains query-level predictions, candidate records, resampling draws, fitted models and the article figures. Third-party database releases and pretrained models are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Concordance means agreement with documented activities. Missing annotation and unknown acceptance decisions retain separate states. The temporal dataset contains 10 records in nine sequence components and supports a descriptive paired comparison.

## Citation and license

Release: **2.7.0**. Code author: **Yugeng Liu**.

- [Software DOI](https://doi.org/10.5281/zenodo.22859697)
- [Data DOI](https://doi.org/10.5281/zenodo.22859607)
- [GitHub release](https://github.com/Eagan-lau/siteguard-annotation-diagnostics/releases/tag/v2.7.0)

Original code uses [MIT](LICENSE). Original derived data and artwork use [CC BY 4.0](LICENSE-DATA.md). Third-party resources retain their provider terms. Machine-readable software citation: `CITATION.cff`.
