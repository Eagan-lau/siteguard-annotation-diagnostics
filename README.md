# SiteGuard

Query-level diagnosis for **Diagnosing reference support and selection failures in enzyme annotation**. Version 2.6.0.

The diagnostic follows recorded activities through reference support, available candidates and selected outputs. It separates reference gaps, candidate losses and selection failures, then evaluates the gains and costs of a specific change. Native adapters demonstrate the diagnostic on CLEAN EC centres and DIAMOND protein references. SiteGuard is the evaluated candidate-scoring workflow; EvidenceJudge is a downstream selection control.

## Quick start

The core diagnostic and synthetic example require only Python 3.9 or later.

```bash
python diagnose.py --example --output demo_output
python -m unittest -v test_diagnose.py
```

For your own workflow, supply the four tables in [INPUT_FORMAT.md](INPUT_FORMAT.md). Native set-valued adapters and the temporal paired replay use NumPy. [REPRODUCE.md](REPRODUCE.md) gives exact commands.

## Article materials

| Directory | Contents |
| --- | --- |
| `example/` | Ten synthetic queries and expected states |
| `analysis/` | Paired CLEAN statistics, component diagnosis, native CLEAN/DIAMOND adapters and temporal replay |
| `article_figures/` | Current source-data renderers and graphical-abstract renderer |
| `workflow/` | Evaluated acquisition, feature, model and evaluation implementations |

Download the [data and final models](https://doi.org/10.5281/zenodo.22845711) and extract the ZIP beside this README to obtain `data/`. It includes final predictions, candidate records, component draws, selected fitted models, current figures and the two standalone extension packages. Provider resources are identified in [DATA_SOURCES.md](DATA_SOURCES.md).

Permanent software archive: https://doi.org/10.5281/zenodo.22845664. Data archive: https://doi.org/10.5281/zenodo.22845711. GitHub release: https://github.com/Eagan-lau/siteguard-annotation-diagnostics/releases/tag/v2.6.0.

Runnable result replays use saved observations and scores. Upstream model fitting also requires provider resources and feature matrices, mapped in [workflow/README.md](workflow/README.md).

## Interpretation

Concordance is agreement with documented activities. Missing documentation is distinct from an observed disagreement. Native adapters use set-valued endpoints. Unavailable acceptance decisions remain unknown. The temporal intervention is a descriptive ten-record, nine-component comparison; it is not a claim of improved generalization.

## License and citation

Original code: [MIT](LICENSE). Original derived data and figures: [CC BY 4.0](LICENSE-DATA.md). Reused resources retain their provider terms. Citation metadata are in `CITATION.cff`.
