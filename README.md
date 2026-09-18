# SiteGuard

Query-level diagnosis for **Diagnosing reference support and selection failures
in enzyme annotation**. Version 2.5.1.

The diagnostic traces recorded labels through a reference library, retrieved
candidates, retained candidates and selected outputs. It separates missing
support from score and tie-related selection losses, while recording acceptance
as a separate decision. SiteGuard is the evaluated annotation workflow;
EvidenceJudge is its downstream selection control.

## Quick start

The diagnostic and synthetic example need only Python 3.9 or later.

```bash
python diagnose.py --example --output demo_output
python -m unittest -v test_diagnose.py
```

For another workflow, provide the four tables in [INPUT_FORMAT.md](INPUT_FORMAT.md):

```bash
python diagnose.py --input my_tables --output my_diagnosis
```

## Article materials

| Directory | Contents |
| --- | --- |
| `example/` | Ten synthetic queries and expected states |
| `analysis/` | Paired CLEAN statistics and query/component diagnosis |
| `figures/` | Rendering code and numerical sources for Figs 1–6 and S12–S13 |
| `workflow/` | Scientific acquisition, feature, model and evaluation source |

The companion data archive contains the final predictions, candidate records,
component resamples, diagnostic tables, model parameters and figures. Extract
it into `data/`, then follow [REPRODUCE.md](REPRODUCE.md).
Provider releases are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Version identifiers: [software](https://doi.org/10.5281/zenodo.22832200)
and [data](https://doi.org/10.5281/zenodo.22832099).

The runnable result-reproduction entries use saved observations and scores.
Upstream model fitting additionally requires the corresponding source resources
and feature matrices. [workflow/README.md](workflow/README.md) maps those stages.

## Interpretation

Agreement is measured against recorded enzyme annotations. Missing recorded
activity is distinct from an observed disagreement. The diagnostic describes
single-label selections; set-valued predictions require set-valued endpoints.
Blank acceptance means unavailable, not rejection. See INPUT_FORMAT.md for the
state definitions and required input scope.

## Licence and citation

Original code: [MIT](LICENSE). Original derived data and figures:
[CC BY 4.0](LICENSE-DATA.md). Third-party terms remain applicable.
Citation metadata are in `CITATION.cff`.
