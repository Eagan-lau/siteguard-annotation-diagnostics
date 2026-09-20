# Scientific workflow

| Stage | Implementation |
| --- | --- |
| Provider downloads and catalogs | `scripts/acquisition/` |
| Cluster-split benchmark | `scripts/cluster_split/` |
| Sequence-separated benchmark | `scripts/sequence_separated/` |
| Embedding extraction | `scripts/embedding_helpers/` |
| Shared annotation and prediction functions | `siteguard/` |

The cluster-split implementation includes activity harmonization, reaction vocabulary, candidate retrieval, pair features, neural and tree models, calibration, candidate restoration and the secondary controls. The sequence-separated implementation includes component assignment, reference-library construction, retrieval, balanced training pairs, three-seed model fitting, calibration and held-out evaluation.

Run individual programs with their input/output arguments. Programs that use an experiment contract expect the input tables and metadata defined by that contract. Input record identifiers and schema keys are retained for compatibility with the scientific datasets. The root `REPRODUCE.md` gives the portable analyses that run on the deposited data.

CPU dependencies are listed in `requirements-analysis.txt`. Model fitting used Python 3.11.3, PyTorch 2.1.2, CUDA 12.1.1 and NumPy 1.25.1; see `training-environment.json`. `configs/sequence_rebuild.json` records the architecture, seeds, role counts and calibration objective. Provider resources are listed in the root `DATA_SOURCES.md`.
