# Scientific workflow source

| Stage | Source directory / principal implementation |
| --- | --- |
| Resource acquisition | `scripts/acquisition/` |
| Truth tables and reaction vocabulary | `scripts/historical/phase01_truth.py`, `phase02_canonicalize.py` |
| Historical partitioning, retrieval and features | `scripts/historical/phase03_*` through `phase07_*` |
| Historical models and calibration | `scripts/historical/phase10_*`, `phase11_*`, `phase12_*` |
| Candidate support and restoration | `scripts/historical/phase444_*`, `phase445_*`, `phase450_paired_transition.py` |
| Native nearest-reference transfer | `scripts/historical/phase450b_native_sequence_diagnostic.py` |
| Local-site, temporal and selection controls | Remaining historical scientific implementations |
| Sequence-component roles and retrieval | `scripts/rebuild/s3_*`, `s4a_*` through `s4m_*` |
| Common role features and outcomes | `scripts/rebuild/s4n_build_role_features.py`, `s4o_join_role_outcomes.py` |
| TRAIN preprocessing | `scripts/rebuild/s4p_prepare_matrices.py` |
| Three-seed residual MLP | `scripts/rebuild/s4q_train_multiseed.py` |
| Ensemble inference | `scripts/rebuild/s4r_ensemble_inference.py` |
| Isotonic fitting and hierarchical rule | `scripts/rebuild/s4s_fit_calibration_and_rule.py` |
| Held-out evaluation | `scripts/rebuild/s4t_final_retest_evaluation.py` |

Source identifiers are retained so the implementations can be matched to the
saved scientific inputs. The source files preserve the evaluated algorithms,
parameters and input checks. Shared modules are under `siteguard/`.

Use the files' command-line arguments for input and output locations. Historical
programs using input contracts require their experiment-specific tables and
metadata. They are source implementations, not a single-command installation.
The root REPRODUCE.md provides the portable entries that run directly on the
public accompanying data.

CPU analysis versions are recorded in `requirements-analysis.txt`. GPU fitting
used Python 3.11.3, PyTorch 2.1.2, CUDA 12.1.1 and NumPy 1.25.1, as recorded in
`training-environment.json`. `configs/sequence_rebuild.json` gives the fixed
architecture, seeds, roles and calibration objective. Output directories are
separate from source inputs.
