# Diagnostic input

Four UTF-8 tab-delimited tables are required in one directory.

| File | Required columns |
| --- | --- |
| `queries.tsv` | query_id, endpoint, truth_labels |
| `library.tsv` | endpoint, label |
| `candidates.tsv` | query_id, endpoint, candidate_id, label, retained, score |
| `outputs.tsv` | query_id, endpoint, selected_candidate_id, accepted |

`truth_labels` is a JSON string list; `[]` denotes unavailable recorded
annotations. The library table lists every distinct label in the reference
library actually used. Candidate IDs are unique within each query and endpoint.
`retained` is 0 or 1, and retained candidates have finite scores. The output
table records the workflow's existing selection; an empty candidate ID denotes
abstention. `accepted` is 0, 1 or blank for an unavailable acceptance decision.

The state identifies the first missing support stage: no recorded annotation,
no library support, no matching saved candidate, or retention loss. With
retained support, it distinguishes concordant selection, abstention,
higher-score loss, a top-score tie containing a matching label, and another
selection-policy loss. Exact equality of the supplied stored scores defines
ties. Activity labels must already be harmonized.

Missing support in a saved candidate union does not identify which search or
deduplication operation caused the loss. The diagnostic preserves that scope.
Acceptance is reported separately, with evaluable denominators shown explicitly.
The ten-query example covers these states using synthetic labels.
