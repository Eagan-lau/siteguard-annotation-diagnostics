#!/usr/bin/env python3
"""One native MMseqs reference-set diagnostic; no fitting or new retrieval.

--self-test uses in-memory synthetic records only. --preflight is read-only and
never opens query functional annotations. Formal prediction bytes and a receipt
are frozen before the evaluation stage opens those annotations.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import csv

WORKFLOW = 'MMSEQS_NATIVE_TOP1_REFERENCE_ALL_ANNOTATIONS'
LEVELS = ('EC_L3', 'EC_L4', 'EXACT_RHEA')
FIELDS = ('ec_l3', 'ec_l4', 'canonical_rhea')
TRUTH_FIELDS = ('ec_l3_json', 'ec_l4_json', 'canonical_rhea_json')
INPUTS = (
    'data/processed/mmseqs_candidates.parquet',
    'data/reference/activity_reference_library.parquet',
    'data/splits/split_sequence.parquet',
    'reports/phase444_1_candidate_lineage_20260907/phase444_query_lineage.parquet',
    'reports/phase445_full_candidate_inference_20260907/phase445_query_results.parquet',
)
SCORE_FIELDS = ('fident', 'qcov', 'tcov', 'evalue', 'bits')
MM_COLUMNS = ['query_protein_id', 'reference_protein_id', 'modality_rank', 'raw_rank',
              'query_partition', 'reference_partition', 'reference_cluster_id_30',
              'modality', 'candidate_provenance', *SCORE_FIELDS]
LIB_COLUMNS = ['activity_id', 'reference_protein_id', *FIELDS, 'evidence_tier',
               'source_release', 'reference_partition', 'provenance']
SPLIT_ID_COLUMNS = ['protein_id', 'split', 'cluster_id_30']
LINEAGE_ID_COLUMNS = ['query_protein_id', 'cluster_id_30', 'level']
PHASE445_ID_COLUMNS = ['query_protein_id', 'cluster_id_30', 'level', 'cohort']
ALLOWED = {INPUTS[0]: MM_COLUMNS, INPUTS[1]: LIB_COLUMNS,
           INPUTS[2]: SPLIT_ID_COLUMNS, INPUTS[3]: LINEAGE_ID_COLUMNS,
           INPUTS[4]: PHASE445_ID_COLUMNS}
STATES = ('NO_DOCUMENTED_TRUTH', 'NO_TRAIN_LIBRARY_SUPPORT',
          'LIBRARY_SUPPORTED_MMSEQS50_MISS', 'POST_RETRIEVAL_SAMPLING_LOSS',
          'NATIVE50_MATCH_RETAINED')
CASE_CATEGORIES = ('NO_TRAIN_LIBRARY_SUPPORT', 'LIBRARY_SUPPORTED_MMSEQS50_MISS',
                   'RETAINED_MATCH_TOP1_MISS', 'TOP1_REFERENCE_MATCH')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    part = Path(relative)
    require(not part.is_absolute() and '..' not in part.parts, 'Unsafe relative path')
    require('phase99' not in str(part).lower(), 'Phase99 access forbidden')
    path = (root / part).resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes project')
    return path


def dump(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def table(path, rows):
    require(bool(rows), 'Refuse schema-less empty TSV')
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path, rows, schema=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    with path.open('xb') as handle:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), handle, compression='zstd')


def identity(root, contract):
    paths = [item['path'] for item in contract['inputs']]
    require(paths[:5] == list(INPUTS), 'First five contract inputs/order differ')
    require(len(paths) == len(set(paths)), 'Duplicate contract inputs')
    for item in contract['inputs']:
        path = safe_path(root, item['path'])
        require(path.is_file(), 'Missing input: ' + item['path'])
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'],
                'Input identity mismatch: ' + item['path'])
    require(len(contract.get('prerequisites', [])) >= 2, 'A prerequisite evidence missing')
    for item in contract['prerequisites']:
        require(item['path'] in paths, 'Prerequisite must be an identity-locked input')
        data = json.loads(safe_path(root, item['path']).read_text(encoding='utf-8'))
        require(data.get('status') == item['status'], 'Prerequisite status mismatch')


def clean_label(value):
    require(value is None or isinstance(value, str), 'Label must be string/null')
    return value if value else None


def make_library(rows, split):
    references = {}
    seen = set()
    for row in rows:
        aid, rid = row['activity_id'], row['reference_protein_id']
        require(isinstance(aid, str) and aid and aid not in seen, 'Duplicate/empty activity key')
        seen.add(aid)
        require(row['reference_partition'] == 'train' and split.get(rid, (None,))[0] == 'train',
                'Non-TRAIN activity reference')
        require(row['evidence_tier'] in ('GOLD', 'SILVER'), 'Unexpected frozen evidence tier')
        info = references.setdefault(rid, {'labels': {l: set() for l in LEVELS},
            'reference_activity_ids': set(), 'reference_evidence_tiers': set(),
            'reference_source_releases': set(), 'reference_provenances': set()})
        info['reference_activity_ids'].add(aid)
        for field, level in zip(FIELDS, LEVELS):
            label = clean_label(row[field])
            if label is not None:
                info['labels'][level].add(label)
        for target, field in [('reference_evidence_tiers', 'evidence_tier'),
                              ('reference_source_releases', 'source_release'),
                              ('reference_provenances', 'provenance')]:
            value = clean_label(row[field])
            if value is not None:
                info[target].add(value)
    return references


def ingest_hit(row, cohort, split, ranks, ref_clusters, candidates, top):
    q = row['query_protein_id']
    if q not in cohort:
        return
    rid, k = row['reference_protein_id'], row['modality_rank']
    require(isinstance(k, int) and 1 <= k <= 100 and k not in ranks[q],
            'Duplicate/invalid native rank')
    require(row['query_partition'] == 'test' and row['reference_partition'] == 'train',
            'Invalid retrieval partition')
    require(split.get(q, (None,))[0] == 'test' and split.get(rid, (None,))[0] == 'train',
            'Retrieval identity absent from split')
    rc = row['reference_cluster_id_30']
    require(rc == split[rid][1] and rc != cohort[q] and q != rid, 'Retrieval cluster leakage')
    require(rc not in ref_clusters[q], 'Reference cluster not deduplicated')
    require(row['modality'] == 'mmseqs', 'Non-MMseqs candidate')
    require(isinstance(row['raw_rank'], int) and row['raw_rank'] >= k, 'Invalid raw rank')
    require(isinstance(row['candidate_provenance'], str) and row['candidate_provenance'],
            'Missing retrieval provenance')
    for field in SCORE_FIELDS:
        require(row[field] is None or (isinstance(row[field], (int, float)) and
                                     math.isfinite(row[field])), 'Non-finite native score')
    ranks[q][k] = row['raw_rank']
    ref_clusters[q].add(rc)
    if k <= 50:
        candidates[q].append((k, rid))
    if k == 1:
        require(q not in top, 'Duplicate native Top1')
        top[q] = row


def construct_predictions(cohort, references, candidates, top):
    rows = []
    for q in sorted(cohort):
        hit = top.get(q)
        rid = hit['reference_protein_id'] if hit else None
        info = references.get(rid)
        for level in LEVELS:
            labels = sorted(info['labels'][level]) if info else []
            available = set()
            for _, rp in sorted(candidates.get(q, [])):
                if rp in references:
                    available.update(references[rp]['labels'][level])
            reason = ('NO_HIT' if hit is None else 'NO_REFERENCE_ACTIVITY' if info is None
                      else 'NO_LEVEL_ANNOTATION' if not labels else 'EMITTED')
            row = {'query_protein_id': q, 'cluster_id_30': cohort[q], 'level': level,
                   'reference_protein_id': rid, 'modality_rank': 1 if hit else None,
                   'raw_rank': hit['raw_rank'] if hit else None,
                   **{f: hit[f] if hit else None for f in SCORE_FIELDS},
                   'predicted_labels': labels, 'top50_labels': sorted(available),
                   'top50_reference_count': len(candidates.get(q, [])),
                   **{f: sorted(info[f]) if info else [] for f in (
                       'reference_activity_ids', 'reference_evidence_tiers',
                       'reference_source_releases', 'reference_provenances')},
                   'output_reason': reason}
            require(set(labels) <= available, 'Top1 labels outside Top50')
            rows.append(row)
    return rows


def load_truth_free(root, expected_n):
    import pyarrow.parquet as pq
    def read(index, columns):
        path = root / INPUTS[index]
        require(set(columns) <= set(pq.ParquetFile(path).schema_arrow.names), 'Input schema mismatch')
        return pq.read_table(path, columns=columns).to_pylist()
    lineage = read(3, LINEAGE_ID_COLUMNS)
    cohort, keys = {}, set()
    for row in lineage:
        q, cluster, level = row['query_protein_id'], row['cluster_id_30'], row['level']
        require(level in LEVELS and (q, level) not in keys, 'Lineage duplicate query-level')
        require(q not in cohort or cohort[q] == cluster, 'Lineage cluster mismatch')
        cohort[q] = cluster
        keys.add((q, level))
    require(len(cohort) == expected_n and len(keys) == expected_n * 3, 'Cohort mismatch')
    reconciled = set()
    for row in read(4, PHASE445_ID_COLUMNS):
        key = (row['query_protein_id'], row['level'], row['cohort'])
        require(key not in reconciled and key[0] in cohort and key[1] in LEVELS and
                key[2] in ('sampled', 'full'), 'Phase445 cohort identity mismatch')
        require(row['cluster_id_30'] == cohort[key[0]], 'Phase445 cluster mismatch')
        reconciled.add(key)
    require(len(reconciled) == expected_n * 6, 'Phase445 cohort cardinality')
    split = {}
    for row in read(2, SPLIT_ID_COLUMNS):
        q = row['protein_id']
        require(q not in split, 'Split key duplication')
        split[q] = (row['split'], row['cluster_id_30'])
    require(all(split.get(q) == ('test', cluster) for q, cluster in cohort.items()), 'Cohort split')
    require(not {cl for sp, cl in split.values() if sp == 'train'} & set(cohort.values()),
            'Train/test cluster overlap')
    references = make_library(read(1, LIB_COLUMNS), split)
    ranks, ref_clusters, candidates, top = defaultdict(dict), defaultdict(set), defaultdict(list), {}
    path = root / INPUTS[0]
    require(set(MM_COLUMNS) <= set(pq.ParquetFile(path).schema_arrow.names), 'Native schema')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=MM_COLUMNS):
        for row in batch.to_pylist():
            ingest_hit(row, cohort, split, ranks, ref_clusters, candidates, top)
    for q, values in ranks.items():
        require(set(values) == set(range(1, max(values) + 1)), 'Gapped native ranks')
        raw_order = [values[k] for k in sorted(values)]
        require(all(a < b for a, b in zip(raw_order, raw_order[1:])),
                'Native raw rank is not strictly increasing')
        require(q in top, 'Ranked hits without rank1')
    return cohort, references, candidates, top


def evaluate(predictions, truth, global_labels):
    rows = []
    for pred in predictions:
        q, level = pred['query_protein_id'], pred['level']
        actual = truth[(q, level)]
        output, retained = set(pred['predicted_labels']), set(pred['top50_labels'])
        documented = bool(actual)
        supported = bool(actual & global_labels[level])
        retrieved = bool(actual & retained)
        matched = len(actual & output) if documented else None
        require(not retrieved or supported, 'Retrieved label absent from library')
        state = (STATES[0] if not documented else STATES[1] if not supported else
                 STATES[2] if not retrieved else STATES[4])
        selection = ('NO_DOCUMENTED_TRUTH' if not documented else 'NO_EMITTED_ANNOTATION'
                     if not output else 'EMITTED_NO_DOCUMENTED_MATCH' if not matched else
                     'EMITTED_ALL_LABELS_DOCUMENTED' if matched == len(output) else
                     'EMITTED_PARTIAL_DOCUMENTED_MATCH')
        rows.append({**pred, 'truth_labels': sorted(actual), 'truth_label_count': len(actual),
                     'predicted_label_count': len(output), 'matched_label_count': matched,
                     'unmatched_predicted_label_count': len(output) - matched if documented else None,
                     'documented_truth': documented, 'library_supported': supported,
                     'retrieved_available': retrieved, 'any_match': bool(matched) if documented else None,
                     'sampling_applicable': False, 'state': state, 'selection_state': selection})
    return rows


def parse_truth(rows, cohort):
    truth = {}
    for row in rows:
        q = row['protein_id']
        if q not in cohort:
            continue
        for level, field in zip(LEVELS, TRUTH_FIELDS):
            require((q, level) not in truth, 'Truth join explosion')
            values = json.loads(row[field])
            require(isinstance(values, list) and all(isinstance(v, str) and v for v in values),
                    'Malformed truth set')
            truth[(q, level)] = set(values)
    require(len(truth) == len(cohort) * 3, 'Truth join incomplete')
    return truth


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summaries(rows):
    states, metrics, sizes = [], [], []
    for level in LEVELS:
        selected = [r for r in rows if r['level'] == level]
        n = len(selected)
        evaluable = [r for r in selected if r['documented_truth']]
        emitted = [r for r in evaluable if r['predicted_label_count']]
        state_counts = Counter(r['state'] for r in selected)
        require(sum(state_counts.values()) == n, 'State closure')
        for state in STATES:
            applicable = state != STATES[3]
            states.append({'level': level, 'state': state, 'applicable': applicable,
                           'queries': state_counts[state], 'denominator': n,
                           'fraction': ratio(state_counts[state], n) if applicable else None})
        matches = sum(r['matched_label_count'] for r in evaluable)
        any_matches = sum(bool(r['any_match']) for r in evaluable)
        definitions = [
            ('emission_coverage_all', sum(bool(r['predicted_label_count']) for r in selected), n),
            ('emission_coverage_evaluable', len(emitted), len(evaluable)),
            ('query_any_match_evaluable', any_matches, len(evaluable)),
            ('query_any_match_emitted_evaluable', any_matches, len(emitted)),
            ('micro_concordance_precision', matches, sum(r['predicted_label_count'] for r in evaluable)),
            ('micro_concordance_recall', matches, sum(r['truth_label_count'] for r in evaluable)),
            ('macro_emitted_set_precision', sum(r['matched_label_count'] / r['predicted_label_count']
                                               for r in emitted), len(emitted)),
        ]
        for name, numerator, denominator in definitions:
            metrics.append({'level': level, 'metric': name, 'numerator': numerator,
                            'denominator': denominator, 'value': ratio(numerator, denominator)})
        for count, queries in sorted(Counter(r['predicted_label_count'] for r in selected).items()):
            sizes.append({'level': level, 'predicted_label_count': count, 'queries': queries,
                          'denominator': n, 'fraction': ratio(queries, n)})
    return states, metrics, sizes


def cases(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['level'] != 'EXACT_RHEA' or not row['documented_truth']:
            continue
        category = (row['state'] if row['state'] in CASE_CATEGORIES[:2] else
                    'TOP1_REFERENCE_MATCH' if row['any_match'] else 'RETAINED_MATCH_TOP1_MISS')
        key = hashlib.sha256(('20260819|' + WORKFLOW + '|' + row['query_protein_id']).encode()).hexdigest()
        groups[category].append((key, row['query_protein_id'], row))
    output = []
    for category in CASE_CATEGORIES:
        available = groups[category]
        key, query, row = min(available) if available else (None, None, None)
        output.append({'category': category, 'query_protein_id': query, 'eligible_queries': len(available),
                       'selection_sha256': key, 'reference_protein_id': row['reference_protein_id'] if row else None,
                       'truth_labels_json': json.dumps(row['truth_labels']) if row else None,
                       'predicted_labels_json': json.dumps(row['predicted_labels']) if row else None})
    return output


def prediction_schema():
    import pyarrow as pa
    fields = [('query_protein_id', pa.string()), ('cluster_id_30', pa.string()),
              ('level', pa.string()), ('reference_protein_id', pa.string()),
              ('modality_rank', pa.int64()), ('raw_rank', pa.int64())]
    fields += [(f, pa.float64()) for f in SCORE_FIELDS]
    fields += [('predicted_labels', pa.list_(pa.string())), ('top50_labels', pa.list_(pa.string())),
               ('top50_reference_count', pa.int64())]
    fields += [(f, pa.list_(pa.string())) for f in ('reference_activity_ids',
               'reference_evidence_tiers', 'reference_source_releases', 'reference_provenances')]
    fields += [('output_reason', pa.string())]
    return pa.schema(fields)


def self_test():
    split = {q: ('test', q + 'c') for q in ('q1', 'q2', 'q3', 'q4')}
    split.update({'r1': ('train', 'r1c'), 'r2': ('train', 'r2c')})
    library = [dict(activity_id=aid, reference_protein_id=rid, ec_l3='1.1.1',
                    ec_l4='1.1.1.1', canonical_rhea=rhea, evidence_tier='GOLD',
                    source_release='FROZEN', reference_partition='train', provenance='REF')
               for aid, rid, rhea in [('a1', 'r1', 'R1'), ('a2', 'r1', 'R2'),
                                      ('a3', 'r1', 'R1'), ('a4', 'r2', None)]]
    refs = make_library(library, split)
    require(refs['r1']['labels']['EXACT_RHEA'] == {'R1', 'R2'}, 'Deduplicate labels')
    try:
        make_library(library + [library[0]], split)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate activity ID not rejected')
    cohort = {q: split[q][1] for q in ('q1', 'q2', 'q3', 'q4')}
    ranks, clusters, candidates, top = defaultdict(dict), defaultdict(set), defaultdict(list), {}
    def hit(q, rid, rank):
        return dict(query_protein_id=q, reference_protein_id=rid, modality_rank=rank,
                    raw_rank=rank, query_partition='test', reference_partition='train',
                    reference_cluster_id_30=split[rid][1], modality='mmseqs',
                    candidate_provenance='NATIVE', **{f: 0.5 for f in SCORE_FIELDS})
    nullable_score_hit = hit('q4', 'r1', 1)
    nullable_score_hit['bits'] = None
    for row in [hit('q1', 'r1', 1), hit('q2', 'r2', 1), hit('q2', 'r1', 2), nullable_score_hit]:
        ingest_hit(row, cohort, split, ranks, clusters, candidates, top)
    try:
        ingest_hit(hit('q1', 'r1', 1), cohort, split, ranks, clusters, candidates, top)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate rank1 not rejected')
    pred = construct_predictions(cohort, refs, candidates, top)
    exact = {r['query_protein_id']: r for r in pred if r['level'] == 'EXACT_RHEA'}
    require(exact['q2']['output_reason'] == 'NO_LEVEL_ANNOTATION' and
            exact['q2']['reference_protein_id'] == 'r2' and exact['q2']['top50_labels'] == ['R1', 'R2'],
            'No rank2 fallback despite native score tie')
    require(exact['q3']['output_reason'] == 'NO_HIT', 'Missing hit preserved')
    require(exact['q4']['bits'] is None and exact['q4']['reference_protein_id'] == 'r1',
            'Missing unused score must not change native selection')
    before = json.dumps(pred, sort_keys=True).encode()
    truths = {(q, l): ({'R1'} if l == 'EXACT_RHEA' else {'1.1.1'} if l == 'EC_L3' else {'1.1.1.1'})
              for q in cohort for l in LEVELS}
    truths[('q4', 'EXACT_RHEA')] = set()
    global_labels = {l: set().union(*(v['labels'][l] for v in refs.values())) for l in LEVELS}
    result = evaluate(pred, truths, global_labels)
    no_truth = next(r for r in result if r['query_protein_id'] == 'q4' and r['level'] == 'EXACT_RHEA')
    require(no_truth['matched_label_count'] is None and no_truth['any_match'] is None, 'No truth != negative')
    require(next(r for r in result if r['query_protein_id'] == 'q1' and r['level'] == 'EXACT_RHEA')
            ['selection_state'] == 'EMITTED_PARTIAL_DOCUMENTED_MATCH', 'Partial multiset support')
    states, metrics, _ = summaries(result)
    require(all(r['queries'] == 0 and r['fraction'] is None for r in states if not r['applicable']), 'Sampling N/A')
    require(next(r for r in metrics if r['level'] == 'EXACT_RHEA' and r['metric'] ==
                 'micro_concordance_precision')['value'] == 0.5, 'Micro precision denominator')
    require(ratio(0, 0) is None, 'Zero denominator null')
    synthetic_truth_rows = [dict(protein_id=q, **{f: json.dumps(sorted(truths[(q, l)]))
                           for f, l in zip(TRUTH_FIELDS, LEVELS)}) for q in cohort]
    require(parse_truth(synthetic_truth_rows, cohort) == truths, 'Truth schema round trip')
    try:
        parse_truth(synthetic_truth_rows + [synthetic_truth_rows[0]], cohort)
    except ValueError:
        pass
    else:
        raise AssertionError('Many-to-many truth join not rejected')
    evaluate(pred, {key: set() for key in truths}, global_labels)
    after = json.dumps(construct_predictions(cohort, refs, candidates, top), sort_keys=True).encode()
    require(before == after, 'Truth permutation changes prediction bytes')
    require(len(cases(result)) == 4, 'Fixed case category census')
    print('PHASE450B_SELF_TEST_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args()
    if args.self_test:
        require(not args.preflight and args.root is None and args.contract is None, 'Isolated self-test only')
        self_test()
        return
    require(args.root is not None and args.contract is not None, 'Root and contract required')
    root = args.root.resolve()
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    require(str(root) == contract['physical_root'], 'Physical root identity mismatch')
    require(platform.python_version() == contract.get('python_version', '3.11.15') == '3.11.15', 'Frozen runtime')
    require(contract.get('cohort_n') == 27639 and contract.get('k') == 50 and
            contract.get('workflow') == WORKFLOW, 'Scientific contract mismatch')
    identity(root, contract)
    out = safe_path(root, contract['output_dir'])
    require(not out.exists(), 'B output directory already exists')
    for key in ('reservation', 'terminal'):
        target = safe_path(root, contract[key])
        require(target.is_relative_to(out) and not target.exists(), 'Existing/outside B canonical target')
    checkpoint = safe_path(root, contract['checkpoint'])
    require(checkpoint.parent == (root / 'checkpoints').resolve() and not checkpoint.exists(),
            'Checkpoint must be absent directly within project checkpoints directory')
    cohort, references, candidates, top = load_truth_free(root, contract['cohort_n'])
    if args.preflight:
        identity(root, contract)
        print(json.dumps({'status': 'PHASE450B_PREFLIGHT_PASS', 'queries': len(cohort),
                          'query_function_labels_read': False, 'output_written': False,
                          'native_queries_with_hits': len(top)}, allow_nan=False), flush=True)
        return
    require(contract.get('armed') is True, 'Formal contract is not armed')
    submission = json.loads(safe_path(root, contract['submission_reservation']).read_text(encoding='utf-8'))
    require(submission.get('contract_sha256') == digest(args.contract), 'Submission reservation mismatch')
    out.mkdir(exist_ok=False)
    dump(safe_path(root, contract['reservation']), {'status': 'STARTED', 'created_utc': utc(),
                                                  'contract_sha256': digest(args.contract)})
    checks = []
    def checked(name, detail=''):
        checks.append({'check': name, 'pass': True, 'detail': str(detail)})
    try:
        predictions = construct_predictions(cohort, references, candidates, top)
        require(len(predictions) == 82917, 'Prediction row count')
        pred_path = out / 'phase450b_native_predictions.parquet'
        write_parquet(pred_path, predictions, prediction_schema())
        pred_hash = digest(pred_path)
        dump(out / 'PREDICTIONS_FROZEN_BEFORE_QUERY_TRUTH.json',
             {'status': 'PREDICTIONS_FROZEN_TRUTH_FREE', 'created_utc': utc(),
              'prediction_file': pred_path.name, 'sha256': pred_hash, 'rows': len(predictions),
              'query_function_labels_read': False, 'allowed_input_columns': ALLOWED,
              'caveat': 'Execution-order receipt, not proof of inaccessible outcome storage.'})
        checked('prediction_frozen_before_query_truth', pred_hash)
        # First opening of query functional annotations starts strictly here.
        truth_opened_utc = utc()
        import pyarrow.parquet as pq
        truth = parse_truth(pq.read_table(root / INPUTS[2],
                            columns=['protein_id', *TRUTH_FIELDS]).to_pylist(), cohort)
        global_labels = {l: set().union(*(r['labels'][l] for r in references.values())) for l in LEVELS}
        evaluated = evaluate(predictions, truth, global_labels)
        previous = {}
        for row in pq.read_table(root / INPUTS[3], columns=['query_protein_id', 'level',
                                  'documented_truth', 'library_supported', 'mmseqs_at_50']).to_pylist():
            key = (row['query_protein_id'], row['level'])
            require(key not in previous, 'Duplicate lineage evaluation key')
            previous[key] = row
        for row in evaluated:
            old = previous[(row['query_protein_id'], row['level'])]
            require(row['documented_truth'] == old['documented_truth'] and
                    row['library_supported'] == old['library_supported'] and
                    row['retrieved_available'] == old['mmseqs_at_50'], 'Phase444 native availability mismatch')
        checked('independent_source_native_mmseqs50_flags_reconciled', len(evaluated))
        write_parquet(out / 'phase450b_query_evaluation.parquet', evaluated)
        states, metrics, sizes = summaries(evaluated)
        table(out / 'phase450b_states.tsv', states)
        table(out / 'phase450b_metrics.tsv', metrics)
        table(out / 'phase450b_output_set_sizes.tsv', sizes)
        table(out / 'phase450b_case_index.tsv', cases(evaluated))
        require(digest(pred_path) == pred_hash, 'Prediction changed after truth join')
        identity(root, contract)
        checked('inputs_unchanged_and_predictions_preserved')
        checked('cohort_state_closure_all_levels', len(cohort))
        checked('all_queries_retained_and_missing_truth_not_negative')
        checked('no_sampling_no_threshold_no_new_model')
        table(out / 'phase450b_producer_checks.tsv', checks)
        dump(out / 'PRODUCER_PASS_DESCRIPTIVE_ONLY.json',
             {'status': 'PHASE450B_PRODUCER_PASS_DESCRIPTIVE_ONLY', 'queries': len(cohort),
              'rows': len(evaluated), 'checks': len(checks), 'truth_opened_utc': truth_opened_utc,
              'prediction_sha256': pred_hash, 'inputs_unchanged': True,
              'sampling_applicable': False, 'new_threshold': False,
              'external_generalization_demonstrated': False, 'independent_audit_pending': True})
        outputs = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for p in sorted(out.iterdir()) if p.is_file()]
        dump(out / 'phase450b_producer_manifest.json', {'inputs': contract['inputs'], 'outputs': outputs})
        print('PHASE450B_PRODUCER_PASS_DESCRIPTIVE_ONLY', flush=True)
    except Exception as error:
        dump(out / 'FAILED_producer.json', {'status': 'FAILED', 'created_utc': utc(),
             'error_type': type(error).__name__, 'error': str(error), 'checks': checks})
        raise


if __name__ == '__main__':
    main()
