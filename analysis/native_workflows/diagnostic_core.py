"""Set-valued annotation diagnosis, independent of any prediction software."""

def trace(truth, library, ranked_candidates, output_labels, retained_limit):
    """Candidates are (identifier, set of labels, score), already in native order.

    Acceptance is deliberately outside this function. Scores are retained for
    inspection, never used to select a replacement prediction using truth.
    """
    truth=set(truth); library=set(library); output=set(output_labels)
    ids=[r[0] for r in ranked_candidates]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate candidate identity')
    if not all(set(labels)<=library for _,labels,_ in ranked_candidates):
        raise ValueError('candidate labels outside reference universe')
    retained=ranked_candidates[:retained_limit]
    if not output<=set().union(*(set(r[1]) for r in retained)):
        raise ValueError('output outside retained candidates')
    ranks=[i+1 for i,r in enumerate(ranked_candidates) if truth & set(r[1])]
    first=ranks[0] if ranks else None
    if not truth: state='NO_RECORDED_ACTIVITY'
    elif not truth & library: state='NO_LIBRARY_SUPPORT'
    elif first is None: state='ABSENT_FROM_RETURNED_CANDIDATES'
    elif first>retained_limit: state='RETENTION_LOSS'
    elif truth & output: state='CONCORDANT_OUTPUT'
    else: state='SELECTION_LOSS'
    return dict(state=state,evaluable=int(bool(truth)),library_match=int(bool(truth&library)),
        returned_match=int(first is not None),retained_match=int(first is not None and first<=retained_limit),
        first_matching_rank=first,output_count=len(output),matched_output_count=len(output&truth),
        truth_count=len(truth),hit=int(bool(output&truth)) if truth else None,
        has_output=int(bool(output)),acceptance_status='NOT_APPLICABLE')
