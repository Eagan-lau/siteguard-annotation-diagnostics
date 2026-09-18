#!/usr/bin/env python3
"""Exact, hash-locked correction of one provenance-label assertion."""
import hashlib
from pathlib import Path

path=Path(__file__).with_name('phase444_candidate_lineage.py')
raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='493dcb74366e748f19dbe42275a342a555f4b7ea6d61723289f6bc65c22d39bd'
before=b"frame.pair_set.eq('population_atlas').all()"
after=b"frame.pair_set.eq('population').all()"
assert raw.count(before)==1 and raw.count(after)==0
patched=raw.replace(before,after)
assert patched.replace(after,before)==raw
exec(compile(patched,str(path)+'::phase444_1_label_only','exec'),{'__name__':'__main__','__file__':str(path)})
