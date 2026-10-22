"""Prepare/extract one additional development document using the frozen study settings."""
import argparse
import json
from pathlib import Path
from . import core, study


def run(root, doc_id, limit=None, prepare_only=False, transport=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    with study.locked(root):
        s = study.load_study(root)  # Includes extraction implementation/source hashes.
        doc, folder = study.document(root, s, doc_id)
        if doc['split'] != 'development':
            raise ValueError('Use the existing frozen evaluation workflow for evaluation documents')
        m = study.ensure_prepared(root, s, doc)
        attempted = 0 if prepare_only else core.extract(folder, transport or core.post, limit=limit)
        rows = [core.read(p) for j in m['jobs']
                if (p := folder / 'raw' / (j['job_id'] + '.json')).exists()]
        return dict(doc_id=doc_id, run=str(folder), conditions=m['conditions'], planned_calls=len(m['jobs']),
                    recorded_calls=len(rows), new_attempts=attempted, prepare_only=prepare_only,
                    extraction_protocol_unchanged=True,
                    note='Recorded includes failed/started calls; no hidden retries. Re-run to attempt remaining jobs.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--doc', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.study, a.doc, a.limit, a.prepare_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
