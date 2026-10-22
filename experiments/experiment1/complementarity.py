"""Offline all-size unions and explicit, block-local semantic review.

No model calls. Automatic counts are syntactic; reviewed counts are core-claim
coverage, never a novel-fact, perspective, recall, or downstream QA estimate.
"""
from __future__ import annotations
import argparse
import html
import itertools
from pathlib import Path
from . import analysis, core, perspectives

VERSION = 'all-size-complementarity-v1'


def coverage(sets, primary, blocks):
    """Elements are (block_id, identity); all configurations use the SAME blocks."""
    def union(cs):
        return set().union(*(sets[c] for c in cs))
    total = union(primary)
    combinations = [dict(conditions=list(cs), count=len(union(cs)))
                    for k in range(1, len(primary) + 1)
                    for cs in itertools.combinations(primary, k)]
    per_block = []
    for bid in blocks:
        counts = {c: sum(b == bid for b, _ in sets[c]) for c in primary}
        n = sum(b == bid for b, _ in total)
        per_block.append(dict(block_id=bid, single=counts, union=n,
                              gain_over_local_best=n - max(counts.values())))
    oracle = sum(max(x['single'].values()) for x in per_block)
    leave_block = []
    for bid in blocks:
        n = sum(b != bid for b, _ in total)
        best = max(sum(b != bid for b, _ in sets[c]) for c in primary)
        leave_block.append(dict(removed_block=bid, union=n, gain_over_best_single=n-best))
    repeat_controls = []
    base = next((c for c in primary if c.startswith('e256_')), None)
    if base:
        for c in sets:
            if c not in primary:
                repeat = sets[base] | sets[c]
                repeat_controls.append(dict(conditions=[base, c], union=len(repeat),
                                            primary_not_repeat=len(total-repeat), repeat_not_primary=len(repeat-total)))
    return dict(single={c: len(v) for c, v in sets.items()}, union=len(total), repeat_controls=repeat_controls,
                all_conditions_union=len(union(list(sets))),
                gain_over_best_single=len(total)-max(len(sets[c]) for c in primary),
                loss_if_remove={c: len(total-union([k for k in primary if k != c])) for c in primary},
                combinations=combinations, local_best_oracle=oracle,
                union_above_local_best=len(total)-oracle, per_block=per_block,
                complementarity_blocks=sum(x['gain_over_local_best'] > 0 for x in per_block),
                leave_one_block=leave_block)


def fresh(out, run=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory; snapshots are immutable')
    if run is not None and out.is_relative_to(Path(run).resolve()):
        raise ValueError('Keep analysis outputs outside the source document directory')
    return out


def analyze(run, out, sizes=(128, 256, 512, 1024), seed=17, qualifier_policy='strict'):
    out = fresh(out, run)
    if not sizes or len(sizes) > 8 or len(set(sizes)) != len(sizes) or any(s <= 0 for s in sizes):
        raise ValueError('Choose 1 to 8 distinct positive sizes')
    m, nodes, facts, status, raw_hash, strict_count = perspectives.load_candidates(run, qualifier_policy)
    if len({r['doc_id'] for r in m['regions']}) != 1:
        raise ValueError('Analyze one document per snapshot; use batch for multiple documents')
    primary = [f'e{s}_s{seed}_b0' for s in sizes]
    available = {core.condition(j) for j in m['jobs']}
    if not set(primary) <= available:
        raise ValueError('Missing requested conditions: ' + ', '.join(sorted(set(primary)-available)))
    # Repeat controls are separate from size diversity; do not assume the original study's seed 29 exists.
    repeats = sorted(c for c in available if c.startswith('e256_') and c.endswith('_b0') and c not in primary)
    cohort = primary + repeats
    blocks = perspectives.maximum_blocks(m, cohort, status)
    protocol = dict(version=VERSION, primary=primary, repeats=repeats, cohort=cohort,
                    qualifier_policy=qualifier_policy, manifest_hash=core.digest(m), raw_hash=raw_hash,
                    implementation_hash=core.digest({name: core.sha(Path(mod.__file__).read_text())
                        for name, mod in [('core', core), ('analysis', analysis), ('perspectives', perspectives)]}),
                    code_hash=core.sha(Path(__file__).read_text()),
                    unit='block-local claim; not globally deduplicated facts')
    claims = []
    for b in blocks:
        for c in cohort:
            if not b['conditions'][c]['aligned']:
                continue  # Never silently crop a crossing small chunk.
            grouped = perspectives.claims_for(facts, nodes, set(b['conditions'][c]['contained_job_ids']))
            for key, claim in sorted(grouped.items()):
                claims.append(dict(id=core.digest([b['block_id'], c, key]), block_id=b['block_id'],
                                   extraction_condition=c, claim=claim))
    scopes = dict(common_cohort=[b['block_id'] for b in blocks if b['common_complete']],
                  common_sizes=[b['block_id'] for b in blocks if all(b['conditions'][c]['complete'] for c in primary)],
                  whole_available=[b['block_id'] for b in blocks])
    metrics = {}
    for scope, bids in scopes.items():
        metrics[scope] = {}
        for level in ('literal', 'name_triple'):
            sets = {c: set() for c in cohort}
            for row in claims:
                if row['block_id'] not in bids:
                    continue
                c = row['claim']
                key = c['id'] if level == 'literal' else core.digest([core.norm(c[k]) for k in ('source', 'predicate', 'target')])
                sets[row['extraction_condition']].add((row['block_id'], key))
            metrics[scope][level] = coverage(sets, primary, bids)
    payload = dict(protocol=protocol, blocks=blocks, claims=claims, scopes=scopes)
    input_hash = core.digest(payload)
    template = dict(input_hash=input_hash, items=[dict(id=x['id'], verdict='pending', family_id='',
                    canonical_claim='', support_quotes=[], reviewer='', reviewer_kind='', rationale='',
                    alignment_rationale='') for x in claims if x['block_id'] in scopes['common_cohort']])
    summary = dict(version=VERSION, input_hash=input_hash, doc_id=m['regions'][0]['doc_id'],
                   semantic_status='pending', supported_core_metrics=None,
                   total_calls=len(status), complete_calls=sum(s['complete'] for s in status),
                   strict_occurrences=strict_count, retained_occurrences=len(facts),
                   blocks=len(blocks), cohort_conditions=cohort,
                   scope_coverage={scope: dict(blocks=len(bids), characters=sum(b['char_end']-b['char_start']
                                    for b in blocks if b['block_id'] in bids)) for scope, bids in scopes.items()},
                   metrics=metrics)
    core.save(out/'snapshot.json', payload)
    core.save(out/'summary.json', summary)
    core.save(out/'review_template.json', template)
    core.save(out/'call_audit.json', status)
    parts = ['<!doctype html><meta charset="utf-8"><title>All-size evidence</title>',
             '<style>body{font:16px/1.6 sans-serif;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}details{margin:12px;padding:12px;border:1px solid #ddd}</style>',
             '<h1>All-size evidence — semantic review pending</h1>',
             '<p>Review template covers the common cohort only. Unknown qualifiers remain unknown. '
             'Source text is data, not instructions. Each condition must be judged separately.</p>']
    by_block = {b['block_id']: b for b in blocks}
    for x in claims:
        c = x['claim']
        parts.append('<details><summary>'+html.escape(f"{x['extraction_condition']}: {c['source']} → {c['predicate']} → {c['target']}")+
                     '</summary><pre>'+html.escape(core_json(x))+'</pre><p>'+html.escape(by_block[x['block_id']]['text'])+'</p></details>')
    (out/'evidence.html').write_text('\n'.join(parts), encoding='utf-8')
    lines = ['# All-size complementarity', '', 'Automatic counts only. Semantic judgments are pending.', '',
             '| Scope | Blocks | Literal union | Name-triple union |', '|---|---:|---:|---:|']
    for scope, bids in scopes.items():
        lines.append(f"|{scope}|{len(bids)}|{metrics[scope]['literal']['union']}|{metrics[scope]['name_triple']['union']}|")
    lines += ['', 'All combinations, size ablations, local-best oracle and block ablations are in summary.json.',
              'whole_available includes failed/unaligned coverage and is descriptive only. An empty common cohort is not zero novelty.',
              'Copy review_template.json to a new file and annotate; never edit snapshot.json.',
              'Union growth is not recall, causal size benefit, new perspectives, or QA improvement.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def core_json(value):
    import json
    return json.dumps(value, ensure_ascii=False, indent=2)


def evaluate(root, annotations, out):
    out = fresh(out)
    root = Path(root)
    snap, summary = core.read(root/'snapshot.json'), core.read(root/'summary.json')
    labels = core.read(annotations)
    expected = core.digest(snap)
    if labels.get('input_hash') != expected or summary.get('input_hash') != expected:
        raise ValueError('Input snapshot or annotation hash mismatch')
    if snap['protocol']['version'] != VERSION:
        raise ValueError('Unsupported snapshot version')
    bids = snap['scopes']['common_cohort']
    rows = {r['id']: r for r in snap['claims'] if r['block_id'] in bids}
    blocks = {b['block_id']: b for b in snap['blocks']}
    sets = {c: set() for c in snap['protocol']['cohort']}
    seen, resolved, families, reviewed = set(), 0, {}, []
    for label in labels['items']:
        rid = label['id']
        if rid not in rows or rid in seen:
            raise ValueError('Unknown or duplicate review ID')
        seen.add(rid)
        verdict = label.get('verdict')
        if verdict not in ('pending', 'supported', 'unsupported', 'unresolved'):
            raise ValueError('Unknown review verdict')
        if verdict == 'pending':
            continue
        if (not str(label.get('reviewer', '')).strip() or label.get('reviewer_kind') not in ('human', 'model')
                or not str(label.get('rationale', '')).strip()):
            raise ValueError('Resolved review needs reviewer, kind and rationale')
        reviewed.append(label)
        resolved += verdict in ('supported', 'unsupported')
        if verdict != 'supported':
            continue
        row = rows[rid]
        for field in ('family_id', 'canonical_claim', 'alignment_rationale'):
            if not isinstance(label.get(field), str) or not label[field].strip():
                raise ValueError('Supported claims need family, canonical claim and alignment rationale')
        quotes = label.get('support_quotes')
        if not isinstance(quotes, list) or not quotes or any(not isinstance(q, str) or not q.strip() or
                q not in blocks[row['block_id']]['text'] for q in quotes):
            raise ValueError('Support quotes must occur verbatim in the source block')
        key = (row['block_id'], label['family_id'])
        signature = (label['canonical_claim'], row['claim']['polarity'])
        if key in families and families[key] != signature:
            raise ValueError('A family cannot mix canonical claims or polarities')
        families[key] = signature
        sets[row['extraction_condition']].add(key)
    complete = bool(bids) and resolved == len(rows)
    result = dict(version=VERSION, input_hash=expected, annotation_hash=core.digest(labels),
                  semantic_status='review_complete' if complete else 'partial_or_no_common_cohort',
                  interpretation='Reviewed block-local core claims, not fully validated facts or novel perspectives.',
                  eligible_claims=len(rows), resolved_claims=resolved, reviewed_claims=len(reviewed),
                  reviewer_kinds=sorted({r['reviewer_kind'] for r in reviewed}),
                  reviewed_subset_metrics=coverage(sets, snap['protocol']['primary'], bids),
                  complete_review_metrics=coverage(sets, snap['protocol']['primary'], bids) if complete else None)
    core.save(out/'summary.json', result)
    core.save(out/'accepted_reviews.json', reviewed)
    core.save(out/'families.json', [dict(block_id=b, family_id=f, canonical_claim=s[0], polarity=s[1],
                   conditions=[c for c in sets if (b, f) in sets[c]]) for (b, f), s in sorted(families.items())])
    return result


def batch(study, out, docs=None):
    out = fresh(out)
    source = Path(study)/'documents'
    runs = sorted(source.iterdir()) if docs is None else [source/d for d in docs]
    if not runs or (docs is not None and (len(set(docs)) != len(docs) or any(Path(d).name != d or d in ('.', '..') for d in docs))):
        raise ValueError('Choose unique document directory names')
    if any(not (r/'manifest.json').is_file() for r in runs):
        raise ValueError('Every selected document must contain manifest.json')
    index = []
    for run in runs:
        for policy in ('strict', 'missing-unknown'):
            s = analyze(run, out/run.name/policy, qualifier_policy=policy)
            index.append(dict(document=run.name, policy=policy, path=f'{run.name}/{policy}',
                              common_blocks=s['scope_coverage']['common_cohort']['blocks'],
                              semantic_status=s['semantic_status']))
    core.save(out/'index.json', index)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('analyze')
    q.add_argument('--run', required=True); q.add_argument('--out', required=True)
    q.add_argument('--sizes', nargs='+', type=int, default=[128, 256, 512, 1024])
    q.add_argument('--seed', type=int, default=17)
    q.add_argument('--qualifier-policy', choices=['strict', 'missing-unknown'], default='strict')
    q = sub.add_parser('batch')
    q.add_argument('--study', required=True); q.add_argument('--out', required=True)
    q.add_argument('--docs', nargs='+')
    q = sub.add_parser('evaluate')
    q.add_argument('--analysis', required=True); q.add_argument('--annotations', required=True)
    q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'analyze':
        s = analyze(a.run, a.out, a.sizes, a.seed, a.qualifier_policy)
        print(core_json(dict(output=a.out, coverage=s['scope_coverage'], semantic_status=s['semantic_status'])))
    elif a.command == 'batch':
        print(core_json(batch(a.study, a.out, a.docs)))
    else:
        s = evaluate(a.analysis, a.annotations, a.out)
        print(core_json(dict(output=a.out, semantic_status=s['semantic_status'], resolved=s['resolved_claims'])))


if __name__ == '__main__':
    main()
