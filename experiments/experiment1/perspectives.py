"""Offline maximum-chunk-block review of new facts, refinement and perspectives.

This versioned successor leaves the immutable v1 study/extraction contracts intact.
No transport, extraction, automatic semantic decisions, or budget matching.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from . import analysis, core

VERSION = 'fact-perspective-blocks-v1'
BASELINE = 'e256_s17_b0'
ADDED = ('e256_s29_b0', 'e512_s17_b0')
INFORMATION = ('equivalent', 'refinement', 'new_claim', 'broader', 'uncertain')
ASPECTS = ('same', 'participant_role', 'action_state', 'cause_purpose_result',
           'time_condition', 'attribution', 'decomposition_aggregation', 'other', 'uncertain')
LABEL_FIELDS = ('support', 'information', 'aspects', 'baseline_matches', 'support_spans',
                'semantic_fact_id', 'annotator', 'rationale')


def load_candidates(run, policy):
    m, nodes, facts, status, raw_hash = analysis.load(run)
    strict_count = len(facts)
    if policy == 'missing-unknown':
        if m.get('version') != 'experiment1-v1':
            raise ValueError('missing-unknown requires the explicit v1 qualifier contract')
        audit = {s['job_id']: s for s in status}
        facts = {}
        for j in m['jobs']:
            s = audit[j['job_id']]
            if not s['complete']:
                continue  # Never salvage malformed or truncated generations.
            record = core.read(Path(run) / 'raw' / (j['job_id'] + '.json'))
            derived = copy.deepcopy(record)
            obj = json.loads(derived['response']['choices'][0]['message']['content'])
            missing = {}
            for i, r in enumerate(obj['relations']):
                if isinstance(r, dict):
                    missing[i] = [k for k in ('time', 'condition') if k not in r]
                    for k in missing[i]:
                        r[k] = ''  # Validator adapter only; restored to None below.
            derived['response']['choices'][0]['message']['content'] = json.dumps(obj)
            parsed = core.parse_record(j, derived)
            for f in parsed['relations']:
                index = int(f['occurrence_id'].rsplit(':', 1)[1])
                context = {k: s[k] for k in ('doc_id', 'region_id', 'extraction_condition', 'job_id',
                           'chunk_size', 'seed', 'boundary', 'chunk_start', 'chunk_end', 'raw_path', 'raw_sha256')}
                f.update(context, missing_qualifiers=missing[index])
                for k in missing[index]:
                    f[k] = None  # Unknown is never equivalent to explicitly absent (empty string).
                facts[f['occurrence_id']] = f
    elif policy != 'strict':
        raise ValueError('Unknown qualifier policy')
    for f in facts.values():
        f.setdefault('missing_qualifiers', [])
    return m, nodes, facts, status, raw_hash, strict_count


def maximum_blocks(m, conditions, status):
    """Use actual largest-chunk boundaries; never cut/copy a crossing small chunk."""
    audit = {s['job_id']: s for s in status}
    blocks = []
    for r in m['regions']:
        jobs = [j for j in m['jobs'] if j['region_id'] == r['region_id']]
        maximum = max(j['size'] for j in jobs)
        anchors = [j for j in jobs if j['size'] == maximum and j.get('boundary', 0) == 0]
        if not anchors:
            raise ValueError('Maximum-size boundary=0 chunks required for a fixed reference grid')
        anchor_condition = sorted({core.condition(j) for j in anchors})[0]
        anchors = sorted((j for j in anchors if core.condition(j) == anchor_condition), key=lambda j: j['char_start'])
        for anchor in anchors:
            lo, hi = anchor['char_start'], anchor['char_end']
            states = {}
            for c in conditions:
                overlap = sorted((j for j in jobs if core.condition(j) == c and
                                  j['char_start'] < hi and j['char_end'] > lo), key=lambda j: j['char_start'])
                contained = [j for j in overlap if lo <= j['char_start'] and j['char_end'] <= hi]
                cursor = lo
                token_cursor = anchor['token_start']
                aligned = bool(contained) and len(contained) == len(overlap)
                for j in contained:
                    aligned = aligned and j['char_start'] == cursor and j['token_start'] == token_cursor
                    cursor, token_cursor = j['char_end'], j['token_end']
                aligned = aligned and cursor == hi and token_cursor == anchor['token_end']
                rows = [audit[j['job_id']] for j in contained]
                states[c] = dict(aligned=aligned, complete=aligned and all(s['complete'] for s in rows),
                                 clean=aligned and all(s['complete'] and not s['errors'] for s in rows),
                                 contained_job_ids=[j['job_id'] for j in contained],
                                 crossing_job_ids=[j['job_id'] for j in overlap if j not in contained],
                                 failed_job_ids=[s['job_id'] for s in rows if not s['complete']])
            offset = lo - r['char_start']
            blocks.append(dict(block_id=core.digest([r['region_id'], lo, hi])[:24],
                               parent_region_id=r['region_id'], doc_id=r['doc_id'],
                               char_start=lo, char_end=hi, token_start=anchor['token_start'],
                               token_end=anchor['token_end'], maximum_chunk_size=maximum,
                               short_tail=anchor['token_end'] - anchor['token_start'] < maximum,
                               text=r['text'][offset:offset + hi - lo], conditions=states,
                               common_complete=all(s['complete'] for s in states.values())))
    return blocks


def claims_for(facts, nodes, ids):
    groups = {}
    for oid, f in facts.items():
        if f['job_id'] not in ids:
            continue
        n, t = nodes[f['source_occurrence']], nodes[f['target_occurrence']]
        identity = [analysis.literal_node(n), core.norm(f['predicate']), analysis.literal_node(t),
                    f['polarity'], f['time'], f['condition'], f['missing_qualifiers']]
        key = core.digest(identity)
        item = groups.setdefault(key, dict(id=key, source=n['name'], source_type=n['type'],
            predicate=f['predicate'], target=t['name'], target_type=t['type'], polarity=f['polarity'],
            time=f['time'], condition=f['condition'], missing_qualifiers=f['missing_qualifiers'],
            evidence=[], occurrences=[]))
        item['occurrences'].append(oid)
        for span in f['spans']:
            ev = dict(quote=f['evidence'], span=span, ambiguous=len(f['spans']) != 1)
            if ev not in item['evidence']:
                item['evidence'].append(ev)
    return groups


def public_claims(claims):
    return [{k: v for k, v in claims[eid].items() if k != 'occurrences'} for eid in sorted(claims)]


def immutable_payload(row):
    return {k: v for k, v in row.items() if k not in LABEL_FIELDS}


def analyze(run, out, baseline=BASELINE, added=ADDED, cohort_conditions=None,
            per_stratum=20, qualifier_policy='strict'):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh output directory; original snapshots are immutable')
    if per_stratum < 1:
        raise ValueError('per_stratum must be positive')
    added = list(added)
    if not added or baseline in added or len(set(added)) != len(added):
        raise ValueError('Added conditions must be unique and different from baseline')
    m, nodes, facts, status, raw_hash, strict_count = load_candidates(run, qualifier_policy)
    conditions = list(cohort_conditions or [baseline] + added)
    available = {core.condition(j) for j in m['jobs']}
    if (len(set(conditions)) != len(conditions) or not set([baseline] + added) <= set(conditions)
            or not set(conditions) <= available):
        raise ValueError('Cohort must include every compared condition and exist in the manifest')
    blocks = maximum_blocks(m, conditions, status)
    protocol = dict(version=VERSION, baseline=baseline, added=added, cohort_conditions=conditions,
                    comparison_unit='actual_maximum_chunk', qualifier_policy=qualifier_policy,
                    per_stratum=per_stratum, sampling_seed=17,
                    code_sha256=core.sha(Path(__file__).read_text()),
                    dependencies={n: core.sha(Path(__file__).with_name(n).read_text()) for n in ('core.py', 'analysis.py')})
    input_hash = core.digest([core.digest(m), raw_hash])
    measurement_hash = core.digest([input_hash, protocol])
    review, keys, catalog, comparisons = [], [], {}, []
    rng = random.Random(17)
    for block in blocks:
        graphs = {c: claims_for(facts, nodes, set(block['conditions'][c]['contained_job_ids'])) for c in [baseline] + added}
        a = graphs[baseline]
        for c in added:
            b = graphs[c]
            eligible = block['common_complete']
            comparisons.append(dict(block_id=block['block_id'], doc_id=block['doc_id'], baseline=baseline,
                added=c, common_complete=eligible, baseline_candidates=len(a), added_candidates=len(b),
                literal_shared=len(a.keys() & b.keys()), literal_added_only=len(b.keys() - a.keys()),
                literal_jaccard=analysis.jaccard(set(a), set(b)),
                validity='candidate_only', partial_context=not all(block['conditions'][x]['aligned'] for x in (baseline, c))))
            if not eligible:
                continue
            ref = core.digest([measurement_hash, block['block_id'], baseline])[:24]
            catalog[ref] = public_claims(a)
            for stratum, pool in [('literal_added_only', b.keys() - a.keys()), ('literal_shared', b.keys() & a.keys())]:
                sampled = rng.sample(sorted(pool), min(per_stratum, len(pool)))
                for eid in sampled:
                    rid = core.digest([measurement_hash, block['block_id'], baseline, c, eid])[:24]
                    review.append(dict(id=rid, block_id=block['block_id'], doc_id=block['doc_id'],
                        claim=public_claims({eid: b[eid]})[0], baseline_ref=ref,
                        support=None, information=None, aspects=[], baseline_matches=[], support_spans=[],
                        semantic_fact_id='', annotator='', rationale=''))
                    keys.append(dict(id=rid, block_id=block['block_id'], doc_id=block['doc_id'], baseline=baseline,
                        added=c, stratum=stratum, population=len(pool), sampled=len(sampled),
                        inclusion_probability=len(sampled) / len(pool), occurrences=b[eid]['occurrences']))
    rng.shuffle(review)
    # The annotation file contains no condition/seed/stratum labels; content can still reveal context.
    envelope = dict(version=VERSION, input_hash=input_hash, measurement_hash=measurement_hash,
                    document_deduplication=dict(complete=False, reviewer='', rationale=''), items=review)
    out.mkdir(parents=True)
    core.save(out / 'protocol.json', protocol)
    core.save(out / 'blocks.json', blocks)
    core.save(out / 'review_blocks.json', [{k: v for k, v in b.items() if k not in ('conditions', 'common_complete', 'maximum_chunk_size')} for b in blocks])
    core.save(out / 'comparisons.json', comparisons)
    core.save(out / 'review_baselines.json', catalog)
    core.save(out / 'review_key.json', keys)
    core.save(out / 'review_blind.json', envelope)
    core.save(out / 'call_audit.json', status)
    core.save(out / 'occurrences.json', dict(entities=nodes, relations=facts))
    summary = dict(version=VERSION, input_hash=input_hash, measurement_hash=measurement_hash, protocol=protocol,
        planned_blocks=len(blocks), common_complete_blocks=sum(b['common_complete'] for b in blocks),
        excluded_block_ids=[b['block_id'] for b in blocks if not b['common_complete']],
        common_complete_characters=sum(b['char_end'] - b['char_start'] for b in blocks if b['common_complete']),
        planned_characters=sum(b['char_end'] - b['char_start'] for b in blocks),
        recorded_calls=sum(s['recorded'] for s in status), total_calls=len(status),
        strict_relation_occurrences=strict_count, candidate_relation_occurrences=len(facts),
        recovered_missing_qualifier_candidates=len(facts) - strict_count,
        review_items=len(review), measurement_status='awaiting_human_review',
        supported_new_facts=None, supported_refinements=None, supported_perspectives=None,
        costs_reference_only={c: analysis.cost([s for s in status if s['extraction_condition'] == c], None) for c in conditions},
        interpretation=['All planned blocks remain in comparisons, including failures and boundary exclusions.',
            'Review estimates describe only the common-success cohort; they are not full-document results.',
            'Literal candidate counts are not semantic fact counts. Costs are reference only.',
            'Reanalysis of existing data is exploratory; freeze this protocol before new evaluation.',
            'Within-document blocks are not independent documents. No significance or QA claim.'])
    core.save(out / 'summary.json', summary)
    report = [
        '# Fact and perspective review — ' + VERSION,
        '',
        'Question: do chunk sizes reveal new claims, refinements, or different supported aspects?',
        '',
        f"Planned maximum-chunk blocks: {len(blocks)}; common generation-complete blocks: {summary['common_complete_blocks']}.",
        f"Review items: {len(review)}. Semantic judgments are pending; candidate counts are not validated facts.",
        f"Qualifier policy: {qualifier_policy}. Recovered candidates: {len(facts) - strict_count}.",
        '',
        '| Condition | Calls | Generation complete | Error-free calls |',
        '| --- | ---: | ---: | ---: |',
    ]
    for c in conditions:
        rows = [r for r in status if r['extraction_condition'] == c]
        report.append(f"| {c} | {len(rows)} | {sum(r['complete'] for r in rows)} | {sum(r['complete'] and not r['errors'] for r in rows)} |")
    report += ['', 'Counts above retain the original strict validation audit, even under missing-unknown sensitivity.',
               '', 'Copy review_blind.json to a separate annotations file. Consult review_blocks.json and review_baselines.json.',
               'Review support, information, and aspects separately. One claim may match multiple baseline claims.',
               'Only a complete census with document-wide class review permits unique fact counts.',
               'All planned blocks remain in comparisons.json; semantic estimates concern only the common-success cohort.',
               'Source data are read-only. No extraction calls. Costs are reference-only metadata.']
    (out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    core.save(out / 'analysis_integrity.json', {p.name: core.sha(p.read_text()) for p in sorted(out.glob('*.json'))})
    return summary


def verify_analysis(root):
    hashes = core.read(root / 'analysis_integrity.json')
    for name, expected in hashes.items():
        if Path(name).name != name or core.sha((root / name).read_text()) != expected:
            raise ValueError('Analysis artifact changed: ' + name)


def evaluate(root, annotations, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Evaluation output already exists')
    verify_analysis(root)
    template = core.read(root / 'review_blind.json')
    submitted = core.read(annotations)
    for k in ('version', 'input_hash', 'measurement_hash'):
        if submitted.get(k) != template[k]:
            raise ValueError('Annotation snapshot/version mismatch')
    original = {r['id']: r for r in template['items']}
    baselines = core.read(root / 'review_baselines.json')
    blocks = {b['block_id']: b for b in core.read(root / 'blocks.json')}
    labels, seen = {}, set()
    for row in submitted['items']:
        rid = row['id']
        if rid in seen or rid not in original or immutable_payload(row) != immutable_payload(original[rid]):
            raise ValueError('Unknown/duplicate id or changed review content')
        seen.add(rid)
        support, info, aspects = row.get('support'), row.get('information'), row.get('aspects', [])
        if support not in (None, 'supported', 'unsupported', 'uncertain') or info not in (None,) + INFORMATION:
            raise ValueError('Invalid support/information label')
        if not isinstance(aspects, list) or any(x not in ASPECTS for x in aspects) or len(set(aspects)) != len(aspects):
            raise ValueError('Invalid aspects')
        if ('same' in aspects or 'uncertain' in aspects) and len(aspects) != 1:
            raise ValueError('same/uncertain cannot be mixed with other aspects')
        ids = {c['id'] for c in baselines[row['baseline_ref']]}
        matches = row.get('baseline_matches', [])
        if not isinstance(matches, list) or len(matches) != len(set(matches)) or not set(matches) <= ids:
            raise ValueError('Baseline matches must refer to this block baseline catalog')
        spans = row.get('support_spans', [])
        if not isinstance(spans, list):
            raise ValueError('support_spans must be a list')
        block = blocks[row['block_id']]
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span)
                    or not block['char_start'] <= span[0] < span[1] <= block['char_end']):
                raise ValueError('Support evidence must be an absolute span inside this block')
        for k in ('semantic_fact_id', 'annotator', 'rationale'):
            if not isinstance(row.get(k, ''), str):
                raise ValueError(k + ' must be a string')
        if support == 'supported' and info in ('equivalent', 'refinement', 'broader') and not matches:
            raise ValueError('Equivalent/refinement/broader judgments need baseline matches (one or many)')
        if support == 'supported' and aspects and aspects not in (['same'], ['uncertain']) and not matches:
            raise ValueError('A different perspective needs related baseline claims, not just a new unrelated fact')
        if support == 'supported' and info == 'equivalent' and aspects and aspects != ['same']:
            raise ValueError('Equivalent claims cannot manufacture a different perspective')
        resolved = support in ('supported', 'unsupported') and bool(row.get('annotator', '').strip()) and bool(row.get('rationale', '').strip())
        if support == 'supported':
            resolved = (resolved and info not in (None, 'uncertain') and bool(aspects) and 'uncertain' not in aspects
                        and bool(spans) and bool(row.get('semantic_fact_id', '').strip()))
        if resolved:
            labels[rid] = row
    summary = core.read(root / 'summary.json')
    keys = core.read(root / 'review_key.json')
    all_comps = core.read(root / 'comparisons.json')
    dedup = submitted.get('document_deduplication', {})
    if type(dedup.get('complete', False)) is not bool:
        raise ValueError('Deduplication complete must be boolean')
    dedup_confirmed = (dedup.get('complete') is True and isinstance(dedup.get('reviewer'), str)
                       and bool(dedup['reviewer'].strip()) and isinstance(dedup.get('rationale'), str)
                       and bool(dedup['rationale'].strip()))
    if dedup.get('complete') and not dedup_confirmed:
        raise ValueError('Document deduplication needs reviewer and rationale')
    for added in summary['protocol']['added']:
        class_labels = {}
        for k in keys:
            row = labels.get(k['id'])
            if k['added'] != added or row is None or row['support'] != 'supported':
                continue
            identity = (k['doc_id'], k['block_id'], row['semantic_fact_id'].strip())
            previous = class_labels.setdefault(identity, row['information'])
            if previous != row['information']:
                raise ValueError('Same semantic fact class has inconsistent information labels in a block')
    metrics = ('supported', 'new_claim', 'refinement', 'broader', 'equivalent', 'different_perspective', 'unsupported')
    results = []
    for added in summary['protocol']['added']:
        comparisons = [c for c in all_comps if c['added'] == added and c['common_complete']]
        ks = [k for k in keys if k['added'] == added]
        documents = sorted({c['doc_id'] for c in comparisons})
        by_doc = {}
        for doc in documents:
            dk = [k for k in ks if k['doc_id'] == doc]
            complete = all(k['id'] in labels for k in dk)
            census = complete and all(k['inclusion_probability'] == 1 for k in dk)
            counts, estimates, aspects_counts, fact_classes = Counter(), Counter(), Counter(), defaultdict(set)
            for k in dk:
                row = labels.get(k['id'])
                if row is None:
                    continue
                supported = row['support'] == 'supported'
                active = ['supported', row['information']] if supported else ['unsupported']
                if supported and row['aspects'] != ['same']:
                    active.append('different_perspective')
                for metric in active:
                    counts[metric] += 1
                    estimates[metric] += 1 / k['inclusion_probability']
                    if supported:
                        fact_classes[metric].add(row['semantic_fact_id'].strip())
                if supported:
                    aspects_counts.update(row['aspects'])
            pool = {(k['block_id'], k['stratum']): k['population'] for k in dk}
            by_doc[doc] = dict(sampled=len(dk), reviewed=sum(k['id'] in labels for k in dk),
                candidate_population=sum(pool.values()), review_complete=complete, census=census,
                observed_candidate_counts={k: counts[k] for k in metrics},
                estimated_candidate_counts={k: estimates[k] for k in metrics} if complete else None,
                observed_aspect_counts=dict(aspects_counts),
                observed_reviewed_fact_classes={k: len(v) for k, v in fact_classes.items()},
                unique_supported_facts=len(fact_classes['supported']) if census and dedup_confirmed else None,
                unique_fact_classes_with_block_local_new_label=len(fact_classes['new_claim']) if census and dedup_confirmed else None)
        finished = bool(documents) and all(v['review_complete'] for v in by_doc.values())
        results.append(dict(baseline=summary['protocol']['baseline'], added=added,
            planned_blocks=summary['planned_blocks'], common_complete_blocks=len(comparisons),
            review_complete=finished, by_document=by_doc,
            document_macro_candidate_counts={k: sum(v['estimated_candidate_counts'][k] for v in by_doc.values()) / len(by_doc)
                for k in metrics} if finished else None))
    result = dict(version=VERSION, input_hash=summary['input_hash'], measurement_hash=summary['measurement_hash'],
        annotations_sha256=core.sha(Path(annotations).read_text()), comparisons=results,
        notes=['New-claim/refinement and perspective axes overlap; do not sum them as disjoint facts.',
               'HT estimates count labeled candidate edges in common-success blocks, not unique semantic facts.',
               'Unique fact counts require a full census and explicit document-wide deduplication review.',
               'Newness is relative to this block baseline; document-global novelty remains unmeasured.',
               'Unresolved judgments never become false; incomplete reviews do not yield population estimates.',
               'No complete blocks means unavailable, not zero. Costs do not enter the outcome.',
               'Document macro means are descriptive; no block-independent CI or QA-performance claim.'])
    core.save(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('analyze')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--baseline', default=BASELINE)
    q.add_argument('--added', nargs='+', default=list(ADDED))
    q.add_argument('--cohort-conditions', nargs='+')
    q.add_argument('--per-stratum', type=int, default=20)
    q.add_argument('--qualifier-policy', choices=('strict', 'missing-unknown'), default='strict')
    q = sub.add_parser('evaluate')
    q.add_argument('--analysis', type=Path, required=True)
    q.add_argument('--annotations', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'analyze':
        result = analyze(a.run, a.out, a.baseline, a.added, a.cohort_conditions, a.per_stratum, a.qualifier_policy)
    else:
        result = evaluate(a.analysis, a.annotations, a.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
