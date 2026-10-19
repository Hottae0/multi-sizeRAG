"""Post-extraction alignment, independent graphs, paired complementarity and review."""
from __future__ import annotations
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from .core import condition, digest, norm, parse_record, read, save, sha, validate_manifest


def load(run):
    run = Path(run)
    m = validate_manifest(read(run / 'manifest.json'))
    legacy = m.get('version') != 'experiment1-v1'
    nodes, facts, status, raw_hashes = {}, {}, [], {}
    for j in m['jobs']:
        p = run / 'raw' / (j['job_id'] + '.json')
        r = read(p) if p.exists() else None
        if r is not None:
            if r.get('job') != j:
                raise ValueError('Raw job differs from manifest: ' + j['job_id'])
            if not legacy and r.get('manifest_hash') != digest(m):
                raise ValueError('Raw protocol differs from manifest')
            raw_hashes[j['job_id']] = sha(p.read_text())
        parsed = parse_record(j, r, legacy)
        context = dict(doc_id=j['doc_id'], region_id=j['region_id'], extraction_condition=condition(j),
                       job_id=j['job_id'], chunk_size=j['size'], seed=j['seed'],
                       boundary=j.get('boundary', 0), chunk_start=j['char_start'], chunk_end=j['char_end'],
                       raw_path=str(p.resolve()), raw_sha256=raw_hashes.get(j['job_id']))
        for n in parsed.pop('entities'):
            nodes[n['occurrence_id']] = dict(n, **context)
        for f in parsed.pop('relations'):
            facts[f['occurrence_id']] = dict(f, **context)
        usage = (r or {}).get('response', {}).get('usage') or {}
        valid_usage = all(isinstance(usage.get(k), (int, float)) and usage[k] >= 0
                          for k in ('prompt_tokens', 'completion_tokens'))
        status.append(dict(context, **parsed, recorded=r is not None,
                           elapsed_seconds=(r or {}).get('elapsed_seconds'),
                           prompt_tokens=usage.get('prompt_tokens') if valid_usage else None,
                           completion_tokens=usage.get('completion_tokens') if valid_usage else None,
                           usage_known=valid_usage))
    return m, nodes, facts, status, digest(raw_hashes)


def literal_node(n):
    return (n['doc_id'], norm(n['name']), norm(n['type']))


def literal_fact(f, nodes):
    return (literal_node(nodes[f['source_occurrence']]), norm(f['predicate']),
            literal_node(nodes[f['target_occurrence']]), f['polarity'], f['time'], f['condition'])


def align(nodes, facts, decisions, input_hash):
    """Reviewed occurrence classes, never fuzzy string unions or transitive pair closure.

    Singleton unreviewed mentions remain separate. Explicit full classes make every merge
    reviewable and permit homonym splits. Inverse predicates may reverse members explicitly.
    """
    if decisions.get('input_hash') != input_hash:
        raise ValueError('Alignment belongs to different input snapshot')
    node_map = {oid: 'unreviewed:' + oid for oid in nodes}
    fact_map = {oid: 'unreviewed:' + oid for oid in facts}
    canonical, used_nodes, used_facts, ids = {}, set(), set(), set()
    for group in decisions.get('entity_classes', []):
        cid = group['id']
        members = group['members']
        if not isinstance(cid, str) or not cid or cid.startswith('unreviewed:') or cid in ids or not members or not group.get('reviewer') or not group.get('reason'):
            raise ValueError('Entity class needs unique id, members, reviewer and reason')
        ids.add(cid)
        if len(set(members)) != len(members) or any(x not in nodes or x in used_nodes for x in members):
            raise ValueError('Unknown/duplicate entity member')
        if len({nodes[x]['doc_id'] for x in members}) != 1:
            raise ValueError('Cross-document entity merge rejected')
        for x in members:
            node_map[x] = cid
            used_nodes.add(x)
    ids = set()
    for group in decisions.get('relation_classes', []):
        cid = group['id']
        members = group['members']
        claim = group['canonical']
        if not isinstance(cid, str) or not cid or cid.startswith('unreviewed:') or cid in ids or not members or not group.get('reviewer') or not group.get('reason'):
            raise ValueError('Relation class needs unique id, members, reviewer and reason')
        ids.add(cid)
        if any(not isinstance(claim.get(k), str) for k in ('source', 'predicate', 'target', 'polarity', 'time', 'condition')) or not claim['predicate'].strip():
            raise ValueError('Invalid canonical claim')
        docs = set()
        for member in members:
            oid = member['id']
            if oid not in facts or oid in used_facts or member.get('label') != 'equivalent':
                raise ValueError('Only unique known equivalent members can merge')
            f = facts[oid]
            docs.add(f['doc_id'])
            s, t = f['source_occurrence'], f['target_occurrence']
            if s not in used_nodes or t not in used_nodes:
                raise ValueError('Resolve relation endpoints first')
            if member.get('reverse', False):
                s, t = t, s
            if (node_map[s], node_map[t]) != (claim['source'], claim['target']):
                raise ValueError('Canonical endpoints do not match; inverse needs explicit reverse')
            if any(f[k] != claim[k] for k in ('polarity', 'time', 'condition')):
                raise ValueError('Qualifier differences require separate classes')
            used_facts.add(oid)
            fact_map[oid] = cid
        if len(docs) != 1:
            raise ValueError('Cross-document relation merge rejected')
        canonical[cid] = claim
    # Different class IDs for identical canonical claims would manufacture novelty.
    keys = [digest(c) for c in canonical.values()]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate canonical claim: consolidate its class')
    return node_map, fact_map, canonical, dict(
        entity_reviewed=len(used_nodes), entity_total=len(nodes),
        relation_reviewed=len(used_facts), relation_total=len(facts),
        complete=len(used_nodes) == len(nodes) and len(used_facts) == len(facts))


def graph(nodes, facts, region, cond, mapping=None):
    ns = {k: n for k, n in nodes.items() if n['region_id'] == region and n['extraction_condition'] == cond}
    fs = {k: f for k, f in facts.items() if f['region_id'] == region and f['extraction_condition'] == cond}
    out_nodes, out_edges = {}, {}
    nm, fm, canonical = mapping if mapping else ({}, {}, {})
    for oid, n in ns.items():
        key = nm[oid] if mapping else digest(literal_node(n))
        row = out_nodes.setdefault(key, {'id': key, 'mentions': []})
        row['mentions'].append(oid)
    for oid, f in fs.items():
        key = fm[oid] if mapping else digest(literal_fact(f, nodes))
        if key in canonical:
            claim = canonical[key]
        else:
            claim = {k: f[k] for k in ('predicate', 'polarity', 'time', 'condition')}
            claim.update(source=nm[f['source_occurrence']] if mapping else digest(literal_node(nodes[f['source_occurrence']])),
                         target=nm[f['target_occurrence']] if mapping else digest(literal_node(nodes[f['target_occurrence']])))
        row = out_edges.setdefault(key, dict(id=key, **claim, occurrences=[], evidence_spans=[], ambiguous_evidence_occurrences=[]))
        row['occurrences'].append(oid)
        if len(f['spans']) == 1:
            row['evidence_spans'] += [tuple(s) for s in f['spans']]
        else:
            row['ambiguous_evidence_occurrences'].append(oid)
    for e in out_edges.values():
        e['evidence_spans'] = sorted(set(e['evidence_spans']))
    return dict(region_id=region, condition=cond, directed=True, multigraph=True,
                validity='candidate_only', nodes=out_nodes, edges=out_edges)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else None


def structure(g):
    nodes, edges = set(g['nodes']), g['edges'].values()
    adjacency = {n: set() for n in nodes}
    undirected = {n: set() for n in nodes}
    for e in edges:
        adjacency[e['source']].add(e['target'])
        undirected[e['source']].add(e['target'])
        undirected[e['target']].add(e['source'])
    labels, components = {}, 0
    for n in sorted(nodes):
        if n in labels:
            continue
        stack = [n]
        labels[n] = components
        while stack:
            for v in undirected[stack.pop()]:
                if v not in labels:
                    labels[v] = components
                    stack.append(v)
        components += 1
    paths = {(a, b, c) for a in nodes for b in adjacency[a] for c in adjacency[b] if len({a, b, c}) == 3}
    return labels, paths, dict(nodes=len(nodes), edges=len(g['edges']), weak_components=components,
                               isolated_nodes=sum(not undirected[n] for n in nodes), directed_two_hop_paths=len(paths))


def compare(a, b):
    an, bn, ae, be = set(a['nodes']), set(b['nodes']), set(a['edges']), set(b['edges'])
    union = dict(nodes={**a['nodes'], **b['nodes']}, edges={**a['edges'], **b['edges']})
    labels, pa, sa = structure(a)
    _, pb, sb = structure(b)
    _, pu, su = structure(union)
    shared_evidence, novel_evidence = 0, 0
    novel_chars = 0
    for key in ae & be:
        av, bv = set(map(tuple, a['edges'][key]['evidence_spans'])), set(map(tuple, b['edges'][key]['evidence_spans']))
        shared_evidence += len(av & bv)
        novel_evidence += len(bv - av)
        # New span boundaries alone do not prove new evidence; also report uncovered characters.
        ac = {i for lo, hi in av for i in range(lo, hi)}
        bc = {i for lo, hi in bv for i in range(lo, hi)}
        novel_chars += len(bc - ac)
    bridges = sum(e['source'] in labels and e['target'] in labels and
                  labels[e['source']] != labels[e['target']] for k, e in b['edges'].items() if k not in ae)
    return dict(node_jaccard=jaccard(an, bn), edge_jaccard=jaccard(ae, be),
                novel_nodes_b_minus_a=len(bn - an), novel_edges_b_minus_a=len(be - ae),
                novel_edge_ratio_b=len(be - ae) / len(be) if be else None,
                shared_edges=len(ae & be), union_edges=len(ae | be), a=sa, b=sb, union=su,
                component_bridging_edges=bridges, novel_union_two_hop_paths=len(pu - pa - pb),
                shared_exact_evidence_spans=shared_evidence, novel_evidence_spans_on_shared_edges=novel_evidence,
                novel_evidence_characters_on_shared_edges=novel_chars,
                path_validity='unverified structural proxy; not evidence-path recall')


def candidate_pairs(nodes, facts, out, input_hash, audit_count=100):
    """Lexical blocking is a queue only. Reservoir sample excluded pairs for recall audit."""
    rng, excluded, sampled, count = random.Random(17), 0, [], 0
    by_region = defaultdict(list)
    for f in facts.values():
        by_region[f['region_id']].append(f)
    with Path(out, 'alignment_pairs.jsonl').open('w') as fp:
        for region, rows in sorted(by_region.items()):
            for a, b in itertools.combinations(rows, 2):
                # Compare duplicates within a size as well as across sizes.
                names_a = {norm(nodes[a[k]]['name']) for k in ('source_occurrence', 'target_occurrence')}
                names_b = {norm(nodes[b[k]]['name']) for k in ('source_occurrence', 'target_occurrence')}
                blocked = bool(names_a & names_b) or norm(a['predicate']) == norm(b['predicate'])
                pair = dict(a=a['occurrence_id'], b=b['occurrence_id'], region_id=region,
                            input_hash=input_hash, label=None, reason='', reviewer='')
                if blocked:
                    fp.write(json.dumps(pair) + '\n')
                    count += 1
                else:
                    excluded += 1
                    if len(sampled) < audit_count:
                        sampled.append(pair)
                    else:
                        index = rng.randrange(excluded)
                        if index < audit_count:
                            sampled[index] = pair
    save(Path(out) / 'alignment_blocking_audit.json', dict(excluded_population=excluded,
         inclusion_probability=min(1, audit_count / excluded) if excluded else None, samples=sampled))
    return dict(candidate_pairs=count, excluded_pairs=excluded, audit_sample=len(sampled))


def cost(rows, rates):
    recorded = [r for r in rows if r['recorded']]
    tokens = {k: sum(r[k] or 0 for r in recorded) for k in ('prompt_tokens', 'completion_tokens')}
    unknown = sum(not r['usage_known'] for r in recorded)
    seconds = sum(r['elapsed_seconds'] or 0 for r in recorded)
    missing_time = sum(r['elapsed_seconds'] is None for r in recorded)
    amount = None
    if rates is not None and not unknown:
        amount = (tokens['prompt_tokens'] * rates[0] + tokens['completion_tokens'] * rates[1]) / 1e6
    return dict(**tokens, calls=len(recorded), planned=len(rows), missing_usage_calls=unknown,
                seconds_recorded=seconds, missing_time_calls=missing_time, monetary_cost=amount,
                monetary_unit='user-specified', rates_per_million=rates)


def analyze(run, out, alignment=None, rates=None, per_stratum=20, pairs=False):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh analysis directory to preserve previous snapshots')
    if per_stratum <= 0:
        raise ValueError('Review sample size must be positive')
    if rates is not None and any(v < 0 for v in rates):
        raise ValueError('Rates must be nonnegative')
    m, nodes, facts, status, raw_hash = load(run)
    input_hash = digest([digest(m), raw_hash])
    out.mkdir(parents=True)
    save(out / 'occurrences.json', dict(input_hash=input_hash, entities=nodes, relations=facts))
    save(out / 'regions.json', m['regions'])
    save(out / 'call_audit.json', status)
    save(out / 'alignment_template.json', dict(input_hash=input_hash, entity_classes=[], relation_classes=[]))
    mapping, coverage = None, None
    if alignment:
        decisions = read(alignment)
        nm, fm, canonical, coverage = align(nodes, facts, decisions, input_hash)
        mapping = nm, fm, canonical
        save(out / 'alignment_applied.json', decisions)
    conditions = sorted({condition(j) for j in m['jobs']})
    regions = sorted({r['region_id'] for r in m['regions']})
    grouped = defaultdict(list)
    for s in status:
        grouped[s['region_id'], s['extraction_condition']].append(s)
    # One common cohort across ALL planned conditions, including same-size controls.
    complete = [r for r in regions if all(grouped[r, c] and all(x['complete'] for x in grouped[r, c]) for c in conditions)]
    graphs, comparisons = {}, []
    for mode in ('literal', 'semantic'):
        if mode == 'semantic' and mapping is None:
            continue
        for rid in regions:
            for c in conditions:
                g = graph(nodes, facts, rid, c, mapping if mode == 'semantic' else None)
                g['complete_generation'] = all(s['complete'] for s in grouped[rid, c])
                g['common_cohort'] = rid in complete
                graphs[mode, rid, c] = g
                save(out / 'graphs' / f'{mode}_{digest([rid, c])[:20]}.json', g)
        for a, b in itertools.permutations(conditions, 2):
            for rid in complete:
                comparisons.append(dict(mode=mode, region_id=rid, baseline=a, added=b,
                    alignment_complete=coverage['complete'] if mode == 'semantic' else None,
                    metrics=compare(graphs[mode, rid, a], graphs[mode, rid, b])))
    save(out / 'comparisons.json', comparisons)
    aggregates = []
    for mode, baseline, added in sorted({(r['mode'], r['baseline'], r['added']) for r in comparisons}):
        rows = [r for r in comparisons if (r['mode'], r['baseline'], r['added']) == (mode, baseline, added)]
        docs = defaultdict(list)
        for row in rows:
            doc = next(r['doc_id'] for r in m['regions'] if r['region_id'] == row['region_id'])
            docs[doc].append(row['metrics'])
        metrics = {}
        for metric in ('edge_jaccard', 'node_jaccard', 'novel_edges_b_minus_a', 'component_bridging_edges', 'novel_union_two_hop_paths', 'novel_evidence_characters_on_shared_edges'):
            values = {doc: [r[metric] for r in rs if r[metric] is not None] for doc, rs in docs.items()}
            means = {doc: sum(v) / len(v) for doc, v in values.items() if v}
            metrics[metric] = dict(by_document=means, document_macro_mean=sum(means.values()) / len(means) if means else None)
        aggregates.append(dict(mode=mode, baseline=baseline, added=added, regions=len(rows), documents=len(docs), metrics=metrics))
    save(out / 'document_macro.json', aggregates)
    review, key, rng = [], [], random.Random(17)
    mode = 'semantic' if mapping else 'literal'
    for a, b in itertools.permutations(conditions, 2):
        for rid in complete:
            ga, gb = graphs[mode, rid, a], graphs[mode, rid, b]
            for stratum, pool in (('novel', set(gb['edges']) - set(ga['edges'])),
                                  ('shared', set(gb['edges']) & set(ga['edges']))):
                chosen = rng.sample(sorted(pool), min(per_stratum, len(pool)))
                for edge_id in chosen:
                    edge = gb['edges'][edge_id]
                    item_id = digest([input_hash, mode, a, b, rid, edge_id])[:24]
                    # Keep size/seed/stratum in private key; annotators receive text and claims only.
                    def display(g, eid):
                        e = g['edges'][eid]
                        def name(nid):
                            return sorted({nodes[o]['name'] for o in g['nodes'][nid]['mentions']})
                        return dict(source=name(e['source']), predicate=e['predicate'], target=name(e['target']),
                                    polarity=e['polarity'], time=e['time'], condition=e['condition'],
                                    evidence_spans=e['evidence_spans'])
                    review.append(dict(id=item_id, region_id=rid, claim=display(gb, edge_id),
                                       baseline_claims=[display(ga, k) for k in sorted(ga['edges'])],
                                       supported=None, semantically_novel=None, error_type=None,
                                       annotator='', rationale=''))
                    key.append(dict(id=item_id, baseline=a, added=b, region_id=rid,
                                    doc_id=next(r['doc_id'] for r in m['regions'] if r['region_id'] == rid),
                                    mode=mode, stratum=stratum, population=len(pool), sampled=len(chosen),
                                    inclusion_probability=len(chosen) / len(pool), edge_id=edge_id))
    rng.shuffle(review)
    save(out / 'review_blind.json', review)
    save(out / 'review_key.json', key)
    summary = dict(analysis_code_sha256=sha(Path(__file__).read_text()), core_code_sha256=sha(Path(__file__).with_name('core.py').read_text()), input_hash=input_hash, manifest_hash=digest(m), raw_hash=raw_hash,
                   legacy=m.get('version') != 'experiment1-v1', conditions=conditions,
                   common_complete_regions=complete, excluded_regions=sorted(set(regions) - set(complete)),
                   total_regions=len(regions), total_calls=len(status), recorded_calls=sum(s['recorded'] for s in status),
                   truncated_calls=sum(s['finish_reason'] == 'length' for s in status),
                   calls_with_item_or_generation_errors=sum(bool(s['errors']) for s in status),
                   empty_calls=sum(s['empty'] for s in status), duplicate_relations=sum(s['duplicate_relations'] for s in status),
                   retained_entity_occurrences=len(nodes), retained_relation_occurrences=len(facts),
                   semantic_alignment=coverage, semantic_jaccard_status='unavailable' if mapping is None else ('reviewed' if coverage['complete'] else 'partial_do_not_claim_final'),
                   costs={c: cost([s for s in status if s['extraction_condition'] == c], rates) for c in conditions},
                   review_items=len(review), valid_novel_edges=None,
                   notes=['Generation completion does not imply item validity. Invalid items are quarantined.',
                          'Literal dedup is normalized surface+type with qualifiers; homonyms remain a limitation.',
                          'Raw outputs and source spans are not ground-truth validity labels.',
                          'All comparison rows use the same complete region cohort.',
                          'Controls change cost; no cost-matched causal claim without further design.'])
    if pairs:
        summary['alignment_queue'] = candidate_pairs(nodes, facts, out, input_hash)
    save(out / 'summary.json', summary)
    return summary


def evaluate(analysis, annotations, output):
    """HT descriptive estimates per directed comparison; unresolved labels never become false."""
    root = Path(analysis)
    key = read(root / 'review_key.json')
    template = {r['id']: r for r in read(root / 'review_blind.json')}
    rows = read(annotations)
    submitted = {}
    for row in rows:
        rid = row['id']
        if rid not in template or rid in submitted:
            raise ValueError('Unknown or duplicate review id')
        for field in ('region_id', 'claim', 'baseline_claims'):
            if row[field] != template[rid][field]:
                raise ValueError('Review input content was modified')
        for field in ('supported', 'semantically_novel'):
            if row.get(field) is not None and type(row[field]) is not bool:
                raise ValueError('Review labels must be true/false/null')
        submitted[rid] = row
    groups = defaultdict(list)
    for k in key:
        groups[k['baseline'], k['added']].append(k)
    costs = read(root / 'summary.json')['costs']
    result = []
    for (a, b), ks in sorted(groups.items()):
        estimate, valid, reviewed, novel_estimate = 0., 0, 0, 0.
        by_doc = defaultdict(float)
        for k in ks:
            r = submitted.get(k['id'], {})
            if type(r.get('supported')) is not bool or type(r.get('semantically_novel')) is not bool or not r.get('annotator') or not r.get('rationale'):
                continue
            reviewed += 1
            if r['supported'] and r['semantically_novel']:
                valid += 1
                weighted = 1 / k['inclusion_probability']
                estimate += weighted
                by_doc[k['doc_id']] += weighted
                if k['stratum'] == 'novel':
                    novel_estimate += weighted
        complete = reviewed == len(ks)
        population = sum({(k['region_id'], k['stratum']): k['population'] for k in ks}.values())
        novel_population = sum({k['region_id']: k['population'] for k in ks if k['stratum'] == 'novel'}.values())
        result.append(dict(baseline=a, added=b, sampled=len(ks), reviewed=reviewed, review_complete=complete,
                           observed_valid_novel=valid, estimated_valid_novel=estimate if complete else None,
                           estimated_valid_novel_ratio_all_added=estimate / population if complete and population else None,
                           estimated_valid_novel_ratio_candidate_novel=novel_estimate / novel_population if complete and novel_population else None,
                           estimated_valid_novel_by_doc=dict(by_doc) if complete else None,
                           additional_seconds_per_estimated_valid_novel=costs[b]['seconds_recorded'] / estimate if complete and estimate and costs[b]['missing_time_calls'] == 0 else None,
                           additional_money_per_estimated_valid_novel=costs[b]['monetary_cost'] / estimate if complete and estimate and costs[b]['monetary_cost'] is not None else None))
    report = dict(comparisons=result, notes=['Descriptive stratified HT estimates, no recall or significance claim.',
          'Costs include all added-condition calls, including excluded regions and failed calls.',
          'Shared-stratum audit can reveal false semantic/literal merges.',
          'Multiple annotators should adjudicate one final label per item externally.'])
    if Path(output).exists():
        raise ValueError('Evaluation output already exists')
    save(output, report)
    return report
