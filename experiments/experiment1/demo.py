"""Fully offline executable fixture, NOT model inference or research evidence."""
import argparse
import json
from pathlib import Path
from . import core as c
from . import analysis as a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh demo output directory')
    text = 'Alice knows Bob. Bob helps Cara. Alicia is Alice. Cara trusts Dana. ' + 'Neutral filler. ' * 70
    region = dict(doc_id='synthetic', region_id='synthetic_r0', char_start=0, char_end=len(text), text=text, sha256=c.sha(text))
    jobs = c.split_region(region, list(text))  # Characters ONLY for this labelled test fixture.
    m = dict(version='experiment1-v1', corpus_sha256=c.sha(text), regions=[region], jobs=jobs,
             protocol=dict(model='SYNTHETIC-NO-LLM', model_revision='fixture-v1', tokenizer_id='character-fixture-only',
                           server_id='fixture', url='http://fixture', temperature=.2, top_p=.9, max_tokens=4096,
                           prompt=c.PROMPT, gleaning=0, retries=0, timeout=1))
    run = args.out / 'run'
    c.save(run / 'manifest.json', m)
    # Chunk content can repeat: identify requests by queue order, as the actual runner does.
    queue = iter(jobs)
    def transport(url, req, timeout):
        job = next(queue)
        text, size = job['text'], job['size']
        ents = [dict(id=n, name=n, type='person', evidence=n) for n in ('Alice', 'Alicia', 'Bob', 'Cara', 'Dana') if n in text]
        rels = []
        if 'Alice knows Bob.' in text:
            source, predicate, target = ('Bob', 'is known by', 'Alice') if size == 512 else ('Alicia' if size == 256 else 'Alice', 'knows', 'Bob')
            rels.append(dict(source=source, predicate=predicate, target=target, evidence='Alice knows Bob.'))
        if size in (128, 512) and 'Bob helps Cara.' in text:
            rels.append(dict(source='Bob', predicate='helps', target='Cara', evidence='Bob helps Cara.'))
        if size in (256, 1024) and 'Cara trusts Dana.' in text:
            rels.append(dict(source='Cara', predicate='trusts', target='Dana', evidence='Cara trusts Dana.'))
        for r in rels:
            r.update(polarity='positive', time='', condition='')
        return json.dumps(dict(choices=[dict(finish_reason='stop', message={'content': json.dumps(dict(entities=ents, relations=rels))})],
                               usage=dict(prompt_tokens=len(text), completion_tokens=100)))
    c.extract(run, transport)
    first = args.out / 'literal'
    a.analyze(run, first)
    occ = c.read(first / 'occurrences.json')
    groups, facts = {}, {}
    for oid, n in occ['entities'].items():
        groups.setdefault('Alice' if n['name'] == 'Alicia' else n['name'], []).append(oid)
    for oid, f in occ['relations'].items():
        facts.setdefault('knows' if f['predicate'] == 'is known by' else f['predicate'], []).append(oid)
    ds = dict(input_hash=occ['input_hash'], entity_classes=[], relation_classes=[])
    for name, members in groups.items():
        ds['entity_classes'].append(dict(id=name, members=members, reviewer='synthetic_fixture', reason='Authored fixture identity'))
    endpoints = {'knows': ('Alice', 'Bob'), 'helps': ('Bob', 'Cara'), 'trusts': ('Cara', 'Dana')}
    for pred, members in facts.items():
        s, t = endpoints[pred]
        ds['relation_classes'].append(dict(id=pred, reviewer='synthetic_fixture', reason='Authored fixture equivalence',
            canonical=dict(source=s, predicate=pred, target=t, polarity='positive', time='', condition=''),
            members=[dict(id=oid, label='equivalent', reverse=occ['relations'][oid]['predicate'] == 'is known by') for oid in members]))
    c.save(args.out / 'synthetic_alignment.json', ds)
    summary = a.analyze(run, args.out / 'semantic', args.out / 'synthetic_alignment.json')
    print(json.dumps(dict(note='SYNTHETIC FIXTURE ONLY', calls=summary['total_calls'],
                         semantic_status=summary['semantic_jaccard_status'], output=str(args.out)), indent=2))


if __name__ == '__main__':
    main()
