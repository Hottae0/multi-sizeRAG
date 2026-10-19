import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from experiments.experiment1 import core as c
from experiments.experiment1 import analysis as a


def region(text='Alice knows Bob. Bob helps Cara.'):
    return dict(doc_id='d', region_id='r', text=text, char_start=100,
                char_end=100 + len(text), sha256=c.sha(text))


def payload():
    return dict(entities=[dict(id='a', name='Alice', type='person', evidence='Alice'),
                          dict(id='b', name='Bob', type='person', evidence='Bob'),
                          dict(id='c', name='Cara', type='person', evidence='Cara')],
                relations=[dict(source='a', predicate='knows', target='b', polarity='positive',
                                time='', condition='', evidence='Alice knows Bob.'),
                           dict(source='b', predicate='helps', target='c', polarity='positive',
                                time='', condition='', evidence='Bob helps Cara.')])


def response(obj=None, finish='stop'):
    return dict(choices=[dict(finish_reason=finish, message={'content': json.dumps(obj if obj is not None else payload())})],
                usage=dict(prompt_tokens=20, completion_tokens=30))


def manifest():
    r = region()
    jobs = c.split_region(r, list(r['text']), seeds=(17,), boundaries=(0,))
    return dict(version='experiment1-v1', corpus_sha256='test', regions=[r], jobs=jobs,
                protocol=dict(model='fixture', model_revision='fixture', tokenizer_id='character-test-only',
                              server_id='fixture', url='http://fixture', temperature=0.2, top_p=.9,
                              prompt=c.PROMPT, max_tokens=4096, gleaning=0, retries=0, timeout=1))


class ExtractionTests(unittest.TestCase):
    def test_prepare_freezes_real_endpoint_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source.json'
            c.save(source, dict(corpus_sha256='corpus', regions=[region()]))
            def tokenize(url, body, timeout=300):
                self.assertTrue(url.endswith('/tokenize'))
                self.assertFalse(body['add_special'])
                return json.dumps({'tokens': [{'piece': x} for x in body['content']]})
            with patch.object(c, 'post', tokenize):
                m = c.prepare(source, root / 'run', 'http://fixture', 'model', 'revision', 'tokenizer', 'server')
            self.assertEqual(len(m['jobs']), 4)
            self.assertEqual(m['protocol']['gleaning'], 0)
            self.assertTrue(m['regions'][0]['tokenization_sha256'])
            with self.assertRaises(ValueError):
                c.prepare(source, root / 'run', 'http://fixture', 'model', 'revision', 'tokenizer', 'server')

    def test_utf8_boundary_and_tail_coverage(self):
        r = region('가나다🙂 abc ' * 400)
        pieces = [bytes([b]) for b in r['text'].encode()]
        jobs = c.split_region(r, pieces, seeds=(17, 29), boundaries=(0, 64))
        c.validate_manifest(dict(regions=[r], jobs=jobs))
        for cond in {c.condition(j) for j in jobs}:
            js = sorted((j for j in jobs if c.condition(j) == cond), key=lambda j: j['char_start'])
            self.assertEqual(''.join(j['text'] for j in js), r['text'])
        self.assertEqual(len({c.condition(j) for j in jobs}), 16)

    def test_gap_duplicate_and_hash_rejected(self):
        for mutation in ('gap', 'id', 'hash'):
            m = manifest()
            if mutation == 'gap': m['jobs'].pop()
            if mutation == 'id': m['jobs'].append(m['jobs'][0])
            if mutation == 'hash': m['jobs'][0]['text'] += '!'
            with self.assertRaises(ValueError): c.validate_manifest(m)

    def test_single_pass_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as d:
            m = manifest(); c.save(Path(d) / 'manifest.json', m)
            calls = []
            def transport(*args):
                calls.append(args)
                return json.dumps(response())
            self.assertEqual(c.extract(d, transport), 4)
            self.assertEqual(c.extract(d, transport), 0)
            self.assertEqual(len(calls), 4)
            m['protocol']['max_tokens'] += 1; c.save(Path(d) / 'manifest.json', m)
            with self.assertRaises(ValueError): c.extract(d, transport)

    def test_failure_and_malformed_wire_retained_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            m = manifest(); c.save(Path(d) / 'manifest.json', m)
            c.extract(d, lambda *args: 'not json')
            self.assertEqual(c.extract(d, lambda *args: self.fail('retried')), 0)
            for p in Path(d, 'raw').glob('*.json'):
                self.assertEqual(c.read(p)['wire_response'], 'not json')
                self.assertEqual(c.read(p)['state'], 'failed')

    def test_started_unknown_attempt_not_resent(self):
        with tempfile.TemporaryDirectory() as d:
            m = manifest(); c.save(Path(d) / 'manifest.json', m)
            j = m['jobs'][0]
            c.save(Path(d) / 'raw' / (j['job_id'] + '.json'), dict(job=j, request=c.request(m, j), manifest_hash=c.digest(m), state='started'))
            self.assertEqual(c.extract(d, lambda *args: json.dumps(response())), 3)

    def test_truncated_valid_json_excluded(self):
        p = c.parse_record(manifest()['jobs'][0], {'response': response(finish='length')})
        self.assertFalse(p['complete']); self.assertFalse(p['relations'])

    def test_evidence_endpoint_errors_and_duplicates(self):
        data = payload(); data['relations'] += [copy.deepcopy(data['relations'][0])]
        data['relations'] += [dict(data['relations'][0], target='missing'), dict(data['relations'][0], evidence='invented')]
        p = c.parse_record(manifest()['jobs'][0], {'response': response(data)})
        self.assertTrue(p['complete']); self.assertEqual(p['duplicate_relations'], 1)
        self.assertEqual(len(p['relations']), 3); self.assertEqual(len(p['errors']), 2)
        self.assertEqual(p['entities'][1]['spans'], [[112, 115], [117, 120]])

    def test_duplicate_entity_ids_quarantined(self):
        data = payload(); data['entities'].append(data['entities'][0])
        p = c.parse_record(manifest()['jobs'][0], {'response': response(data)})
        self.assertEqual(len(p['entities']), 2); self.assertEqual(len(p['relations']), 1)


class AnalysisTests(unittest.TestCase):
    def create_run(self, root):
        run = root / 'run'; m = manifest(); c.save(run / 'manifest.json', m)
        c.extract(run, lambda *args: json.dumps(response()))
        return run

    def decisions(self, nodes, facts, ih):
        classes, relations = {}, {}
        for oid, n in nodes.items(): classes.setdefault(n['name'], []).append(oid)
        for oid, f in facts.items(): relations.setdefault(f['predicate'], []).append(oid)
        return dict(input_hash=ih,
            entity_classes=[dict(id=name, members=members, reviewer='test', reason='fixture identity') for name, members in classes.items()],
            relation_classes=[dict(id=p, members=[dict(id=x, label='equivalent') for x in members], reviewer='test', reason='fixture equivalence',
                                   canonical=dict(source='Alice' if p == 'knows' else 'Bob', predicate=p, target='Bob' if p == 'knows' else 'Cara',
                                                  polarity='positive', time='', condition='')) for p, members in relations.items()])

    def test_end_to_end_semantic_and_blind_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = self.create_run(root)
            s = a.analyze(run, root / 'literal', per_stratum=2, pairs=True)
            self.assertIsNone(s['semantic_alignment']); self.assertEqual(s['common_complete_regions'], ['r'])
            occ = c.read(root / 'literal/occurrences.json')
            decisions = self.decisions(occ['entities'], occ['relations'], occ['input_hash'])
            c.save(root / 'alignment.json', decisions)
            s = a.analyze(run, root / 'semantic', root / 'alignment.json', rates=(1, 2))
            self.assertEqual(s['semantic_jaccard_status'], 'reviewed')
            rows = c.read(root / 'semantic/comparisons.json')
            self.assertTrue(all(x['metrics']['edge_jaccard'] == 1 for x in rows))
            gs = list((root / 'semantic/graphs').glob('semantic*'))
            for gp in gs:
                g = c.read(gp)
                self.assertEqual(len(g['edges']), 2)
                for edge in g['edges'].values(): self.assertEqual(len(edge['occurrences']), 1)
            review = c.read(root / 'semantic/review_blind.json')
            self.assertNotIn('condition', review[0]); self.assertNotIn('stratum', review[0])
            c.save(root / 'labels.json', review)
            result = a.evaluate(root / 'semantic', root / 'labels.json', root / 'incomplete.json')
            self.assertTrue(all(x['estimated_valid_novel'] is None for x in result['comparisons']))
            for r in review: r.update(supported=True, semantically_novel=False, annotator='tester', rationale='same claim')
            c.save(root / 'labels.json', review)
            result = a.evaluate(root / 'semantic', root / 'labels.json', root / 'complete.json')
            self.assertTrue(all(x['estimated_valid_novel'] == 0 for x in result['comparisons']))

    def test_failure_excludes_common_region_and_cost_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = self.create_run(root)
            p = next((run / 'raw').glob('*.json')); r = c.read(p)
            r['response'] = response(finish='length'); r['response'].pop('usage'); c.save(p, r)
            s = a.analyze(run, root / 'out', rates=(1, 2))
            self.assertEqual(s['common_complete_regions'], [])
            self.assertEqual(c.read(root / 'out/comparisons.json'), [])
            self.assertTrue(any(x['monetary_cost'] is None for x in s['costs'].values()))
            self.assertEqual(s['truncated_calls'], 1)

    def test_alignment_guards_and_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = self.create_run(root)
            m, ns, fs, st, raw = a.load(run); ih = c.digest([c.digest(m), raw])
            ds = self.decisions(ns, fs, ih)
            # Different surface name and predicate still map to the same reviewed identities.
            next(iter(ns.values()))['name'] = 'Alicia'
            next(iter(fs.values()))['predicate'] = 'is acquainted with'
            nm, fm, canonical, coverage = a.align(ns, fs, ds, ih)
            self.assertTrue(coverage['complete']); self.assertEqual(len(set(fm.values())), 2)
            for mutation in ('qualifier', 'entails', 'input', 'duplicate', 'direction', 'document'):
                bad = copy.deepcopy(ds); ncopy = copy.deepcopy(ns)
                if mutation == 'qualifier': bad['relation_classes'][0]['canonical']['polarity'] = 'negative'
                if mutation == 'entails': bad['relation_classes'][0]['members'][0]['label'] = 'entails'
                if mutation == 'input': bad['input_hash'] = 'stale'
                if mutation == 'duplicate': bad['entity_classes'][0]['members'] *= 2
                if mutation == 'direction': bad['relation_classes'][0]['members'][0]['reverse'] = True
                if mutation == 'document': next(iter(ncopy.values()))['doc_id'] = 'other'
                with self.assertRaises(ValueError, msg=mutation): a.align(ncopy, fs, bad, ih)

    def test_direction_structure_and_evidence(self):
        def g(nodes, edges):
            return dict(nodes={n: {} for n in nodes}, edges={k: dict(source=s, target=t, evidence_spans=ev) for k, s, t, ev in edges})
        ga = g('ABCD', [('ab', 'A', 'B', [(0, 5)]), ('cd', 'C', 'D', [(20, 25)])])
        gb = g('ABCD', [('ab', 'A', 'B', [(0, 7)]), ('bc', 'B', 'C', [(10, 15)])])
        result = a.compare(ga, gb)
        self.assertEqual(result['component_bridging_edges'], 1)
        self.assertEqual(result['novel_union_two_hop_paths'], 1)  # B-C-D; A-B-C already in B.
        self.assertEqual(result['novel_evidence_characters_on_shared_edges'], 2)
        self.assertEqual(result['edge_jaccard'], 1/3)
        self.assertIsNone(a.jaccard(set(), set()))

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): a.analyze('not-needed', d)


if __name__ == '__main__':
    unittest.main()
