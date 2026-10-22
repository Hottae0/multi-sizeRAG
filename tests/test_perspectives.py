import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.experiment1 import core, perspectives as p


class PerspectiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        text = 'Alice knows Bob. ' * 140
        region = dict(doc_id='d', region_id='full', char_start=0, char_end=len(text),
                      text=text, sha256=core.sha(text))
        jobs = core.split_region(region, list(text), seeds=(17, 29))
        jobs = [j for j in jobs if j['seed'] == 17 or j['size'] == 256]
        self.m = dict(version='experiment1-v1', regions=[region], jobs=jobs)
        core.save(self.run / 'manifest.json', self.m)
        for j in jobs:
            self.write(j)

    def write(self, j, missing=False, finish='stop', empty=False):
        obj = dict(entities=[], relations=[])
        if not empty:
            obj = dict(entities=[dict(id='a', name='Alice', type='person', evidence='Alice'),
                                 dict(id='b', name='Bob', type='person', evidence='Bob')],
                       relations=[dict(source='a', target='b', predicate='knows', polarity='positive',
                                       time='', condition='', evidence='Alice knows Bob.')])
            if missing:
                obj['relations'][0].pop('time')
                obj['relations'][0].pop('condition')
        r = dict(job=j, manifest_hash=core.digest(self.m), elapsed_seconds=1,
                 response=dict(choices=[dict(finish_reason=finish, message={'content': json.dumps(obj)})],
                               usage=dict(prompt_tokens=3, completion_tokens=2)))
        core.save(self.run / 'raw' / (j['job_id'] + '.json'), r)

    def analyze(self, name='out', **kwargs):
        out = self.root / name
        return out, p.analyze(self.run, out, **kwargs)

    def review(self, out, info='new_claim', aspects=None):
        data = core.read(out / 'review_blind.json')
        blocks = {b['block_id']: b for b in core.read(out / 'blocks.json')}
        catalog = core.read(out / 'review_baselines.json')
        for row in data['items']:
            b = blocks[row['block_id']]
            row.update(support='supported', information=info, aspects=aspects or ['participant_role'],
                       support_spans=[[b['char_start'], b['char_end']]], semantic_fact_id='same-document-fact',
                       annotator='fixture', rationale='Test fixture, not a semantic gold label',
                       baseline_matches=[catalog[row['baseline_ref']][0]['id']] if info != 'new_claim' or (aspects or ['participant_role']) != ['same'] else [])
        return data

    def evaluate(self, out, data, name='evaluation.json'):
        labels = self.root / ('labels-' + name)
        core.save(labels, data)
        return p.evaluate(out, labels, self.root / name)

    def test_maximum_blocks_and_selected_cohort_not_all_five(self):
        bad = next(j for j in self.m['jobs'] if j['size'] == 128)
        self.write(bad, finish='length')
        out, s = self.analyze()
        self.assertEqual(s['planned_blocks'], 3)
        self.assertEqual(s['common_complete_blocks'], 3)
        blocks = core.read(out / 'blocks.json')
        self.assertEqual(len(blocks[0]['conditions'][p.BASELINE]['contained_job_ids']), 4)
        self.assertTrue(blocks[-1]['short_tail'])
        _, all_s = self.analyze('all', cohort_conditions=[p.BASELINE, *p.ADDED, 'e128_s17_b0', 'e1024_s17_b0'])
        self.assertEqual(all_s['common_complete_blocks'], 2)

    def test_failure_keeps_planned_blocks_and_no_fabricated_zero(self):
        for j in self.m['jobs']:
            if core.condition(j) == p.ADDED[0]:
                self.write(j, finish='length')
        out, s = self.analyze()
        self.assertEqual(s['common_complete_blocks'], 0)
        self.assertEqual(len(core.read(out / 'comparisons.json')), 6)
        result = self.evaluate(out, core.read(out / 'review_blind.json'))
        self.assertTrue(all(r['document_macro_candidate_counts'] is None for r in result['comparisons']))

    def test_crossing_chunks_excluded_not_clipped(self):
        region = self.m['regions'][0]
        shifted = core.split_region(region, list(region['text']), sizes=(512,), boundaries=(64,))
        self.m['jobs'] += shifted
        core.save(self.run / 'manifest.json', self.m)
        for j in self.m['jobs']:
            self.write(j)
        out, s = self.analyze(added=['e512_s17_b64'])
        self.assertEqual(s['common_complete_blocks'], 0)
        self.assertTrue(any(b['conditions']['e512_s17_b64']['crossing_job_ids'] for b in core.read(out / 'blocks.json')))

    def test_missing_unknown_separate_from_explicit_absence_and_raw_unchanged(self):
        js = [j for j in self.m['jobs'] if core.condition(j) == p.BASELINE]
        self.write(js[0], missing=True)
        before = {str(x): x.read_bytes() for x in self.run.rglob('*.json')}
        _, strict = self.analyze()
        out, relaxed = self.analyze('relaxed', qualifier_policy='missing-unknown')
        self.assertEqual(relaxed['candidate_relation_occurrences'], strict['candidate_relation_occurrences'] + 1)
        facts = core.read(out / 'occurrences.json')['relations']
        missing = facts[js[0]['job_id'] + ':r:0']
        self.assertIsNone(missing['time'])
        self.assertEqual(missing['missing_qualifiers'], ['time', 'condition'])
        comp = next(r for r in core.read(out / 'comparisons.json') if r['block_id'] == core.read(out / 'blocks.json')[0]['block_id'])
        self.assertEqual(comp['baseline_candidates'], 2)  # unknown and explicit empty must not merge
        self.assertEqual(before, {str(x): x.read_bytes() for x in self.run.rglob('*.json')})

    def test_orthogonal_axes_and_no_unique_fact_claim_without_review(self):
        out, _ = self.analyze()
        pending = self.evaluate(out, core.read(out / 'review_blind.json'), 'pending.json')
        self.assertTrue(all(r['document_macro_candidate_counts'] is None for r in pending['comparisons']))
        data = self.review(out, info='refinement', aspects=['time_condition'])
        result = self.evaluate(out, data)
        for r in result['comparisons']:
            counts = r['document_macro_candidate_counts']
            self.assertEqual(counts['new_claim'], 0)
            self.assertEqual(counts['refinement'], 3)
            self.assertEqual(counts['different_perspective'], 3)
            self.assertIsNone(r['by_document']['d']['unique_supported_facts'])
        data['document_deduplication'] = dict(complete=True, reviewer='fixture', rationale='Same fact across blocks')
        result = self.evaluate(out, data, 'dedup.json')
        self.assertEqual(result['comparisons'][0]['by_document']['d']['unique_supported_facts'], 1)

    def test_unsupported_and_uncertain_not_counted_as_supported_novel(self):
        out, _ = self.analyze()
        data = self.review(out)
        for row in data['items']:
            row.update(support='unsupported', information=None, aspects=[], semantic_fact_id='', support_spans=[])
        result = self.evaluate(out, data)
        self.assertEqual(result['comparisons'][0]['document_macro_candidate_counts']['new_claim'], 0)
        data['items'][0]['support'] = 'uncertain'
        result = self.evaluate(out, data, 'uncertain.json')
        self.assertTrue(any(r['document_macro_candidate_counts'] is None for r in result['comparisons']))

    def test_review_guards_snapshot_content_links_spans_and_aspects(self):
        out, _ = self.analyze()
        for kind in ('snapshot', 'content', 'link', 'span', 'aspects', 'equivalent', 'duplicate'):
            data = self.review(out)
            row = data['items'][0]
            if kind == 'snapshot': data['measurement_hash'] = 'stale'
            if kind == 'content': row['claim']['predicate'] = 'invented'
            if kind == 'link': row['baseline_matches'] = ['foreign']
            if kind == 'span': row['support_spans'] = [[-1, 10]]
            if kind == 'aspects': row['aspects'] = ['same', 'attribution']
            if kind == 'equivalent': row.update(information='equivalent', baseline_matches=[core.read(out/'review_baselines.json')[row['baseline_ref']][0]['id']])
            if kind == 'duplicate': data['items'].append(copy.deepcopy(row))
            with self.assertRaises(ValueError, msg=kind):
                self.evaluate(out, data, kind+'.json')
        catalog = core.read(out / 'review_baselines.json')
        next(iter(catalog.values()))[0]['predicate'] = 'changed'
        core.save(out / 'review_baselines.json', catalog)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.evaluate(out, self.review(out), 'tampered.json')

    def test_sample_weighting_is_candidate_estimate_not_unique_fact_total(self):
        for j in self.m['jobs']:
            if core.condition(j) != p.ADDED[1]:
                continue
            path = self.run / 'raw' / (j['job_id'] + '.json')
            record = core.read(path)
            data = json.loads(record['response']['choices'][0]['message']['content'])
            base = data['relations'][0]
            data['relations'] += [dict(base, predicate='fixture-' + str(i)) for i in range(5)]
            record['response']['choices'][0]['message']['content'] = json.dumps(data)
            core.save(path, record)
        out, _ = self.analyze(per_stratum=1)
        data = self.review(out)
        data['document_deduplication'] = dict(complete=True, reviewer='fixture', rationale='Test only')
        result = self.evaluate(out, data)
        doc = next(r for r in result['comparisons'] if r['added'] == p.ADDED[1])['by_document']['d']
        self.assertEqual(doc['candidate_population'], 18)
        self.assertEqual(doc['sampled'], 6)
        self.assertEqual(doc['estimated_candidate_counts']['new_claim'], 18)
        self.assertFalse(doc['census'])
        self.assertIsNone(doc['unique_supported_facts'])

    def test_empty_graphs_are_zero_with_successful_coverage(self):
        for j in self.m['jobs']:
            self.write(j, empty=True)
        out, _ = self.analyze()
        result = self.evaluate(out, core.read(out / 'review_blind.json'))
        self.assertTrue(all(r['review_complete'] for r in result['comparisons']))
        self.assertEqual(result['comparisons'][0]['document_macro_candidate_counts']['new_claim'], 0)


if __name__ == '__main__':
    unittest.main()
