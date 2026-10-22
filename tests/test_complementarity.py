import json
import tempfile
import unittest
from pathlib import Path
from experiments.experiment1 import core, complementarity as c


class ComplementarityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'document'
        text = 'Alice knows Bob. ' * 140
        region = dict(doc_id='d', region_id='full', char_start=0, char_end=len(text), text=text, sha256=core.sha(text))
        jobs = [j for j in core.split_region(region, list(text), seeds=(17, 29)) if j['seed'] == 17 or j['size'] == 256]
        self.manifest = dict(version='experiment1-v1', regions=[region], jobs=jobs)
        core.save(self.run/'manifest.json', self.manifest)
        for j in jobs:
            self.write(j)

    def write(self, j, finish='stop', missing=False):
        obj = dict(entities=[dict(id='a', name='Alice', type='person', evidence='Alice'),
                             dict(id='b', name='Bob', type='person', evidence='Bob')],
                   relations=[dict(source='a', target='b', predicate='knows', polarity='positive', time='', condition='', evidence='Alice knows Bob.')])
        if missing:
            obj['relations'][0].pop('time')
        core.save(self.run/'raw'/(j['job_id']+'.json'), dict(job=j, manifest_hash=core.digest(self.manifest),
                  response=dict(choices=[dict(finish_reason=finish, message=dict(content=json.dumps(obj)))])))

    def analyze(self, **kw):
        out = self.root/'analysis'
        return out, c.analyze(self.run, out, **kw)

    def labels(self, out):
        labels = core.read(out/'review_template.json')
        for row in labels['items']:
            row.update(verdict='supported', family_id='knows', canonical_claim='Alice knows Bob',
                       support_quotes=['Alice knows Bob.'], reviewer='fixture', reviewer_kind='human',
                       rationale='fixture only', alignment_rationale='same core fact')
        return labels

    def evaluate(self, out, labels):
        p = self.root/'labels.json';core.save(p, labels)
        return c.evaluate(out, p, self.root/'evaluation')

    def test_oracle_and_ablation_distinguish_local_complementarity(self):
        s = {'a': {('b1','x'),('b2','z')}, 'b': {('b1','y'),('b2','z')}}
        r = c.coverage(s, ['a','b'], ['b1','b2'])
        self.assertEqual((r['union'],r['local_best_oracle'],r['union_above_local_best']), (3,2,1))
        self.assertEqual(r['loss_if_remove'], {'a':1,'b':1})

    def test_duplicate_fact_does_not_multiply_across_sizes(self):
        out, s = self.analyze()
        r = self.evaluate(out, self.labels(out))
        self.assertEqual(r['semantic_status'], 'review_complete')
        metrics = r['complete_review_metrics']
        self.assertEqual(metrics['gain_over_best_single'], 0)
        self.assertTrue(all(v == 0 for v in metrics['loss_if_remove'].values()))
        self.assertEqual(len(metrics['combinations']), 15)
        self.assertIsNone(s['supported_core_metrics'])

    def test_failed_repeat_changes_common_cohort_not_size_cohort(self):
        j = next(j for j in self.manifest['jobs'] if j['seed']==29)
        self.write(j, finish='length')
        out, s = self.analyze()
        self.assertLess(s['scope_coverage']['common_cohort']['blocks'], s['scope_coverage']['common_sizes']['blocks'])

    def test_partial_review_not_population_metrics(self):
        out, _ = self.analyze()
        labels = self.labels(out);labels['items'] = labels['items'][:1]
        r = self.evaluate(out, labels)
        self.assertIsNone(r['complete_review_metrics'])
        self.assertEqual(r['resolved_claims'], 1)

    def test_invalid_quote_rejected(self):
        out, _ = self.analyze();labels=self.labels(out)
        labels['items'][0]['support_quotes']=['invented quote']
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.evaluate(out, labels)

    def test_stale_labels_rejected(self):
        out, _ = self.analyze();labels=self.labels(out);labels['input_hash']='wrong'
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.evaluate(out, labels)

    def test_missing_qualifier_preserved_as_unknown(self):
        for j in self.manifest['jobs']:
            if j['size']==512:self.write(j, missing=True)
        out, _ = self.analyze(qualifier_policy='missing-unknown')
        rows=core.read(out/'snapshot.json')['claims']
        self.assertTrue(any(r['claim']['time'] is None and r['claim']['missing_qualifiers']==['time'] for r in rows))

    def test_no_common_blocks_is_not_zero_novelty(self):
        for j in self.manifest['jobs']:
            if j['size']==1024:self.write(j, finish='length')
        out, s = self.analyze()
        self.assertEqual(s['scope_coverage']['common_cohort']['blocks'], 0)
        r = self.evaluate(out, self.labels(out))
        self.assertIsNone(r['complete_review_metrics'])

    def test_no_overwrite_or_source_output(self):
        out, _ = self.analyze()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            c.analyze(self.run, out)
        with self.assertRaisesRegex(ValueError, 'outside'):
            c.analyze(self.run, self.run/'raw'/'new')

    def test_missing_condition_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Missing requested'):
            self.analyze(sizes=[2048])

    def test_duplicate_annotation_rejected(self):
        out, _ = self.analyze();labels=self.labels(out);labels['items'].append(labels['items'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.evaluate(out, labels)


if __name__ == '__main__':
    unittest.main()
