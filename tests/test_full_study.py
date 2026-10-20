import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.experiment1 import core, analysis, study
from experiments.experiment1.cuda_server import token_pieces


class FakeMonitor:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def result(self): return {'gpu_device_peak_memory_mib': None, 'wall_seconds': 0}


class FullStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'study'
        self.corpus = self.root / 'novel.json'
        ids = list(study.DEV_DOCS) + [f'Novel-{90000+i}' for i in range(15)]
        self.text = 'Alice knows Bob. ' * 65 + 'TAIL 끝🙂'
        core.save(self.corpus, [dict(corpus_name=d, context=self.text) for d in ids])
        self.protocol = dict(url='http://fixture', model='fixture', model_revision='rev', tokenizer_id='tok',
                             server_id='server', max_tokens=100, temperature=.2, prompt=core.PROMPT, top_p=.9, timeout=1)
        self.calls = 0
        self.addCleanup(patch.stopall)
        patch('experiments.experiment1.study.Monitor', FakeMonitor).start()
        patch('experiments.experiment1.core.post', self.tokenize).start()

    def tokenize(self, url, body, timeout=300):
        self.assertTrue(url.endswith('/tokenize'))
        return json.dumps({'tokens': [{'id': i, 'piece': c} for i, c in enumerate(body['content'])]})

    def transport(self, url, req, timeout):
        self.calls += 1
        # Valid empty response: tests lifecycle without a model or fabricated validity labels.
        return json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content='{"entities":[],"relations":[]}'))],
                               usage=dict(prompt_tokens=3, completion_tokens=2)))

    def init(self):
        return study.init_study(self.corpus, self.out, self.protocol, per_stratum=2, max_paths=100)

    def stage(self, stage):
        with contextlib.redirect_stdout(io.StringIO()):
            return study.run_stage(self.out, stage, self.transport)

    def policy(self):
        p = self.root / 'policy.json'
        core.save(p, {k: 'Fixture decision, not real evidence' for k in study.POLICY_FIELDS})
        return p

    def test_full_coverage_exact_five_conditions_and_cumulative_reuse(self):
        s = self.init()
        with self.assertRaises(ValueError): self.stage(5)
        with self.assertRaises(ValueError): self.stage(20)
        r = self.stage(1)
        doc = s['documents'][0]['doc_id']
        run = self.out / 'documents' / doc
        m = core.read(run / 'manifest.json')
        self.assertEqual(len({core.condition(j) for j in m['jobs']}), 5)
        self.assertEqual(m['regions'][0]['text'], self.text)
        self.assertEqual(m['regions'][0]['char_start'], 0)
        for cond in {core.condition(j) for j in m['jobs']}:
            chunks = sorted((j for j in m['jobs'] if core.condition(j) == cond), key=lambda j: j['char_start'])
            self.assertEqual(''.join(j['text'] for j in chunks), self.text)
            self.assertTrue(chunks[-1]['text'].endswith('TAIL 끝🙂'))
        before = self.calls
        self.stage(1)
        self.assertEqual(self.calls, before)
        ptr = core.read(run / 'literal.json')
        analysis_dir = run / ptr['analysis']
        self.assertEqual(len(list((analysis_dir / 'graphs').glob('*.json'))), 5)
        self.assertEqual(len(core.read(analysis_dir / 'comparisons.json')), 14)
        self.assertIsNone(r['groups']['development']['valid_novel_delta_mean'])
        self.stage(5)
        self.assertEqual(self.calls, 5 * before)
        with self.assertRaises(ValueError): self.stage(20)
        study.freeze(self.out, self.policy())
        r = self.stage(20)
        self.assertEqual(self.calls, 20 * before)
        self.assertEqual(r['groups']['development']['complete_documents'], 5)
        self.assertEqual(r['groups']['evaluation']['complete_documents'], 15)
        self.assertEqual(sum(len(list(p.glob('analyses/*/graphs/*.json'))) for p in (self.out / 'documents').iterdir()), 100)
        self.assertIsNone(r['groups']['evaluation']['valid_novel_delta_mean'])

    def test_plan_does_not_generate_or_unlock_evaluation(self):
        self.init()
        result = study.plan(self.out, 20)
        self.assertEqual(result['model_generation_calls'], 0)
        self.assertEqual(len(result['documents']), 20)
        self.assertEqual(self.calls, 0)
        with self.assertRaises(ValueError): self.stage(20)

    def test_semantic_primary_contrast_compact_review_and_missing_labels(self):
        self.text = 'Alice knows Bob. Bob helps Cara. ' * 45
        rows = core.read(self.corpus)
        for row in rows: row['context'] = self.text
        core.save(self.corpus, rows)
        s = self.init()
        def extraction(url, req, timeout):
            text = req['messages'][-1]['content'].split('PASSAGE:\n', 1)[1]
            entities, relations = [], []
            if 'Alice knows Bob.' in text:
                entities = [dict(id=n, name=n, type='person', evidence=n) for n in ('Alice', 'Bob')]
                relations = [dict(source='Alice', predicate='knows', target='Bob', polarity='positive',
                                  time='', condition='', evidence='Alice knows Bob.')]
                if len(text) == 512 and 'Bob helps Cara.' in text:
                    entities.append(dict(id='Cara', name='Cara', type='person', evidence='Cara'))
                    relations.append(dict(source='Bob', predicate='helps', target='Cara', polarity='positive',
                                          time='', condition='', evidence='Bob helps Cara.'))
            return json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
                dict(entities=entities, relations=relations))))], usage=dict(prompt_tokens=10, completion_tokens=10)))
        with contextlib.redirect_stdout(io.StringIO()):
            study.run_stage(self.out, 1, extraction)
        doc = s['documents'][0]['doc_id']; run = self.out / 'documents' / doc
        ptr = core.read(run / 'literal.json')
        occ = core.read(run / ptr['analysis'] / 'occurrences.json')
        nodes, facts = occ['entities'], occ['relations']
        classes, rels = {}, {}
        for oid, n in nodes.items(): classes.setdefault(n['name'], []).append(oid)
        for oid, f in facts.items(): rels.setdefault(f['predicate'], []).append(oid)
        decisions = dict(input_hash=occ['input_hash'], entity_classes=[dict(id=n, members=ids, reviewer='fixture', reason='identity')
                         for n, ids in classes.items()], relation_classes=[])
        for predicate, ids in rels.items():
            fact = facts[ids[0]]
            decisions['relation_classes'].append(dict(id=predicate, members=[dict(id=i, label='equivalent') for i in ids],
                reviewer='fixture', reason='equivalence', canonical=dict(source=fact['source'], target=fact['target'],
                predicate=predicate, polarity='positive', time='', condition='')))
        alignment = self.root / 'alignment.json'; core.save(alignment, decisions)
        ptr = study.analyze_document(self.out, doc, alignment)
        out = run / ptr['analysis']
        review = core.read(out / 'review_blind.json')
        self.assertTrue(review)
        self.assertTrue(all('baseline_ref' in r and 'baseline_claims' not in r for r in review))
        baselines = core.read(out / 'review_baselines.json')
        partial = self.root / 'partial.json'; core.save(partial, review)
        study.analyze_document(self.out, doc, alignment, partial)
        self.assertIsNone(study.report(self.out)['documents'][0]['valid_novel_delta'])
        for r in review:
            r.update(supported=True, semantically_novel=r['claim']['predicate'] not in
                     {c['predicate'] for c in baselines[r['baseline_ref']]}, annotator='fixture', rationale='synthetic truth')
        labels = self.root / 'labels.json'; core.save(labels, review)
        study.analyze_document(self.out, doc, alignment, labels)
        result = study.report(self.out)['documents'][0]
        self.assertEqual(result['valid_novel_delta'], 1)
        self.assertEqual(result['modes']['semantic']['candidate_novel_delta'], 1)
        review[0]['baseline_ref'] = 'tampered'
        core.save(labels, review)
        with self.assertRaises(ValueError): study.analyze_document(self.out, doc, alignment, labels)

    def test_document_is_saved_before_next_document_extracts(self):
        s = self.init()
        self.stage(1)
        original = self.transport
        calls_per_doc = self.calls
        def transport(url, req, timeout):
            # At first call of doc3, doc2 must already have its graphs.
            if self.calls == calls_per_doc * 2:
                run = self.out / 'documents' / s['documents'][1]['doc_id']
                ptr = core.read(run / 'literal.json')
                self.assertEqual(len(list((run / ptr['analysis'] / 'graphs').glob('*.json'))), 5)
            return original(url, req, timeout)
        with contextlib.redirect_stdout(io.StringIO()):
            study.run_stage(self.out, 5, transport)

    def test_changed_source_protocol_and_frozen_policy_rejected(self):
        s = self.init()
        self.stage(1)
        run = self.out / 'documents' / s['documents'][0]['doc_id']
        original = core.read(run / 'manifest.json')
        changed = copy.deepcopy(original); changed['protocol']['max_tokens'] += 1
        core.save(run / 'manifest.json', changed)
        with self.assertRaises(ValueError): self.stage(1)
        core.save(run / 'manifest.json', original)
        self.stage(5)
        p = self.policy(); study.freeze(self.out, p)
        core.save(p, {k: 'Changed decision' for k in study.POLICY_FIELDS})
        with self.assertRaises(ValueError): study.freeze(self.out, p)
        changed_study = copy.deepcopy(s); changed_study['analysis']['per_stratum'] = 9
        core.save(self.out / 'study.json', changed_study)
        with self.assertRaises(ValueError): study.load_study(self.out)
        core.save(self.out / 'study.json', s)
        src = self.out / s['documents'][0]['source']
        source = core.read(src); source['regions'][0]['text'] = 'edited'
        core.save(src, source)
        with self.assertRaises(ValueError): study.load_study(self.out)

    def test_failed_attempt_not_retried_and_document_excluded(self):
        self.init()
        def fail(url, req, timeout):
            self.calls += 1
            raise OSError('fixture timeout')
        with contextlib.redirect_stdout(io.StringIO()):
            study.run_stage(self.out, 1, fail)
        before = self.calls
        r = self.stage(1)
        self.assertEqual(self.calls, before)
        row = r['documents'][0]
        self.assertTrue(row['attempted_all'])
        self.assertFalse(row['complete_generation'])
        self.assertEqual(row['modes']['literal']['comparisons'], [])
        self.assertEqual(sum(c['calls'] for c in row['costs'].values()), before)
        self.assertIsNone(row['valid_novel_delta'])

    def test_valid_json_nonobject_response_is_audited_as_failure(self):
        self.init()
        with contextlib.redirect_stdout(io.StringIO()):
            result = study.run_stage(self.out, 1, lambda *args: '[]')
        self.assertTrue(result['documents'][0]['attempted_all'])
        self.assertFalse(result['documents'][0]['complete_generation'])

    def test_empty_semantic_graphs_are_zero_only_after_reviewed_alignment(self):
        s = self.init(); self.stage(1)
        doc = s['documents'][0]['doc_id']; run = self.out / 'documents' / doc
        ptr = core.read(run / 'literal.json')
        alignment = run / ptr['analysis'] / 'alignment_template.json'
        semantic = study.analyze_document(self.out, doc, alignment)
        labels = run / semantic['analysis'] / 'review_blind.json'
        study.analyze_document(self.out, doc, alignment, labels)
        r = study.report(self.out)
        self.assertEqual(r['documents'][0]['valid_novel_delta'], 0)
        self.assertIsNone(r['groups']['development']['document_bootstrap_95_ci'])
        self.stage(1)  # diagnostic resume must not discard reviewed results
        self.assertEqual(study.report(self.out)['documents'][0]['valid_novel_delta'], 0)

    def test_reject_incomplete_corpus_and_wrong_dev_split(self):
        rows = core.read(self.corpus)
        core.save(self.corpus, rows[:-1])
        with self.assertRaises(ValueError): self.init()
        rows[0]['corpus_name'] = 'Novel-99999'
        core.save(self.corpus, rows)
        with self.assertRaises(ValueError): self.init()

    def test_cuda_identity_checks_and_offset_contract(self):
        pieces = token_pieces('가나 끝 ', [1, 2, 3, 4], [(0, 1), (0, 1), (1, 2), (3, 4)])
        self.assertEqual(''.join(t['piece'] for t in pieces), '가나 끝 ')
        self.assertEqual(pieces[0]['piece'], '')
        with self.assertRaises(ValueError): token_pieces('abc', [1, 2], [(1, 3), (0, 1)])
        s = self.init()
        d = s['documents'][0]
        with patch('experiments.experiment1.core.post', return_value=json.dumps({'tokens': [{'piece': 'wrong'}], 'server_id': 'wrong'})):
            with self.assertRaises(ValueError):
                core.prepare(self.out / d['source'], self.root / 'wrong', 'http://fixture', 'm', 'r', 't', 'expected',
                             protocol_overrides={'verify_server_id': True})


class AnalysisScalingTests(unittest.TestCase):
    def test_legacy_list_condition_manifest_remains_readable(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/expanded/results/manifest.json'
        m = core.validate_manifest(core.read(path))
        self.assertEqual(len(m['jobs']), 1140)

    def test_path_cap_marks_unavailable_instead_of_wrong_zero(self):
        g = dict(nodes={c: {} for c in 'ABCD'}, edges={a+b: dict(source=a, target=b, evidence_spans=[])
                     for a in 'ABCD' for b in 'ABCD' if a != b})
        r = analysis.compare(g, g, max_paths=2)
        self.assertIsNone(r['novel_union_two_hop_paths'])
        self.assertTrue(r['a']['paths_capped'])
        self.assertEqual(r['edge_jaccard'], 1)


if __name__ == '__main__':
    unittest.main()
