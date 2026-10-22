import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments.experiment1 import core, judgments as j, perspectives, restore, next_document
import test_perspectives as fixtures


def judgment(packet, correction=False):
    checks = {d: dict(status='supported', reason='Synthetic fixture assertion') for d in j.DIMENSIONS}
    quote = packet['source']['text']  # Entire block, unique even in a repeated fixture.
    baseline = packet['baseline_claims'][0]
    bchecks = copy.deepcopy(checks)
    if correction:
        bchecks['direction']['status'] = 'unsupported'
    return dict(support='supported', checks=checks, evidence_quotes=[quote],
        examined_baseline_ids=[b['id'] for b in packet['baseline_claims']], baseline_matches=[baseline['id']],
        baseline_reviews=[dict(claim_id=baseline['id'], support='unsupported' if correction else 'supported',
            checks=bchecks, evidence_quotes=[quote], rationale='Synthetic baseline review')],
        information='new_claim' if correction else 'refinement', aspects=['time_condition'], same_event='yes',
        error_correction=correction, baseline_information='Prior fixture claim', candidate_information='Added fixture claim',
        added_information='Fixture added detail', perspective_explanation='Fixture temporal aspect',
        correction_explanation='Fixture corrected direction' if correction else '', rationale='Synthetic fixture only')


class JudgmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PerspectiveTests('test_empty_graphs_are_zero_with_successful_coverage')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.analysis, _ = self.fixture.analyze()
        self.packets = self.root / 'packets'
        j.prepare(self.analysis, self.packets)
        _, self.data = j.load_packets(self.packets)
        self.packet = next(iter(self.data.values()))

    def transport(self, url, request, timeout):
        self.assertTrue(url.endswith('/v1/chat/completions'))
        p = json.loads(request['messages'][1]['content'])
        return json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(judgment(p, True))))],
                               usage=dict(prompt_tokens=20, completion_tokens=30)))

    def test_evidence_dimensions_and_correction_rules(self):
        proposed = judgment(self.packet, True)
        v = j.validate(proposed, self.packet)
        self.assertTrue(v['error_correction'])
        self.assertFalse(v['different_perspective'])
        for bad in ('quote', 'support', 'correction', 'event', 'screen', 'equivalent'):
            x = copy.deepcopy(proposed)
            if bad == 'quote': x['evidence_quotes'] = ['invented source quote']
            if bad == 'support': x['support'] = 'unsupported'
            if bad == 'correction': x['baseline_reviews'][0].update(support='supported', checks=x['checks'])
            if bad == 'event': x['same_event'] = 'no'
            if bad == 'screen': x['examined_baseline_ids'] = []
            if bad == 'equivalent': x['information'] = 'equivalent'
            with self.assertRaises(ValueError, msg=bad):
                j.validate(x, self.packet)

    def test_missing_qualifier_not_not_applicable_and_ambiguous_quote_pending(self):
        p = copy.deepcopy(self.packet)
        p['claim']['missing_qualifiers'] = ['time']
        x = judgment(p)
        x['checks']['time']['status'] = 'not_applicable'
        with self.assertRaises(ValueError): j.validate(x, p)
        x = judgment(self.packet)
        x['evidence_quotes'] = ['Alice']
        self.assertFalse(j.validate(x, self.packet)['resolved'])

    def test_model_draft_resume_failed_not_resent_and_provenance(self):
        out = self.root / 'drafts'
        first = j.draft(self.packets, out, 'http://fixture/v1', 'fixture', limit=2, transport=self.transport)
        self.assertEqual(first['new_attempts'], 2)
        def bad(*args): return 'not json'
        second = j.draft(self.packets, out, 'http://fixture/v1', 'fixture', limit=20, transport=bad)
        self.assertGreater(second['states']['failed'], 0)
        again = j.draft(self.packets, out, 'http://fixture/v1', 'fixture', limit=20,
                        transport=lambda *args: self.fail('retried a recorded attempt'))
        self.assertEqual(again['new_attempts'], 0)
        with self.assertRaises(ValueError):
            j.draft(self.packets, out, 'http://fixture/v1', 'changed-model', transport=self.transport)
        path = next((out/'raw').glob('*.json'))
        old = core.read(path);old['packet_hash']='foreign';core.save(path,old)
        with self.assertRaises(ValueError):j.draft(self.packets,out,'http://fixture/v1','fixture',transport=self.transport)

    def test_corrections_do_not_inflate_novelty_and_models_not_humans(self):
        out = self.root / 'drafts'
        j.draft(self.packets, out, 'http://fixture/v1', 'fixture', limit=100, transport=self.transport)
        queue = self.root/'queue.json'
        q = j.review_queue(self.packets, out, queue, audit_fraction=0)
        self.assertEqual(q['review_items'], len(self.data))  # All corrections force re-review.
        ev = j.evaluate(self.packets,out,self.analysis,self.root/'ev1.json')
        for c in ev['comparisons']:
            self.assertGreater(c['model_provisional_observed_candidates']['error_correction'],0)
            self.assertEqual(c['model_provisional_observed_candidates']['new_claim'],0)
            self.assertEqual(c['model_provisional_observed_candidates']['different_perspective'],0)
            self.assertIsNone(c['fully_reviewed_candidate_estimates'])
        reviews = core.read(queue)
        for row in reviews['items']:
            row.update(final_judgment=row['model_proposal'],reviewer='second-model-fixture',reviewer_kind='model',rationale='Test re-review')
        core.save(queue,reviews)
        ev = j.evaluate(self.packets,out,self.analysis,self.root/'ev2.json',queue)
        for c in ev['comparisons']:
            self.assertGreater(c['reviewer_kinds']['model'],0)
            self.assertNotIn('human',c['reviewer_kinds'])
            self.assertEqual(c['fully_reviewed_candidate_estimates']['new_claim'],0)

    def test_cuda_identity_seed_contract_and_wrong_server_rejected(self):
        def transport(url, request, timeout):
            self.assertEqual(request['_experiment_server_id'], 'fixture-server')
            self.assertEqual(request['seed'], 17)
            self.assertEqual(request['top_p'], .9)
            wire = json.loads(self.transport(url, request, timeout))
            wire['server_id'] = 'fixture-server'
            return json.dumps(wire)
        s = j.draft(self.packets, self.root/'cuda', 'http://fixture/v1', 'fixture', limit=1,
                    transport=transport, server_id='fixture-server')
        self.assertEqual(s['states']['proposed'],1)
        bad = j.draft(self.packets, self.root/'wrong-server', 'http://fixture/v1', 'fixture', limit=1,
                      transport=self.transport, server_id='fixture-server')
        self.assertEqual(bad['states']['failed'],1)

    def test_packet_tamper_rejected(self):
        path=next((self.packets/'packets').glob('*.json'))
        p=core.read(path);p['source']['text']='changed';core.save(path,p)
        with self.assertRaises(ValueError):j.load_packets(self.packets)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PerspectiveTests('test_empty_graphs_are_zero_with_successful_coverage')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root

    def archive(self, extra=None):
        buff=io.BytesIO()
        with tarfile.open(fileobj=buff,mode='w:gz') as t:
            for p in self.fixture.run.rglob('*.json'):
                b=p.read_bytes();m=tarfile.TarInfo('study/documents/d/'+str(p.relative_to(self.fixture.run)));m.size=len(b);t.addfile(m,io.BytesIO(b))
            if extra:t.addfile(extra,io.BytesIO(b'x') if extra.isfile() else None)
        path=self.root/'snapshot.zip'
        with zipfile.ZipFile(path,'w') as z:z.writestr('experiment1_full.tar.gz',buff.getvalue())
        return path

    def test_nested_zip_restore_is_byte_preserving_and_no_overwrite(self):
        p=self.archive();out=self.root/'restored';s=restore.restore(p,out)
        self.assertEqual(s['runs'][0]['missing_calls'],0)
        for src in self.fixture.run.rglob('*.json'):
            self.assertEqual(src.read_bytes(),(out/'source/study/documents/d'/src.relative_to(self.fixture.run)).read_bytes())
        with self.assertRaises(ValueError):restore.restore(p,out)

    def test_traversal_and_links_fail_without_partial_output(self):
        for name in ('traversal','link'):
            m=tarfile.TarInfo('../escape' if name=='traversal' else 'link')
            if name=='link':m.type=tarfile.SYMTYPE;m.linkname='/tmp/escape'
            else:m.size=1
            p=self.archive(m);out=self.root/name
            with self.assertRaises(ValueError):restore.restore(p,out)
            self.assertFalse(out.exists())


class NextDocumentTests(unittest.TestCase):
    def test_uses_frozen_protocol_and_no_evaluation_bypass(self):
        from test_full_study import FullStudyTests
        f=FullStudyTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.init()
        with patch('experiments.experiment1.core.post', f.tokenize):
            result=next_document.run(f.out,'Novel-30752',limit=2,transport=f.transport)
            self.assertEqual(result['new_attempts'],2)
            self.assertEqual(len(result['conditions']),5)
            m=core.read(Path(result['run'])/'manifest.json')
            for field in ('model','model_revision','tokenizer_id','server_id','temperature','top_p','prompt','max_tokens'):
                self.assertEqual(m['protocol'][field],f.protocol[field])
            again=next_document.run(f.out,'Novel-30752',limit=1,transport=f.transport)
            self.assertEqual(again['recorded_calls'],3)
            with self.assertRaises(ValueError):next_document.run(f.out,'Novel-90000',transport=f.transport)
