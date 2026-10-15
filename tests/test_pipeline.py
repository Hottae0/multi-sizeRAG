import hashlib
import json
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'experiments/pilot'
sys.path.insert(0, str(PILOT))
from align_relations import align
from extract_names import link_names

class PipelineTests(unittest.TestCase):
    def test_inverse_and_alias_alignment(self):
        self.assertEqual(
            align('Novel-10146', ['the winslow', 'was constructed by', 'J.M. Winslow']),
            align('Novel-10146', ['J.M. Winslow', 'constructed hotel', 'Winslow house']))

    def test_no_repair_of_incorrect_direction_or_negation(self):
        self.assertNotEqual(
            align('Novel-10146', ['Henry M. Rice', 'constructed by', 'Rice house']),
            align('Novel-10146', ['Henry M. Rice', 'constructed', 'Rice house']))
        self.assertNotEqual(align('x', ['A','visited','B']), align('x', ['A','never visited','B']))
        self.assertEqual(align('Novel-10146',['ominie hotel','near','x'])[0], 'ominie hotel')

    def test_unknown_name_is_not_invented(self):
        parsed = {'entities':[{'name':'Alice','type':'person','evidence':'Alice'}],
                  'relations':[{'head':'Alice','predicate':'knows','tail':'Bob','evidence':'Alice knows Bob'}]}
        result = link_names(parsed, 'Alice knows Bob')
        self.assertEqual(result['relations'], [])
        self.assertIn('relation 0: unknown endpoint', result['errors'])

    def test_names_are_linked_without_model_ids(self):
        parsed = {'entities':[{'name':x,'type':'person','evidence':x} for x in ['Alice','Bob']],
                  'relations':[{'head':'Alice','predicate':'knows','tail':'Bob','evidence':'Alice knows Bob'}]}
        result = link_names(parsed, 'Alice knows Bob')
        self.assertEqual(result['errors'], [])
        self.assertEqual((result['relations'][0]['source'], result['relations'][0]['target']), ('e1','e2'))

    def test_recorded_chunks_cover_identical_regions(self):
        m = json.loads((PILOT/'pilot_raw/manifest.json').read_text())
        for region in m['regions']:
            for size, seed in [(128,17),(256,17),(512,17),(256,29)]:
                jobs = sorted((j for j in m['jobs'] if (j['doc_id'],j['size'],j['seed'])==(region['doc_id'],size,seed)), key=lambda j:j['token_start'])
                self.assertEqual(''.join(j['text'] for j in jobs),region['text'])
                self.assertEqual(jobs[0]['char_start'],region['char_start'])
                self.assertEqual(jobs[-1]['char_end'],region['char_end'])
                for left,right in zip(jobs,jobs[1:]):
                    self.assertEqual(left['token_end'],right['token_start'])
                    self.assertEqual(left['char_end'],right['char_start'])

    def test_name_diagnostic_preserves_raw_and_recomputes_validation(self):
        for folder in ['name_endpoint_raw', 'name_endpoint_schema_raw']:
            for path in (PILOT/folder).glob('*.json'):
                record = json.loads(path.read_text())
                self.assertEqual(record['linked'], link_names(record['parsed'],record['job']['text']))
                self.assertEqual(record['request']['seed'],record['job']['seed'])
                self.assertEqual(record['request']['messages'][1]['content'],'PASSAGE:\n'+record['job']['text'])
                self.assertEqual(record['prompt_sha256'],hashlib.sha256(record['request']['messages'][0]['content'].encode()).hexdigest())
                self.assertEqual(record['response']['choices'][0]['finish_reason'],'stop')

    def test_e1024_reuses_exact_baseline_regions(self):
        from prepare_e1024 import prepare
        base = json.loads((PILOT/'pilot_raw/manifest.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            extension = prepare(PILOT/'pilot_raw/manifest.json',Path(d))
            self.assertEqual(extension['prompt_sha256'],base['prompt_sha256'])
            self.assertEqual(len(extension['jobs']),2)
            for region,job in zip(base['regions'],extension['jobs']):
                self.assertEqual(job['text'],region['text'])
                self.assertEqual(job['sha256'],region['sha256'])
                self.assertEqual(job['token_end']-job['token_start'],1024)
                self.assertEqual(job['char_end']-job['char_start'],len(job['text']))
            self.assertEqual(prepare(PILOT/'pilot_raw/manifest.json',Path(d)),extension)

    def test_e1024_generation_settings_are_matched(self):
        baseline=json.loads((PILOT/'pilot_raw/Novel-10146_e512_s17_c00.json').read_text())
        expected={k:v for k,v in baseline['request'].items() if k!='messages'}
        manifest=json.loads((PILOT/'e1024_raw/manifest.json').read_text())
        for job in manifest['jobs']:
            row=json.loads((PILOT/'e1024_raw'/f"{job['job_id']}.json").read_text())
            self.assertEqual(row['job'],job)
            self.assertEqual({k:v for k,v in row['request'].items() if k!='messages'},expected)
            self.assertEqual(row['request']['messages'][0]['content'],manifest['prompt'])
            self.assertEqual(row['request']['messages'][1]['content'],'PASSAGE:\n'+job['text'])
            if row['response']['choices'][0]['finish_reason']=='length':
                self.assertIn('truncated',row['automatic_errors'])

    def test_raw_provenance_and_completion(self):
        m = json.loads((PILOT/'pilot_raw/manifest.json').read_text())
        self.assertEqual(len(m['jobs']),36)
        for job in m['jobs']:
            row = json.loads((PILOT/'pilot_raw'/f"{job['job_id']}.json").read_text())
            self.assertEqual(row['job'],job)
            self.assertEqual(hashlib.sha256(job['text'].encode()).hexdigest(),job['sha256'])
            self.assertEqual(row['request']['messages'][0]['content'],m['prompt'])
            self.assertEqual(row['request']['messages'][1]['content'],'PASSAGE:\n'+job['text'])
            self.assertEqual(row['response']['choices'][0]['finish_reason'],'stop')

if __name__ == '__main__':
    unittest.main()
