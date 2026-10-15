import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expanded',ROOT/'experiments/expanded/run_expanded.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ExpandedTests(unittest.TestCase):
 def test_same_request(self):
  r=json.loads((ROOT/'experiments/pilot/pilot_raw/Novel-10146_e256_s17_c00.json').read_text())
  self.assertEqual(m.request(r['job']),r['request'])
 def test_full_coverage(self):
  parts=[b'a']*4096;region={'doc_id':'test','region_id':'test_r00','char_start':50}
  jobs=m.split_region(region,parts)
  self.assertEqual(len(jobs),76)
  for size,seed in m.CONDITIONS:
   group=[j for j in jobs if (j['size'],j['seed'])==(size,seed)]
   self.assertEqual(''.join(j['text'] for j in group),'a'*4096)
   for left,right in zip(group,group[1:]):self.assertEqual(left['char_end'],right['char_start'])
 def test_failed_condition_excluded(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'raw').mkdir()
   job={'job_id':'a','doc_id':'x','region_id':'r','size':256,'seed':17}
   m.save(out/'raw/a.json',{'parsed':None,'automatic_errors':['truncated'],'elapsed_seconds':1,'response':{'choices':[{'finish_reason':'length'}]}})
   m.summarize(out,{'jobs':[job]},'completed')
   row=json.loads((out/'summary.json').read_text())['groups'][0]
   self.assertIsNone(row['entities']);self.assertFalse(row['complete'])
if __name__=='__main__':unittest.main()
