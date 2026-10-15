"""Describe lexical extraction differences; do not label them semantic validity."""
import argparse,json,itertools,unicodedata
from pathlib import Path
from collections import defaultdict

def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' .,;:!?')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=Path(__file__).parent/'pilot_raw');parser.add_argument('--out',type=Path,default=Path(__file__).parent/'pilot_comparison.json');args=parser.parse_args()
 raw=args.raw;m=json.loads((raw/'manifest.json').read_text());sets=defaultdict(lambda:{'entities':set(),'relations':set(),'entity_evidence_ok':set(),'relation_evidence_ok':set()});details=[];runs=[]
 for j in m['jobs']:
  path=raw/(j['job_id']+'.json')
  if not path.exists():continue
  r=json.loads(path.read_text());p=r.get('parsed');key=(j['doc_id'],j['size'],j['seed']);g=sets[key];usage=r.get('response',{}).get('usage',{})
  runs.append({'id':j['job_id'],'doc':j['doc_id'],'size':j['size'],'seed':j['seed'],'seconds':r['elapsed_seconds'],'errors':r['automatic_errors'],'usage':usage,'raw_entities':len(p.get('entities',[])) if isinstance(p,dict) else None,'raw_relations':len(p.get('relations',[])) if isinstance(p,dict) else None,'finish_reason':r.get('response',{}).get('choices',[{}])[0].get('finish_reason')})
  if not isinstance(p,dict):continue
  ids={e.get('id'):e.get('name','') for e in p.get('entities',[]) if isinstance(e,dict)}
  for e in p.get('entities',[]):
   if not isinstance(e,dict) or not isinstance(e.get('name'),str):continue
   name=norm(e['name']);g['entities'].add(name)
   if isinstance(e.get('evidence'),str) and e['evidence'] and e['evidence'] in j['text']:g['entity_evidence_ok'].add(name)
  for e in p.get('relations',[]):
   if not isinstance(e,dict) or e.get('source') not in ids or e.get('target') not in ids or not isinstance(e.get('predicate'),str):continue
   fact=(norm(ids[e['source']]),norm(e['predicate']),norm(ids[e['target']]))
   g['relations'].add(fact);ok=isinstance(e.get('evidence'),str) and bool(e['evidence']) and e['evidence'] in j['text']
   if ok:g['relation_evidence_ok'].add(fact)
   details.append({'doc':j['doc_id'],'size':j['size'],'seed':j['seed'],'chunk':j['job_id'],'fact':fact,'evidence':e.get('evidence'),'evidence_exact_match':ok})
 groups=[];pairs=[];unique=[]
 for key,g in sorted(sets.items()):
  rr=[r for r in runs if (r['doc'],r['size'],r['seed'])==key]
  groups.append({'doc':key[0],'size':key[1],'seed':key[2],**{k:len(v) for k,v in g.items()},'raw_entities':sum(r['raw_entities'] or 0 for r in rr),'raw_relations':sum(r['raw_relations'] or 0 for r in rr),'empty_calls':sum(r['raw_entities']==0 and r['raw_relations']==0 for r in rr),'unparseable_calls':sum(r['raw_entities'] is None for r in rr),'calls':len(rr),'seconds':sum(r['seconds'] for r in rr),'prompt_tokens':sum(r['usage'].get('prompt_tokens',0) for r in rr),'completion_tokens':sum(r['usage'].get('completion_tokens',0) for r in rr),'calls_with_errors':sum(bool(r['errors']) for r in rr),'truncated':sum(r['finish_reason']=='length' for r in rr)})
 for doc in [r['doc_id'] for r in m['regions']]:
  combos=[((128,17),(256,17)),((128,17),(512,17)),((256,17),(512,17)),((256,17),(256,29))]
  for a,b in combos:
   ga=sets[(doc,*a)];gb=sets[(doc,*b)]
   for kind in ['entities','relations']:
    x,y=ga[kind],gb[kind];pairs.append({'doc':doc,'a':a,'b':b,'kind':kind,'intersection':len(x&y),'union':len(x|y),'jaccard':len(x&y)/len(x|y) if x|y else None})
  for size in [128,256,512]:
   other=set().union(*(sets[(doc,s,17)]['relations'] for s in [128,256,512] if s!=size))
   only=sets[(doc,size,17)]['relations']-other
   unique.append({'doc':doc,'size':size,'lexically_unique_relations':len(only),'facts':sorted(only)})
 report={'disclaimer':'Lexical normalized comparison. Exact evidence presence is NOT validity. No semantic entity/relation alignment performed. One region per document, one primary seed, one extra seed at E256.','completed':len(runs),'planned':len(m['jobs']),'groups':groups,'pairs':pairs,'unique':unique,'runs':runs,'relation_details':details}
 args.out.parent.mkdir(parents=True,exist_ok=True)
 args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'completed':len(runs),'planned':len(m['jobs']),'groups':groups},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
