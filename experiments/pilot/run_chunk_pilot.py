"""Local pilot: deterministic sampling, exact text chunks, checkpointed extraction."""
import json,random,hashlib,time,urllib.request,re,unicodedata,argparse
from pathlib import Path
from extract_probe import PROMPT as BASE_PROMPT
PROMPT = BASE_PROMPT + " Return compact JSON without indentation. Do not repeat entities or relations. Do not create generic category nodes such as hotel or person merely to express an entity type. A relation must connect two independently identified entities and its evidence must support that specific pair."

def post(url,path,payload,timeout=240):
 req=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def normalize(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' .,;:!?')
def check(parsed,chunk):
 errors=[]
 if not isinstance(parsed,dict) or not all(isinstance(parsed.get(k),list) for k in ('entities','relations')):return ['invalid schema']
 ids=[e.get('id') for e in parsed['entities'] if isinstance(e,dict)]
 if len(set(ids))!=len(ids):errors.append('duplicate ids')
 for k in ['entities','relations']:
  for i,e in enumerate(parsed[k]):
   if not isinstance(e,dict):errors.append(f'{k}:{i}:not object');continue
   fields=['id','name','type','evidence'] if k=='entities' else ['source','predicate','target','evidence']
   if any(not isinstance(e.get(f),str) or not e[f] for f in fields):errors.append(f'{k}:{i}:missing/invalid field')
   ev=e.get('evidence')
   if not isinstance(ev,str) or not ev or ev not in chunk:errors.append(f'{k}:{i}:evidence mismatch')
   if k=='relations' and (e.get('source') not in ids or e.get('target') not in ids):errors.append(f'{k}:{i}:endpoint missing')
 return errors

def main():
 a=argparse.ArgumentParser();a.add_argument('--corpus',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--max-tokens',type=int,default=4096);a.add_argument('--url',default='http://127.0.0.1:11439');args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 corpus=json.loads(args.corpus.read_text());manifest=args.out/'manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['corpus_sha256'] != hashlib.sha256(args.corpus.read_bytes()).hexdigest():raise ValueError('Corpus hash differs from manifest')
  if m['prompt_sha256']!=sha(PROMPT):raise ValueError('Prompt changed: use a new output directory to avoid mixing protocols')
 else:
  rng=random.Random(20261015);indices=sorted(rng.sample(range(len(corpus)),2));regions=[];jobs=[]
  for i in indices:
   d=corpus[i];text=d['context'];anchor=int(len(text)*rng.uniform(.25,.65));start=text.find('. ',anchor)+2
   if start<2:raise ValueError('sentence start unavailable')
   # Token pieces must round-trip byte-for-byte. Cut only at valid UTF-8 boundaries.
   nearby=text[start:start+15000];pieces=post(args.url,'/tokenize',{'content':nearby,'add_special':False,'with_pieces':True})['tokens']
   bs=[]
   for p in pieces:
    part=p['piece'];bs.append(part.encode('utf-8') if isinstance(part,str) else bytes(part))
   assert b''.join(bs)==nearby.encode('utf-8'),'Tokenizer pieces do not round-trip; abort rather than alter corpus'
   ends=[0]
   for b in bs:ends.append(ends[-1]+len(b))
   n=1024
   while True:
    try:region=b''.join(bs[:n]).decode('utf-8');break
    except UnicodeDecodeError:n-=1
   regions.append({'doc_id':d['corpus_name'],'source_index':i,'char_start':start,'char_end':start+len(region),'tokens':n,'text':region,'sha256':sha(region)})
   for size,seed in [(128,17),(256,17),(512,17),(256,29)]:
    bounds=[0];k=size
    while k<n:
     while k<n:
      try:b''.join(bs[:k]).decode('utf-8');break
      except UnicodeDecodeError:k+=1
     if k<n:bounds.append(k)
     k+=size
    bounds.append(n)
    for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
     chunk=b''.join(bs[lo:hi]).decode('utf-8');prefix=b''.join(bs[:lo]).decode('utf-8')
     jobs.append({'job_id':f"{d['corpus_name']}_e{size}_s{seed}_c{j:02d}",'doc_id':d['corpus_name'],'size':size,'seed':seed,'token_start':lo,'token_end':hi,'char_start':start+len(prefix),'char_end':start+len(prefix)+len(chunk),'text':chunk,'sha256':sha(chunk)})
  rng.shuffle(jobs)
  m={'corpus_source':'https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Datasets/Corpus/novel.json','corpus_sha256':hashlib.sha256(args.corpus.read_bytes()).hexdigest(),'sampling_seed':20261015,'selection':'two corpus entries sampled uniformly, one sentence-aligned anchor in 25%-65% character interval; first 1024 tokenizer tokens','prompt':PROMPT,'prompt_sha256':sha(PROMPT),'regions':regions,'jobs':jobs,'scope':'exploratory pilot; literal normalized matching only; no gold graph or human adjudication'}
  manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2))
 for idx,j in enumerate(m['jobs']):
  target=args.out/(j['job_id']+'.json')
  req={'model':'qwen-extraction','messages':[{'role':'system','content':PROMPT},{'role':'user','content':'PASSAGE:\n'+j['text']}],'temperature':.2,'top_p':.9,'seed':j['seed'],'max_tokens':args.max_tokens,'response_format':{'type':'json_object'},'cache_prompt':False,'repeat_penalty':1.1}
  if target.exists():
   previous=json.loads(target.read_text())
   if previous.get('job')!=j or previous.get('request')!=req:raise ValueError('Checkpoint input/config mismatch: '+str(target))
   if previous.get('parsed') is not None and previous.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='stop':continue
   raise ValueError('Incomplete checkpoint: preserve this directory and rerun in a new --out directory: '+str(target))
  start=time.monotonic();record={'job':j,'request':req}
  try:
   r=post(args.url,'/v1/chat/completions',req);record['response']=r;raw=r['choices'][0]['message']['content']
   try:parsed=json.loads(raw);errors=check(parsed,j['text'])
   except Exception as e:parsed=None;errors=['parse error: '+str(e)]
   if r['choices'][0].get('finish_reason')=='length':errors.append('truncated')
   record.update(parsed=parsed,automatic_errors=errors)
  except Exception as e:record.update(parsed=None,automatic_errors=['request failed: '+str(e)])
  record['elapsed_seconds']=time.monotonic()-start;target.write_text(json.dumps(record,ensure_ascii=False,indent=2))
  print(json.dumps({'done':idx+1,'total':len(m['jobs']),'id':j['job_id'],'seconds':round(record['elapsed_seconds'],1),'errors':record['automatic_errors']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
