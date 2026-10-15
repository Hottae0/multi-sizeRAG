"""Expanded v2 extraction: same settings, 5 documents x 3 regions x 4096 tokens.
Owns a local server for the duration of the batch; saves every response atomically.
"""
import argparse, collections, hashlib, json, random, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pilot'))
from run_chunk_pilot import PROMPT, check, post, sha
from compare_pilot import norm

CONDITIONS=[(128,17),(256,17),(512,17),(1024,17),(256,29)]

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    tmp.replace(path)

def request(job):
    return {'model':'qwen-extraction','messages':[{'role':'system','content':PROMPT},{'role':'user','content':'PASSAGE:\n'+job['text']}],'temperature':.2,'top_p':.9,'seed':job['seed'],'max_tokens':4096,'response_format':{'type':'json_object'},'cache_prompt':False,'repeat_penalty':1.1}

def split_region(region,parts):
    jobs=[]
    for size,seed in CONDITIONS:
        bounds=[0];k=size
        while k<len(parts):
            while k<len(parts):
                try:b''.join(parts[:k]).decode('utf-8');break
                except UnicodeDecodeError:k+=1
            if k<len(parts):bounds.append(k)
            k+=size
        bounds.append(len(parts))
        for idx,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
            text=b''.join(parts[lo:hi]).decode('utf-8')
            start=region['char_start']+len(b''.join(parts[:lo]).decode('utf-8'))
            jobs.append({'job_id':f"{region['region_id']}_e{size}_s{seed}_c{idx:03d}",'doc_id':region['doc_id'],'region_id':region['region_id'],'size':size,'seed':seed,'token_start':lo,'token_end':hi,'char_start':start,'char_end':start+len(text),'text':text,'sha256':sha(text)})
    return jobs

def prepare(corpus_path,out,url):
    digest=hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    expected=json.loads((Path(__file__).resolve().parents[1]/'pilot/pilot_raw/manifest.json').read_text())['corpus_sha256']
    if digest!=expected:raise ValueError('Corpus differs from original pilot')
    target=out/'manifest.json'
    if target.exists():
        m=json.loads(target.read_text())
        if m['corpus_sha256']!=digest or m['prompt_sha256']!=sha(PROMPT):raise ValueError('Manifest mismatch')
        return m
    rng=random.Random(20261015);corpus=json.loads(corpus_path.read_text());indices=sorted(rng.sample(range(len(corpus)),5));regions=[];jobs=[]
    for i in indices:
        doc=corpus[i];text=doc['context'];selected=[]
        for ri in range(3):
            for attempt in range(200):
                anchor=int(len(text)*rng.uniform(.1,.8));start=text.find('. ',anchor)+2
                if start<2:continue
                nearby=text[start:start+50000]
                pieces=post(url,'/tokenize',{'content':nearby,'add_special':False,'with_pieces':True})['tokens']
                parts=[p['piece'].encode() if isinstance(p['piece'],str) else bytes(p['piece']) for p in pieces]
                if b''.join(parts)!=nearby.encode():raise ValueError('Tokenizer round-trip failed')
                if len(parts)<4096:continue
                n=4096
                while True:
                    try:sample=b''.join(parts[:n]).decode();break
                    except UnicodeDecodeError:n-=1
                end=start+len(sample)
                if any(start<b and a<end for a,b in selected):continue
                region={'doc_id':doc['corpus_name'],'region_id':f"{doc['corpus_name']}_r{ri:02d}",'source_index':i,'char_start':start,'char_end':end,'tokens':n,'text':sample,'sha256':sha(sample)}
                selected.append((start,end));regions.append(region);jobs.extend(split_region(region,parts[:n]));break
            else:raise ValueError('Unable to select nonoverlapping regions')
    rng.shuffle(jobs)
    m={'protocol':'expanded-v2-scope-only','corpus_sha256':digest,'corpus_source':'https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Datasets/Corpus/novel.json','sampling_seed':20261015,'selection':'5 uniformly sampled documents; 3 nonoverlapping sentence-start regions per document; random anchors in 10%-80% character range; 4096 tokenizer tokens (UTF-8 safe)','prompt':PROMPT,'prompt_sha256':sha(PROMPT),'conditions':CONDITIONS,'overlap':0,'max_tokens':4096,'regions':regions,'jobs':jobs}
    save(target,m);return m

def summarize(out,m,state):
    groups=collections.defaultdict(list)
    for j in m['jobs']:
        path=out/'raw'/(j['job_id']+'.json')
        groups[(j['doc_id'],j['region_id'],j['size'],j['seed'])].append((j,json.loads(path.read_text()) if path.exists() else None))
    rows=[]
    for key,items in sorted(groups.items()):
        entities=set();relations=set();valid=0;recorded=0;errors=0;seconds=0
        for job,r in items:
            if r is None:continue
            recorded+=1;seconds+=r['elapsed_seconds'];errors+=bool(r['automatic_errors'])
            p=r.get('parsed')
            if not isinstance(p,dict) or r.get('response',{}).get('choices',[{}])[0].get('finish_reason')!='stop':continue
            valid+=1
            ids={e.get('id'):e.get('name','') for e in p.get('entities',[]) if isinstance(e,dict)}
            entities.update(norm(x) for x in ids.values() if isinstance(x,str))
            for e in p.get('relations',[]):
                if isinstance(e,dict) and e.get('source') in ids and e.get('target') in ids and isinstance(e.get('predicate'),str):
                    relations.add((norm(ids[e['source']]),norm(e['predicate']),norm(ids[e['target']])))
        complete=valid==len(items)
        rows.append({'doc':key[0],'region':key[1],'size':key[2],'seed':key[3],'planned':len(items),'recorded':recorded,'normal_parseable':valid,'complete':complete,'entities':len(entities) if complete else None,'relations':len(relations) if complete else None,'calls_with_errors':errors,'seconds':seconds})
    save(out/'summary.json',{'state':state,'note':'Lexical counts only, not semantic validity. Incomplete/truncated conditions excluded.','groups':rows})
    lines=['# 확대 추출 실험', '',f'상태: {state}. 5문서 × 3구간 × 약 4096토큰. 기존 v2 추출 설정 유지, overlap 0.', '', '크기별 조건 전체가 파싱 가능하고 정상 종료한 경우만 개수를 제시한다. 근거/endpoint 오류와 의미적 정확성은 별개다. 사람 검증·QA 평가는 하지 않았다.', '', '|문서/구간|크기|seed|기록/계획|정상 파싱|고유 entity/relation|', '|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"|{r['region']}|{r['size']}|{r['seed']}|{r['recorded']}/{r['planned']}|{r['normal_parseable']}|{str(r['entities'])+' / '+str(r['relations']) if r['complete'] else '비교 제외/진행 중'}|")
    (out/'결과.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--server',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--port',type=int,default=11439);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'raw').mkdir(exist_ok=True)
    url=f'http://127.0.0.1:{a.port}';server=None;awake=None;m=None;done=0
    # Never silently attach to an unrelated model server.
    try:
        urllib.request.urlopen(url+'/health',timeout=2)
    except Exception:pass
    else:raise RuntimeError('Port already in use; choose --port or stop the existing server')
    try:
        log=(a.out/'server.log').open('a')
        server=subprocess.Popen([str(a.server),'--model',str(a.model),'--host','127.0.0.1','--port',str(a.port),'--ctx-size','8192','--gpu-layers','99','--parallel','1','--alias','qwen-extraction','--no-webui'],stdout=log,stderr=subprocess.STDOUT)
        for _ in range(90):
            if server.poll() is not None:raise RuntimeError('Server exited')
            try:
                urllib.request.urlopen(url+'/health',timeout=2);break
            except Exception:time.sleep(1)
        else:raise RuntimeError('Server readiness timeout')
        if sys.platform=='darwin':awake=subprocess.Popen(['caffeinate','-i'])
        m=prepare(a.corpus,a.out,url)
        save(a.out/'environment.json',{'model_name':a.model.name,'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'server_sha256':hashlib.sha256(a.server.read_bytes()).hexdigest(),'context_size':8192,'gpu_layers':99,'parallel':1})
        for j in m['jobs']:
            path=a.out/'raw'/(j['job_id']+'.json');req=request(j)
            if path.exists():
                previous=json.loads(path.read_text())
                if previous['job']!=j or previous['request']!=req:raise ValueError('Checkpoint mismatch')
                if 'response' not in previous:raise RuntimeError('Transport failure checkpoint; inspect before retrying')
                done+=1;continue
            save(a.out/'status.json',{'state':'running','recorded':done,'planned':len(m['jobs']),'current_job':j['job_id'],'updated_at_unix':time.time()})
            start=time.monotonic();record={'job':j,'request':req}
            try:
                response=post(url,'/v1/chat/completions',req,timeout=300);record['response']=response
                try:parsed=json.loads(response['choices'][0]['message']['content']);errors=check(parsed,j['text'])
                except Exception as e:parsed=None;errors=['parse/check error: '+str(e)]
                if response['choices'][0].get('finish_reason')!='stop':errors.append('non-stop finish: '+str(response['choices'][0].get('finish_reason')))
                record.update(parsed=parsed,automatic_errors=errors)
            except Exception as e:
                record.update(parsed=None,automatic_errors=['request failed: '+str(e)],elapsed_seconds=time.monotonic()-start);save(path,record);raise
            record['elapsed_seconds']=time.monotonic()-start;save(path,record);done+=1
            print(json.dumps({'recorded':done,'planned':len(m['jobs']),'job':j['job_id'],'seconds':record['elapsed_seconds'],'errors':record['automatic_errors']}),flush=True)
            if done%10==0:summarize(a.out,m,'running')
        summarize(a.out,m,'completed')
        save(a.out/'status.json',{'state':'completed','recorded':done,'planned':len(m['jobs']),'updated_at_unix':time.time()})
    except BaseException as e:
        save(a.out/'status.json',{'state':'stopped_with_error','recorded':done,'error':str(e),'updated_at_unix':time.time()})
        if m:summarize(a.out,m,'stopped_with_error')
        raise
    finally:
        if server:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill()
        if awake:awake.terminate()

if __name__=='__main__':main()
