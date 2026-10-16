"""Retry only length-terminated requests; preserve originals and checkpoint each retry."""
import argparse, collections, hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pilot'))
from run_chunk_pilot import check
from run_expanded import save, summarize

def post(url,payload):
    req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=86400) as response:return json.load(response)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--server',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--port',type=int,default=11440);a=ap.parse_args()
    a.source=a.source.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);(a.out/'raw').mkdir(exist_ok=True)
    manifest=json.loads((a.source/'manifest.json').read_text());environment=json.loads((a.source/'environment.json').read_text())
    for path,key in [(a.model,'model_sha256'),(a.server,'server_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=environment[key]:raise RuntimeError('Original binary hash mismatch: '+str(path))
    selected=[]
    for job in manifest['jobs']:
        path=a.source/'raw'/(job['job_id']+'.json');r=json.loads(path.read_text())
        if r['response']['choices'][0]['finish_reason']=='length':
            assert r['job']==job
            selected.append((job,r,hashlib.sha256(path.read_bytes()).hexdigest()))
    plan={'protocol':'retry-original-length-only-v1','source':str(a.source),'max_tokens':-1,'context_size':65536,'context_shift':False,'note':'No generation token cap; finite context remains. Non-stop results remain failures. Original inputs, prompts, seeds and sampling parameters preserved.','jobs':[{'job_id':j['job_id'],'source_record_sha256':h} for j,r,h in selected]}
    save(a.out/'retry_plan.json',plan)
    url=f'http://127.0.0.1:{a.port}';server=None;awake=None;done=0;started=time.time()
    def status(state,**kw):save(a.out/'status.json',{'state':state,'pid':os.getpid(),'recorded':done,'planned':len(selected),'started_at_unix':started,'updated_at_unix':time.time(),**kw})
    try:
        try:urllib.request.urlopen(url+'/health',timeout=2)
        except Exception:pass
        else:raise RuntimeError('Port occupied; refusing unrelated server')
        args=[str(a.server),'--model',str(a.model),'--host','127.0.0.1','--port',str(a.port),'--ctx-size','65536','--gpu-layers','99','--parallel','1','--alias','qwen-extraction','--no-webui','--n-predict','-1','--no-context-shift']
        status('loading_model')
        with (a.out/'server.log').open('a') as log:server=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(180):
            if server.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as r:json.load(r)
                break
            except Exception:time.sleep(1)
        else:raise RuntimeError('Server readiness timeout')
        awake=subprocess.Popen(['caffeinate','-i']) if sys.platform=='darwin' else None
        save(a.out/'environment.json',{**environment,'context_size':65536,'max_tokens':-1,'server_args':args})
        for job,original,digest in selected:
            target=a.out/'raw'/(job['job_id']+'.json');req=dict(original['request']);req['max_tokens']=-1
            if target.exists():
                prev=json.loads(target.read_text())
                if prev['request']!=req or prev['source_record_sha256']!=digest:raise RuntimeError('Checkpoint mismatch')
                if 'response' not in prev:raise RuntimeError('Prior transport failure requires inspection')
                done+=1;continue
            status('running',current_job=job['job_id'])
            t=time.monotonic();record={'job':job,'request':req,'source_record_sha256':digest,'original_completion_tokens':original['response']['usage']['completion_tokens']}
            try:
                response=post(url,req);record['response']=response
                if response['usage']['prompt_tokens']!=original['response']['usage']['prompt_tokens']:raise RuntimeError('Prompt token count changed')
                try:
                    parsed=json.loads(response['choices'][0]['message']['content']);errors=check(parsed,job['text'])
                except Exception as e:parsed=None;errors=['parse/check error: '+str(e)]
                if response['choices'][0].get('finish_reason')!='stop':errors.append('non-stop finish: '+str(response['choices'][0].get('finish_reason')))
                record.update(parsed=parsed,automatic_errors=errors)
            except Exception as e:
                record.update(parsed=None,automatic_errors=['request failed: '+str(e)],elapsed_seconds=time.monotonic()-t);save(target,record);raise
            record['elapsed_seconds']=time.monotonic()-t;save(target,record);done+=1
            print(json.dumps({'done':done,'total':len(selected),'job':job['job_id'],'finish_reason':response['choices'][0]['finish_reason'],'tokens':response['usage']['completion_tokens'],'parseable':parsed is not None,'seconds':record['elapsed_seconds']},ensure_ascii=False),flush=True)
        # Build a separate combined view with explicit per-request provenance.
        combined=a.out/'combined';(combined/'raw').mkdir(parents=True,exist_ok=True)
        revised=dict(manifest);revised['protocol']='expanded-v2-with-length-only-retries';revised['max_tokens']='4096 originals; -1 on selected retries';revised['retry_plan']=plan;save(combined/'manifest.json',revised)
        results=[]
        for job in manifest['jobs']:
            retry=a.out/'raw'/(job['job_id']+'.json');source=a.source/'raw'/(job['job_id']+'.json');use=source
            if retry.exists():
                rr=json.loads(retry.read_text());ok=rr.get('parsed') is not None and rr.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='stop'
                results.append({'job_id':job['job_id'],'resolved':ok,'finish_reason':rr.get('response',{}).get('choices',[{}])[0].get('finish_reason'),'completion_tokens':rr.get('response',{}).get('usage',{}).get('completion_tokens'),'seconds':rr['elapsed_seconds'],'automatic_errors':rr['automatic_errors']})
                if ok:use=retry
            (combined/'raw'/source.name).write_bytes(use.read_bytes())
        save(a.out/'retry_results.json',{'retries':results,'resolved':sum(r['resolved'] for r in results),'total':len(results),'seconds':sum(r['seconds'] for r in results)})
        summarize(combined,revised,'completed')
        unresolved=sum(not r['resolved'] for r in results);status('completed' if not unresolved else 'completed_with_unresolved',unresolved=unresolved)
    except BaseException as e:status('stopped_with_error',error=str(e));raise
    finally:
        if server:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        if awake:awake.terminate()

if __name__=='__main__':main()
