"""Add full-region E1024 jobs without resampling or changing extraction settings."""
import argparse
import json
from pathlib import Path
from run_chunk_pilot import PROMPT, sha

def prepare(source, out):
    base = json.loads(source.read_text())
    if base['prompt_sha256'] != sha(PROMPT):
        raise ValueError('Baseline prompt differs')
    jobs = []
    for region in base['regions']:
        if region['tokens'] != 1024 or sha(region['text']) != region['sha256']:
            raise ValueError('Expected an intact 1024-token baseline region')
        jobs.append({'job_id':f"{region['doc_id']}_e1024_s17_c00", 'doc_id':region['doc_id'],
                     'size':1024, 'seed':17, 'token_start':0, 'token_end':1024,
                     **{key:region[key] for key in ['char_start','char_end','text','sha256']}})
    manifest = {**base, 'jobs':jobs, 'extension':'E1024 full-region, seed17; unchanged v2 prompt and generation settings'}
    out.mkdir(parents=True,exist_ok=True)
    target = out/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Output manifest differs; choose another directory')
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,default=Path(__file__).parent/'pilot_raw/manifest.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.baseline,args.out)
