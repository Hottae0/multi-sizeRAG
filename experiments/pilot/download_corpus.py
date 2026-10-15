import argparse, urllib.request, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
url='https://raw.githubusercontent.com/GraphRAG-Bench/GraphRAG-Benchmark/main/Datasets/Corpus/novel.json'
data=urllib.request.urlopen(url,timeout=60).read(); digest=hashlib.sha256(data).hexdigest()
expected=json.loads((Path(__file__).parent/'pilot_raw/manifest.json').read_text())['corpus_sha256']
if digest!=expected:raise ValueError('Upstream corpus changed; SHA256 does not match the recorded experiment')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(data);print(digest)
