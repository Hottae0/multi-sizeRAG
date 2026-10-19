import argparse
import json
from pathlib import Path
from .core import prepare, extract, read, validate_manifest, condition
from .analysis import analyze, evaluate


def main():
    p = argparse.ArgumentParser(description='Experiment 1: single-pass extraction and complementarity')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare', help='Reuse fixed source regions, retokenize with llama.cpp /tokenize')
    q.add_argument('--regions', type=Path, required=True, help='Existing manifest containing regions and corpus_sha256')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--url', required=True, help='Server root, without /v1')
    for name in ('model', 'model-revision', 'tokenizer-id', 'server-id'):
        q.add_argument('--' + name, required=True)
    q.add_argument('--seeds', type=int, nargs='+', default=[17])
    q.add_argument('--boundaries', type=int, nargs='+', default=[0])
    q.add_argument('--max-tokens', type=int, default=4096)
    q.add_argument('--temperature', type=float, default=.2)
    q = sub.add_parser('extract')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--limit', type=int)
    q = sub.add_parser('plan')
    q.add_argument('--run', type=Path, required=True)
    q = sub.add_parser('analyze', help='Read new or original expanded-v2 raw outputs without mutation')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--alignment', type=Path)
    q.add_argument('--rates', type=float, nargs=2, metavar=('INPUT_PER_M', 'OUTPUT_PER_M'))
    q.add_argument('--per-stratum', type=int, default=20)
    q.add_argument('--alignment-pairs', action='store_true')
    q = sub.add_parser('evaluate')
    q.add_argument('--analysis', type=Path, required=True)
    q.add_argument('--annotations', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        m = prepare(a.regions, a.out, a.url, a.model, a.model_revision, a.tokenizer_id, a.server_id,
                    a.seeds, a.boundaries, a.max_tokens, a.temperature)
        print(json.dumps({'jobs': len(m['jobs']), 'gleaning': 0}))
    elif a.command == 'extract':
        print(json.dumps({'new_attempts': extract(a.run, limit=a.limit)}))
    elif a.command == 'plan':
        m = validate_manifest(read(a.run / 'manifest.json'))
        cs = sorted({condition(j) for j in m['jobs']})
        print(json.dumps({'regions': len(m['regions']), 'calls': len(m['jobs']),
                          'conditions': {c: sum(condition(j) == c for j in m['jobs']) for c in cs},
                          'gleaning': 0, 'network_calls': 0}, indent=2))
    elif a.command == 'analyze':
        print(json.dumps(analyze(a.run, a.out, a.alignment, a.rates, a.per_stratum, a.alignment_pairs), indent=2))
    elif a.command == 'evaluate':
        print(json.dumps(evaluate(a.analysis, a.annotations, a.out), indent=2))


if __name__ == '__main__':
    main()
