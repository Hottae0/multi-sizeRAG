"""Whole-document Experiment 1: immutable corpus/protocol, cumulative 1/5/20 stages.

No model/server is started here. Each document is one full-source region in a separate
run, preserving the existing provenance and reviewed-alignment contracts.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import threading
import time

from . import core, analysis

DEV_DOCS = ('Novel-10146', 'Novel-30752', 'Novel-41603', 'Novel-44557', 'Novel-54537')
BASE_CONDITIONS = [dict(size=s, seed=17, boundary=0) for s in core.SIZES] + [dict(size=256, seed=29, boundary=0)]
BASELINE, REPEAT, CROSS = 'e256_s17_b0', 'e256_s29_b0', 'e512_s17_b0'
SAME_SEED = [f'e{s}_s17_b0' for s in core.SIZES]
COMPARISONS = list(itertools.permutations(SAME_SEED, 2)) + [(BASELINE, REPEAT), (REPEAT, BASELINE)]
POLICY_FIELDS = ('alignment_policy', 'validity_policy', 'reviewers_and_adjudication',
                 'development_findings', 'decision_to_expand')


def implementation_hash():
    return core.digest({name: core.sha(Path(__file__).with_name(name).read_text())
                        for name in ('core.py', 'analysis.py', 'study.py', 'cuda_server.py')})


@contextmanager
def locked(root):
    path = Path(root) / '.study.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        yield
    finally:
        path.unlink()


def init_study(corpus, root, protocol, first_doc=DEV_DOCS[0], per_stratum=20, max_paths=1000000):
    root, corpus = Path(root), Path(corpus)
    if root.exists():
        raise ValueError('Use a new study directory; changed settings require a new study')
    rows = core.read(corpus)
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError('Expected all 20 Novel corpus documents')
    docs = {}
    for i, row in enumerate(rows):
        doc, text = row.get('corpus_name'), row.get('context')
        if not isinstance(doc, str) or not re.fullmatch(r'Novel-\d+', doc) or doc in docs:
            raise ValueError('Invalid/duplicate corpus_name')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Missing full document context')
        docs[doc] = dict(doc_id=doc, source_index=i, text=text, sha256=core.sha(text))
    if not set(DEV_DOCS) <= docs.keys() or first_doc not in DEV_DOCS:
        raise ValueError('Development document IDs do not match the agreed split')
    required = {'url', 'model', 'model_revision', 'tokenizer_id', 'server_id',
                'max_tokens', 'temperature', 'prompt', 'top_p', 'timeout'}
    if set(protocol) - {'verify_server_id'} != required or any(protocol[k] is None or not str(protocol[k]).strip() for k in required):
        raise ValueError('Incomplete extraction protocol')
    if protocol['max_tokens'] <= 0 or protocol['timeout'] <= 0 or protocol['temperature'] < 0 or not 0 < protocol['top_p'] <= 1:
        raise ValueError('Invalid decoding/timeout configuration')
    if per_stratum < 1 or max_paths < 1:
        raise ValueError('Invalid analysis budget')
    protocol = dict(protocol, url=protocol['url'].rstrip('/'))
    order = [first_doc] + [d for d in DEV_DOCS if d != first_doc] + sorted(set(docs) - set(DEV_DOCS))
    root.mkdir(parents=True)
    catalog = []
    for doc in order:
        d = docs[doc]
        region = dict(doc_id=doc, region_id=doc + '_full', source_index=d['source_index'],
                      char_start=0, char_end=len(d['text']), text=d['text'], sha256=d['sha256'])
        source = dict(corpus_sha256=core.sha(corpus.read_text()), regions=[region], scope='full_document')
        relative = f'sources/{doc}.json'
        core.save(root / relative, source)
        catalog.append(dict(doc_id=doc, split='development' if doc in DEV_DOCS else 'evaluation',
                            source=relative, source_hash=core.digest(source), text_sha256=d['sha256'],
                            characters=len(d['text'])))
    study = dict(version='experiment1-full-v1', corpus_sha256=core.sha(corpus.read_text()),
                 corpus_source=str(corpus.resolve()), documents=catalog, conditions=BASE_CONDITIONS,
                 comparison_pairs=COMPARISONS, extraction=protocol,
                 analysis=dict(per_stratum=per_stratum, max_paths=max_paths, compact_review=True),
                 implementation_hash=implementation_hash(),
                 completion_policy='all five conditions complete; failures excluded, costs retained')
    study['study_hash'] = core.digest(study)
    core.save(root / 'study.json', study)
    return study


def load_study(root):
    root = Path(root)
    s = core.read(root / 'study.json')
    if s['study_hash'] != core.digest({k: v for k, v in s.items() if k != 'study_hash'}):
        raise ValueError('Study edited; use a new study for protocol/split changes')
    if s['implementation_hash'] != implementation_hash():
        raise ValueError('Code changed; retain old checkout or start a new study')
    for d in s['documents']:
        if core.digest(core.read(root / d['source'])) != d['source_hash']:
            raise ValueError('Whole-document source changed: ' + d['doc_id'])
    return s


def document(root, study, doc_id):
    d = next((d for d in study['documents'] if d['doc_id'] == doc_id), None)
    if d is None:
        raise ValueError('Unknown document')
    return d, Path(root) / 'documents' / doc_id


def frozen(root, study):
    path = Path(root) / 'frozen.json'
    if not path.exists():
        raise ValueError('Freeze the development protocol before accessing evaluation documents')
    f = core.read(path)
    if f['study_hash'] != study['study_hash'] or f['freeze_hash'] != core.digest({k: v for k, v in f.items() if k != 'freeze_hash'}):
        raise ValueError('Freeze record mismatch')
    # Development inputs must remain the exact snapshot on which the decision was based.
    for doc, expected in f['development_snapshots'].items():
        _, run = document(root, study, doc)
        if snapshot(run) != expected:
            raise ValueError('Development raw outputs changed after freeze')
    return f


def ensure_prepared(root, study, d):
    _, run = document(root, study, d['doc_id'])
    if (run / 'manifest.json').exists():
        m = core.validate_manifest(core.read(run / 'manifest.json'))
        source = core.read(Path(root) / d['source'])
        if m['source_manifest_sha256'] != core.sha((Path(root) / d['source']).read_text()) or m['regions'][0]['sha256'] != source['regions'][0]['sha256']:
            raise ValueError('Document run/source mismatch')
        expected = study['extraction']
        if any(m['protocol'].get(k) != v for k, v in expected.items()) or m.get('conditions') != study['conditions']:
            raise ValueError('Prepared document protocol differs from study')
        if m['protocol']['implementation_sha256'] != core.sha(Path(core.__file__).read_text()):
            raise ValueError('Prepared document code differs from study')
        return m
    p = study['extraction']
    return core.prepare(Path(root) / d['source'], run, p['url'], p['model'], p['model_revision'],
                        p['tokenizer_id'], p['server_id'], max_tokens=p['max_tokens'], temperature=p['temperature'],
                        conditions=study['conditions'], protocol_overrides={k: p[k] for k in ('prompt', 'top_p', 'timeout', 'verify_server_id') if k in p})


def snapshot(run):
    """Hash the manifest and all actual raw records; missing records are reflected in count."""
    m = core.read(Path(run) / 'manifest.json')
    raw = {j['job_id']: core.sha(p.read_text()) for j in m['jobs']
           if (p := Path(run) / 'raw' / (j['job_id'] + '.json')).exists()}
    return core.digest([core.digest(m), core.digest(raw)])


class Monitor:
    """Best-effort device-wide GPU memory and runner peak RSS, not server process RSS."""
    def __init__(self):
        self.stop = threading.Event()
        self.peaks = {}
        self.samples = 0
        self.error = None

    def sample(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3, check=True)
            for line in result.stdout.splitlines():
                gpu, used = line.split(',')
                self.peaks[gpu.strip()] = max(self.peaks.get(gpu.strip(), 0), float(used))
            self.samples += 1
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.error = type(exc).__name__

    def loop(self):
        self.sample()
        while not self.stop.wait(5):
            self.sample()

    def __enter__(self):
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)

    def result(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(wall_seconds=time.monotonic() - self.start,
                    runner_lifetime_peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024,
                    gpu_device_peak_memory_mib=self.peaks or None, gpu_samples=self.samples,
                    gpu_monitor_error=self.error,
                    scope='GPU device-wide sampled every 5s (may include other processes); runner RSS excludes model server')


def analyze_document(root, doc_id, alignment=None, annotations=None):
    study = load_study(root)
    d, run = document(root, study, doc_id)
    if d['split'] == 'evaluation':
        frozen(root, study)
    if not (run / 'manifest.json').exists():
        raise ValueError('Document has not been prepared')
    options = study['analysis']
    identity = core.digest([snapshot(run), core.read(alignment) if alignment else None, options, study['comparison_pairs']])
    out = run / 'analyses' / identity
    marker = out / 'complete.json'
    if not marker.exists():
        # Preserve interrupted analyses; never mistake a half-written snapshot for completion.
        if out.exists():
            out.rename(out.with_name(identity + '.interrupted-' + str(time.time_ns())))
        analysis.analyze(run, out, alignment=alignment, per_stratum=options['per_stratum'],
                         comparison_pairs=study['comparison_pairs'], max_paths=options['max_paths'],
                         compact_review=True)
        core.save(marker, dict(snapshot_hash=snapshot(run), analysis_id=identity))
    evaluation_path = None
    if annotations:
        if not alignment:
            raise ValueError('Final study evaluation requires reviewed semantic alignment')
        if core.read(out / 'summary.json')['semantic_jaccard_status'] != 'reviewed':
            raise ValueError('Complete the semantic alignment before final evaluation')
        evaluation_path = out / ('evaluation-' + core.sha(Path(annotations).read_text()) + '.json')
        if not evaluation_path.exists():
            analysis.evaluate(out, annotations, evaluation_path)
    pointer = dict(analysis=str(out.relative_to(run)), snapshot_hash=snapshot(run),
                   evaluation=str(evaluation_path.relative_to(run)) if evaluation_path else None)
    pointer_path = run / ('semantic.json' if alignment else 'literal.json')
    # Re-running identical alignment without labels should not discard its existing evaluation.
    if pointer_path.exists():
        prior = core.read(pointer_path)
        if prior['analysis'] == pointer['analysis'] and evaluation_path is None:
            pointer['evaluation'] = prior['evaluation']
    core.save(pointer_path, pointer)
    return pointer


def plan(root, stage):
    """Tokenize and count only: evaluation text length is not an evaluation outcome."""
    study = load_study(root)
    rows = []
    for d in study['documents'][:stage]:
        m = ensure_prepared(root, study, d)
        rows.append(dict(doc_id=d['doc_id'], split=d['split'], tokens=m['regions'][0]['tokens'],
                         conditions={core.condition(c): sum(core.condition(j) == core.condition(c) for j in m['jobs'])
                                     for c in study['conditions']}, calls=len(m['jobs'])))
    result = dict(stage=stage, documents=rows, calls=sum(r['calls'] for r in rows),
                  output_token_upper_bound=sum(r['calls'] for r in rows) * study['extraction']['max_tokens'],
                  graph_conditions=5, model_generation_calls=0)
    core.save(Path(root) / f'plan-{stage}.json', result)
    return result


def run_stage(root, stage, transport=None):
    study = load_study(root)
    if stage not in (1, 5, 20):
        raise ValueError('Stages are cumulative 1, 5 or 20 documents')
    if stage == 5:
        _, first_run = document(root, study, study['documents'][0]['doc_id'])
        if not (first_run / 'literal.json').exists():
            raise ValueError('Run and inspect stage 1 before expanding to stage 5')
    if stage == 20:
        frozen(root, study)
    for d in study['documents'][:stage]:
        _, run = document(root, study, d['doc_id'])
        m = ensure_prepared(root, study, d)
        # A resume verifies all existing raw requests against the immutable manifest.
        print(json.dumps(dict(event='document_start', doc=d['doc_id'], calls=len(m['jobs']))), flush=True)
        with Monitor() as monitor:
            attempted = core.extract(run, transport or core.post)
        core.save(run / 'telemetry' / (str(time.time_ns()) + '.json'), dict(new_attempts=attempted, **monitor.result()))
        analyze_document(root, d['doc_id'])
        core.save(Path(root) / 'progress.json', dict(stage=stage, last_saved_document=d['doc_id'],
                                                   note='Extraction/diagnostic graph saved; semantic review is separate'))
        print(json.dumps(dict(event='document_graphs_saved', doc=d['doc_id'], new_attempts=attempted)), flush=True)
    return report(root)


def freeze(root, policy):
    study = load_study(root)
    decisions = core.read(policy)
    if any(not isinstance(decisions.get(k), str) or not decisions[k].strip() for k in POLICY_FIELDS):
        raise ValueError('Freeze policy must describe: ' + ', '.join(POLICY_FIELDS))
    snapshots = {}
    for d in study['documents'][:5]:
        _, run = document(root, study, d['doc_id'])
        if not (run / 'literal.json').exists():
            raise ValueError('Run all five development documents before freeze')
        ptr = core.read(run / 'literal.json')
        summary = core.read(run / ptr['analysis'] / 'summary.json')
        if summary['recorded_calls'] != summary['total_calls'] or ptr['snapshot_hash'] != snapshot(run):
            raise ValueError('Development attempts/analysis are incomplete or stale')
        snapshots[d['doc_id']] = ptr['snapshot_hash']
    f = dict(study_hash=study['study_hash'], policy=decisions, development_snapshots=snapshots)
    f['freeze_hash'] = core.digest(f)
    path = Path(root) / 'frozen.json'
    if path.exists() and core.read(path) != f:
        raise ValueError('Frozen policy cannot be changed; use a new study')
    core.save(path, f)
    return f


def bootstrap(values, repeats=2000):
    if len(values) < 2:
        return None
    rng = random.Random(17)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    return [samples[int(.025 * repeats)], samples[int(.975 * repeats) - 1]]


def report(root):
    study = load_study(root)
    rows = []
    for d in study['documents']:
        _, run = document(root, study, d['doc_id'])
        row = dict(doc_id=d['doc_id'], split=d['split'], characters=d['characters'],
                   prepared=False, attempted_all=False, complete_generation=False,
                   semantic_status='unavailable', valid_novel_delta=None, modes={})
        if not (run / 'manifest.json').exists():
            rows.append(row)
            continue
        m = core.read(run / 'manifest.json')
        row.update(prepared=True, tokens=m['regions'][0]['tokens'],
                   planned_calls={core.condition(c): sum(core.condition(j) == core.condition(c) for j in m['jobs']) for c in study['conditions']})
        current = snapshot(run)
        for mode in ('literal', 'semantic'):
            path = run / (mode + '.json')
            if not path.exists():
                continue
            ptr = core.read(path)
            if ptr['snapshot_hash'] != current:
                row[mode + '_stale'] = True
                continue
            out = run / ptr['analysis']
            summary = core.read(out / 'summary.json')
            row.update(attempted_all=summary['recorded_calls'] == summary['total_calls'],
                       complete_generation=bool(summary['common_complete_regions']),
                       costs=summary['costs'], errors=dict(truncated=summary['truncated_calls'],
                       item_or_generation=summary['calls_with_item_or_generation_errors'], empty=summary['empty_calls']))
            if mode == 'semantic':
                row['semantic_status'] = summary['semantic_jaccard_status']
            comps = [r for r in core.read(out / 'comparisons.json') if r['mode'] == mode]
            main = {r['added']: r['metrics'] for r in comps if r['baseline'] == BASELINE}
            row['modes'][mode] = dict(comparisons=comps, candidate_novel_delta=
                main[CROSS]['novel_edges_b_minus_a'] - main[REPEAT]['novel_edges_b_minus_a']
                if CROSS in main and REPEAT in main else None)
            if mode == 'semantic' and row['semantic_status'] == 'reviewed' and ptr['evaluation']:
                ev = core.read(run / ptr['evaluation'])
                if ev['input_hash'] != summary['input_hash']:
                    raise ValueError('Evaluation snapshot mismatch')
                row['evaluation'] = ev['comparisons']
                valid = {r['added']: r['estimated_valid_novel'] for r in ev['comparisons'] if r['baseline'] == BASELINE}
                if valid.get(CROSS) is not None and valid.get(REPEAT) is not None:
                    row['valid_novel_delta'] = valid[CROSS] - valid[REPEAT]
        rows.append(row)
    groups = {}
    for split in ('development', 'evaluation'):
        selected = [r for r in rows if r['split'] == split]
        values = [r['valid_novel_delta'] for r in selected if r['valid_novel_delta'] is not None]
        metrics = {}
        for mode in ('literal', 'semantic'):
            for a, b in study['comparison_pairs']:
                comps = [c for r in selected for c in r['modes'].get(mode, {}).get('comparisons', [])
                         if c['baseline'] == a and c['added'] == b and
                         (mode == 'literal' or r['semantic_status'] == 'reviewed')]
                metric_means = {}
                for metric in ('node_jaccard', 'edge_jaccard', 'novel_edges_b_minus_a', 'component_bridging_edges',
                               'novel_union_two_hop_paths', 'novel_evidence_characters_on_shared_edges'):
                    vs = [c['metrics'][metric] for c in comps if c['metrics'][metric] is not None]
                    metric_means[metric] = dict(n=len(vs), mean=sum(vs) / len(vs) if vs else None)
                metrics[f'{mode}:{a}->{b}'] = metric_means
        groups[split] = dict(target_documents=len(selected), attempted_documents=sum(r['attempted_all'] for r in selected),
                             complete_documents=sum(r['complete_generation'] for r in selected),
                             valid_delta_documents=len(values), valid_novel_delta_mean=sum(values) / len(values) if values else None,
                             document_bootstrap_95_ci=bootstrap(values), comparison_macro=metrics,
                             excluded_or_pending=[r['doc_id'] for r in selected if r['valid_novel_delta'] is None])
    result = dict(study_hash=study['study_hash'], frozen=(Path(root) / 'frozen.json').exists(), documents=rows, groups=groups,
                  notes=['20 whole documents; 5 development and 15 evaluation, never pooled for primary inference.',
                         'Six size pairs (12 directions) plus same-size repeat (2 directions); primary contrast uses shared E256 baseline.',
                         'Semantic equality does not establish source validity; independent annotations are required.',
                         'Bootstrap resamples documents only; it does not model within-document annotation/sampling uncertainty.',
                         'No cost matching, QA claim, automatic expansion based on positive results, or hidden retries.'])
    core.save(Path(root) / 'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('init')
    q.add_argument('--corpus', type=Path, required=True)
    q.add_argument('--study', type=Path, required=True)
    q.add_argument('--first-doc', default=DEV_DOCS[0], choices=DEV_DOCS)
    for name in ('url', 'model', 'model-revision', 'tokenizer-id', 'server-id'):
        q.add_argument('--' + name, required=name == 'url')
    q.add_argument('--server-metadata', type=Path, help='CUDA backend metadata JSON; supplies immutable model/tokenizer/server IDs')
    q.add_argument('--max-tokens', type=int, default=4096)
    q.add_argument('--temperature', type=float, default=.2)
    q.add_argument('--top-p', type=float, default=.9)
    q.add_argument('--timeout', type=int, default=300)
    q.add_argument('--prompt-file', type=Path)
    q.add_argument('--per-stratum', type=int, default=20)
    q.add_argument('--max-paths', type=int, default=1000000)
    for name in ('plan', 'run', 'freeze', 'analyze', 'report'):
        q = sub.add_parser(name)
        q.add_argument('--study', type=Path, required=True)
        if name in ('run', 'plan'):
            q.add_argument('--stage', type=int, choices=(1, 5, 20), required=True)
        elif name == 'freeze':
            q.add_argument('--policy', type=Path, required=True)
        elif name == 'analyze':
            q.add_argument('--doc', required=True)
            q.add_argument('--alignment', type=Path)
            q.add_argument('--annotations', type=Path)
    a = p.parse_args()
    if a.command == 'init':
        protocol = {k: getattr(a, k) for k in ('url', 'model', 'model_revision', 'tokenizer_id', 'server_id',
                                               'max_tokens', 'temperature', 'top_p', 'timeout')}
        if a.server_metadata:
            meta = core.read(a.server_metadata)
            for k in ('model', 'model_revision', 'tokenizer_id', 'server_id'):
                protocol[k] = meta[k]
            protocol['verify_server_id'] = True
        if any(protocol[k] is None for k in ('model', 'model_revision', 'tokenizer_id', 'server_id')):
            p.error('Supply --server-metadata or all model/tokenizer/server identifiers')
        protocol['prompt'] = a.prompt_file.read_text() if a.prompt_file else core.PROMPT
        result = init_study(a.corpus, a.study, protocol, a.first_doc, a.per_stratum, a.max_paths)
    else:
        with locked(a.study):
            if a.command == 'plan':
                result = plan(a.study, a.stage)
            elif a.command == 'run':
                result = run_stage(a.study, a.stage)
            elif a.command == 'freeze':
                result = freeze(a.study, a.policy)
            elif a.command == 'analyze':
                result = analyze_document(a.study, a.doc, a.alignment, a.annotations)
                report(a.study)
            else:
                result = report(a.study)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
