"""Single-worker CUDA-only Transformers backend for Experiment 1.

Run on the rented NVIDIA server, not on the analysis workstation. The small HTTP
adapter supplies the existing tokenize/chat contract without llama.cpp or GGUF.
JSON output is prompt-guided, NOT grammar-constrained; malformed output is audited.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import time

from .core import digest, save, sha


def token_pieces(text, ids, offsets):
    """Keep source exactly, using monotonic original-character offsets from a fast tokenizer.

    Overlapping offsets (byte-fallback tokens) are assigned to the last token in
    that overlapping group. This keeps cuts outside the group's UTF-8 character.
    The source-token index is not necessarily the retokenized chunk input length.
    """
    if len(ids) != len(offsets) or not ids:
        raise ValueError('Missing token offsets')
    ends = []
    for start, end in offsets:
        if not 0 <= start <= end <= len(text) or (ends and end < ends[-1]):
            raise ValueError('Nonmonotonic tokenizer offsets; unsupported tokenizer')
        ends.append(end)
    cursor, result = 0, []
    for i, (tid, end) in enumerate(zip(ids, ends)):
        # Delaying a cut across overlapping character offsets prevents splitting a token group.
        cut = cursor if i + 1 < len(ids) and offsets[i + 1][0] < end else end
        if i == len(ids) - 1:
            cut = len(text)  # preserve trailing whitespace omitted by tokenizer offsets
        result.append(dict(id=int(tid), piece=text[cursor:cut]))
        cursor = cut
    if ''.join(t['piece'] for t in result) != text:
        raise ValueError('Tokenizer/source round-trip failed')
    return result


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Backend:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: install the CUDA PyTorch build and check the NVIDIA driver')
        if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This CUDA device does not support BF16')
        torch.set_num_threads(args.cpu_threads)
        model_path = Path(args.model)
        if model_path.is_dir():
            hashes = {str(p.relative_to(model_path)): file_hash(p) for p in sorted(model_path.rglob('*'))
                      if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.jinja')}
            revision = digest(hashes)
        else:
            if not args.revision or not re.fullmatch(r'[0-9a-f]{40}', args.revision):
                raise ValueError('For a Hub model, provide its immutable 40-character --revision commit SHA')
            revision = args.revision
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True, trust_remote_code=False)
        if not self.tokenizer.is_fast or not self.tokenizer.chat_template:
            raise ValueError('A fast tokenizer with an instruct chat template is required')
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
            device_map={'': 'cuda:0'}, attn_implementation='sdpa', trust_remote_code=False).eval()
        if any(p.device.type != 'cuda' for p in self.model.parameters()):
            raise RuntimeError('All model weights must be on CUDA; CPU offload is disabled')
        self.torch, self.max_context = torch, args.max_context
        context_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if context_limit and args.max_context > context_limit:
            raise ValueError('Requested context exceeds model configuration')
        self.chat_options = json.loads(args.chat_template_kwargs)
        if not isinstance(self.chat_options, dict):
            raise ValueError('Chat-template kwargs must be a JSON object')
        self.served_model = args.served_model
        meta = dict(backend='transformers-cuda', model=args.served_model, source_model=args.model,
                    model_revision=revision, tokenizer_id=digest([self.tokenizer.backend_tokenizer.to_str(), self.tokenizer.chat_template]),
                    torch_version=torch.__version__, transformers_version=transformers.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), dtype=args.dtype, max_context=args.max_context,
                    cpu_threads=args.cpu_threads, chat_template_kwargs=self.chat_options,
                    generation_config=self.model.generation_config.to_dict(),
                    implementation_hash=sha(Path(__file__).read_text()), json_mode='prompt_only_no_grammar')
        meta['server_id'] = digest(meta)
        self.metadata = meta
        save(args.metadata_out, meta)

    def tokenize(self, body):
        text = body['content']
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return dict(tokens=token_pieces(text, encoded['input_ids'], encoded['offset_mapping']),
                    method='fast_tokenizer_original_offsets', server_id=self.metadata['server_id'])

    def generate(self, body):
        if body.get('model') != self.served_model or body.get('_experiment_server_id') != self.metadata['server_id']:
            raise ValueError('Model/server identity mismatch; initialize study from server metadata')
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(body['messages'], tokenize=False, add_generation_prompt=True, **self.chat_options)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt', truncation=False).to('cuda:0')
        n = inputs['input_ids'].shape[-1]
        cap = body['max_tokens']
        if type(cap) is not int or cap <= 0 or n + cap > self.max_context:
            raise ValueError('Prompt plus output cap exceeds context; no silent truncation')
        temperature = body['temperature']
        torch.manual_seed(body['seed'])
        torch.cuda.manual_seed_all(body['seed'])
        torch.cuda.reset_peak_memory_stats()
        config = copy.deepcopy(self.model.generation_config)
        config.do_sample = temperature > 0
        config.max_new_tokens = cap
        config.num_beams = 1
        config.num_return_sequences = 1
        config.repetition_penalty = 1.0
        config.top_k = 0
        if temperature > 0:
            config.temperature, config.top_p = temperature, body['top_p']
        if config.pad_token_id is None:
            eos = config.eos_token_id
            config.pad_token_id = eos[0] if isinstance(eos, list) else eos
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=config)[0, n:]
        torch.cuda.synchronize()
        ids = output.tolist()
        eos = config.eos_token_id
        stops = eos if isinstance(eos, list) else [eos]
        finish = 'stop' if ids and ids[-1] in stops else 'length'
        return dict(choices=[dict(finish_reason=finish, message=dict(role='assistant', content=self.tokenizer.decode(ids, skip_special_tokens=True)))],
                    usage=dict(prompt_tokens=n, completion_tokens=len(ids)),
                    server_id=self.metadata['server_id'], telemetry=dict(seconds=time.monotonic() - start,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local HF checkpoint directory or Hub ID (text causal LM)')
    p.add_argument('--revision', help='Required immutable Hub commit SHA for remote models')
    p.add_argument('--served-model', default='extraction-cuda')
    p.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    p.add_argument('--max-context', type=int, default=8192)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--port', type=int, default=11439)
    p.add_argument('--metadata-out', type=Path, required=True)
    p.add_argument('--chat-template-kwargs', default='{}', help='Model-specific frozen template options, e.g. enable_thinking')
    args = p.parse_args()
    backend = Backend(args)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            wire = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(wire)))
            self.end_headers()
            try:
                self.wfile.write(wire)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client timeout stays an unknown/failed attempt; no retries.

        def do_GET(self):
            self.reply(200 if self.path == '/metadata' else 404,
                       backend.metadata if self.path == '/metadata' else {'error': 'not found'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    result = backend.tokenize(body)
                elif self.path == '/v1/chat/completions':
                    result = backend.generate(body)
                else:
                    self.reply(404, {'error': 'not found'})
                    return
                self.reply(200, result)
            except Exception as exc:
                self.reply(400 if isinstance(exc, (ValueError, KeyError)) else 500,
                           {'error': type(exc).__name__, 'message': str(exc)})

    # Serial request processing keeps per-request seeds isolated. Bind only locally.
    print(json.dumps(backend.metadata, indent=2), flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
