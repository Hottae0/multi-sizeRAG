# 전체 원문 Experiment 1 — CUDA 서버 실행 안내

> 2026-10-22 측정 변경: 새 사실·구체화·다른 측면의 최대 청크 블록 비교는 [FACT_PERSPECTIVES.md](FACT_PERSPECTIVES.md)를 사용한다. 이 문서는 기존 v1 추출/분석 재현용으로 유지한다. 기존 study/raw를 수정하지 않는다.

CPU 4코어 / RAM 128GB / A100 80GB 1개 / 저장공간 500GB 서버를 기준으로 준비했습니다.
**실제 GPU 추론은 아직 검증하지 않았습니다.** 아래 backend와 흐름을 서버에서 첫 개발 문서로 검증해 주세요.
모델은 임의로 확정하지 않았습니다. Text instruct causal LM의 로컬 Hugging Face checkpoint를 사용하거나
Hub ID와 immutable revision을 지정합니다. 기존 4B GGUF 사전실험과는 별도 프로토콜입니다.

## 1. 환경 준비

Python 3.11 가상환경을 권장합니다. 먼저 `nvidia-smi`로 A100과 드라이버를 확인합니다.
서버의 드라이버에 맞는 CUDA PyTorch wheel은 [공식 설치 선택기](https://pytorch.org/get-started/locally/)의 명령을 사용합니다.
CUDA toolkit 버전만으로 설치 성공을 가정하지 않습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 위 공식 선택기의 CUDA용 torch 설치 명령을 여기에서 실행합니다.
python -m pip install -r experiments/experiment1/requirements-cuda.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())'
mkdir -p runs/cuda_environment
python -m pip freeze > runs/cuda_environment/pip-freeze.txt
nvidia-smi > runs/cuda_environment/nvidia-smi.txt
```

서버는 단일 요청을 순차 처리합니다. Seed를 요청별로 고정하며 CUDA가 없으면 실패합니다.
CPU로 추론을 대체하거나 모델을 CPU에 자동 offload하지 않습니다. 모델이 VRAM에 들어가지 않으면
개발 단계에서 모델/정밀도를 다시 정하고 새 study를 만듭니다. CPU 4코어에 맞춰 기본 thread 수는 4입니다.

## 2. CUDA 모델 서버 시작 — 터미널 A

`/path/to/hf-model`은 실제 모델 디렉터리로 교체합니다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
python -m experiments.experiment1.cuda_server \
  --model /path/to/hf-model \
  --dtype bfloat16 --max-context 8192 --cpu-threads 4 \
  --port 11439 --metadata-out runs/cuda_environment/server.json
```

Hub에서 로드할 경우 `--model 조직/모델명 --revision 40자리커밋SHA`를 사용합니다.
Local checkpoint는 구성·가중치·tokenizer 파일을 hash하므로 초기 metadata 생성에 시간이 걸릴 수 있습니다.
모델별 thinking 옵션 등이 필요하면 `--chat-template-kwargs '{"enable_thinking": false}'`처럼 지정하되,
해당 모델에서 지원하는지 확인하고 개발 후 고정합니다. 기본값은 `{}`입니다.

서버는 localhost:11439에만 바인딩합니다. 모델 서버와 아래 실험 명령은 같은 서버에서 실행합니다.
요청에 전달하는 JSON 형식 지시는 prompt 기반이며 grammar decoding은 구현하지 않았습니다.
JSON 오류·반복·잘림은 자동 검사 대상입니다. Context 초과 시 입력을 자르지 않고 오류로 기록합니다.

서버 metadata에는 모델 revision/hash, tokenizer/chat template hash, torch/transformers/CUDA 버전,
GPU, dtype, context와 generation config를 기록합니다. CUDA 동일 seed도 환경을 넘는 bitwise 재현을 보장하지 않습니다.

## 3. Study 초기화와 1문서 계획 — 터미널 B

원본 `novel.json`은 GraphRAG-Bench 형식의 20개 항목 배열이어야 합니다.
각 항목의 `corpus_name`을 ID로, `context`를 **잘라내지 않은 전체 원문**으로 읽습니다.
원본은 외부에서 준비하며 자동 다운로드하거나 과거 region manifest를 재사용하지 않습니다.

```bash
python -m experiments.experiment1.study init \
  --corpus /path/to/novel.json --study runs/novel_final \
  --url http://127.0.0.1:11439 \
  --server-metadata runs/cuda_environment/server.json \
  --first-doc Novel-10146 \
  --max-tokens 4096 --temperature 0.2 --top-p 0.9 --timeout 600 \
  --per-stratum 20
python -m experiments.experiment1.study plan --study runs/novel_final --stage 1
python -m experiments.experiment1.study run --study runs/novel_final --stage 1
```

`plan`은 토큰화와 호출 수 계산만 수행하며 모델 생성은 하지 않습니다.
`--stage 20`으로 전체 호출 수를 미리 계산할 수도 있습니다. 평가군의 길이 계획은 결과 평가가 아닙니다.
5조건은 E128/s17, E256/s17, E512/s17, E1024/s17, E256/s29, boundary0입니다.
서버 metadata로 고정한 identity가 tokenize/추론 응답과 다르면 거부합니다.

첫 문서에서도 처음부터 끝까지 처리합니다. 실제 입력 토큰 수는 chat template·prompt를 포함하므로
청크 목표 크기와 다릅니다. HF 청킹은 fast tokenizer의 원문 offset을 사용하며 byte-fallback 겹침은 안전한 경계로 이동합니다.
마지막 꼬리와 모든 원문 문자를 보존합니다. 토큰/문자 offset과 원문 hash는 manifest에서 확인합니다.

## 4. 의미 정렬과 원문 판정

문서별 run은 `runs/novel_final/documents/Novel-10146/`입니다.
`literal.json`의 `analysis` 경로가 현재 점검용 분석 snapshot입니다.
그 디렉터리의 `occurrences.json`, `regions.json`, `alignment_template.json`을 사용하여 검토된 정렬 파일을 작성합니다.
기존 [README의 정렬 스키마](README.md#의미-정렬-추출-뒤-그래프-비교-전)를 따릅니다.

```bash
python -m experiments.experiment1.study analyze \
  --study runs/novel_final --doc Novel-10146 --alignment /path/to/reviewed_alignment.json
```

`semantic.json`에 연결된 분석 디렉터리에서 다음을 함께 판정자에게 제공합니다.

- `review_blind.json`: claim, 불투명 baseline_ref, 판정 필드를 제공합니다.
- `review_baselines.json`: baseline_ref에 대응하는 기준 그래프의 **전체 claim**을 제공합니다.
- `regions.json`: 전체 원문을 제공합니다.

조건 정보가 있는 `review_key.json`, raw/occurrences는 blind 판정자에게 제공하지 않습니다.
`baseline_ref`만 보고 판정하지 않으며 연결된 전체 기준 claim을 확인해야 합니다.
두 판정자가 독립 검토한 뒤 불일치를 합의하여 최종 annotations 파일을 만듭니다.
`id`, `region_id`, `claim`, `baseline_ref`는 수정하지 않습니다.

```bash
python -m experiments.experiment1.study analyze \
  --study runs/novel_final --doc Novel-10146 \
  --alignment /path/to/reviewed_alignment.json --annotations /path/to/adjudicated_labels.json
python -m experiments.experiment1.study report --study runs/novel_final
```

정렬과 독립 판정은 자동으로 수행되지 않습니다. 완전 의미 정렬 및 필요한 판정이 없으면 유효 신규 수는 null입니다.
기존의 구간 CLI `evaluate`도 compact baseline_ref 양식을 지원합니다.

## 5. 총 5문서로 확대

```bash
python -m experiments.experiment1.study plan --study runs/novel_final --stage 5
python -m experiments.experiment1.study run --study runs/novel_final --stage 5
```

첫 문서의 raw 응답은 재요청하지 않습니다. 나머지 4문서 각각에 대해 추출 → 점검·literal 그래프 저장을 마친 뒤 다음 문서로 진행합니다.
Semantic 분석·독립 원문 판정은 문서별로 별도 수행합니다.
모델/프롬프트/출력 상한을 바꾸면 **새 study**를 생성합니다. 이전 실패를 지우거나 manifest를 편집하지 않습니다.

## 6. 설정 동결 후 총 20문서로 확대

실제 개발 결과와 최종 판정 정책을 작성한 JSON을 준비합니다. 다음 필드는 빈 문자열일 수 없습니다.
아래 설명을 그대로 복사해 동결하지 말고 실제 결정 내용으로 작성합니다.

```json
{
  "alignment_policy": "별칭·동명이인·방향·극성·시점·조건의 동치 판정 규칙을 작성합니다.",
  "validity_policy": "원문 지지와 전체 baseline 대비 의미 신규성 기준을 작성합니다.",
  "reviewers_and_adjudication": "독립 판정자 구성 및 불일치 합의 절차를 작성합니다.",
  "development_findings": "5문서의 현상·실패·시간·메모리·검토 상태를 작성합니다.",
  "decision_to_expand": "설정·표집 수·비용을 확인한 후 평가군으로 확대하는 이유를 작성합니다."
}
```

```bash
python -m experiments.experiment1.study plan --study runs/novel_final --stage 20
python -m experiments.experiment1.study freeze --study runs/novel_final --policy /path/to/final_policy.json
python -m experiments.experiment1.study run --study runs/novel_final --stage 20
```

`freeze` 전에는 평가 15문서 추출이 거부됩니다. `freeze`는 검토의 질을 자동 증명하지 않으며
개발 5문서의 추출 시도·분석 저장과 정책 기록을 확인하는 절차입니다. 실제 검토 상태는 정직하게 기록합니다.
개발 결과를 보고 확대하지 않을 수도 있습니다. 평가군 결과를 본 뒤 유리한 설정으로 바꾸지 않습니다.
같은 설정이면 이미 처리한 5문서는 재사용하고 나머지 15문서만 새로 추출합니다.

## 7. 재개·산출물·성능 한계

같은 `run --stage` 명령으로 재개합니다. 성공·실패·결과 불명 started는 재요청하지 않습니다.
중단 시 `.study.lock`/`.extract.lock`이 남으면 실행 중인 프로세스가 없는지 확인한 후 해당 stale lock만 제거합니다.
Raw·manifest·study.json은 지우거나 편집하지 않습니다. 실행 코드가 바뀌면 기존 checkout을 유지하거나 새 study를 만듭니다.

- `study.json`, `sources/`: 전체 20문서, 5/15 분리, 프로토콜·원문 hash를 보존합니다.
- 문서별 `manifest.json`, `raw/`: 모든 청크와 최초 요청·응답·오류를 보존합니다.
- `analyses/<hash>/`: audit, occurrence, 조건별 graph, 14방향 비교와 검토 자료를 보존합니다.
- `literal.json`, `semantic.json`: 현재 snapshot을 가리킵니다. 미완료 분석은 완료로 재사용하지 않습니다.
- `report.json`: 개발/평가별 지표, 문서별 paired delta, 실패·미판정·비용을 분리합니다.
- `telemetry/`: 실행 구간 시간, runner RSS, GPU device-wide 표본 peak를 기록합니다.
- CUDA raw response의 `telemetry`: 서버의 요청별 CUDA allocated/reserved peak를 기록합니다.

Runner RSS는 모델 서버의 host RAM이 아닙니다. GPU device-wide 값에는 다른 프로세스가 포함될 수 있으며
5초 표본의 peak는 순간 peak를 놓칠 수 있습니다. CUDA allocator peak와 함께 해석합니다.
500GB 저장소에 충분한지는 실제 모델/cache/raw 크기로 확인합니다. `data/`, `runs/`, 모델 가중치는 Git에 올리지 않습니다.

2-hop 경로 열거는 기본 100만 개 상한을 넘으면 null로 보고합니다. `--max-paths`는 init에서 고정합니다.
기존 alignment-pairs 전수 생성은 큰 문서에서 비싸므로 study는 자동 실행하지 않습니다.
한 문서라도 한 조건의 생성이 실패하면 전체 조건 공통 비교에서는 그 문서를 제외합니다.
부분 그래프와 모든 비용은 남으며, 이를 전체 문서 성공으로 표시하지 않습니다.

## 참고 구현 문서

- [Transformers 모델 적재](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model)
- [PyTorch CUDA 가용성](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
- [PyTorch BF16 확인](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
