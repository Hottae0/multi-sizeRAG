# multi-sizeRAG

같은 원문을 서로 다른 크기로 나누면 **유효한 entity와 relation을 상호보완적으로 얻는가?** 이를 확인하기 위한 Qwen 로컬 추출 파일럿이다. 현재 구현 범위는 다운로드 → 토큰 청킹 → 관계 추출 → 근거·endpoint 검사 → 비교·부분 정렬이다. 검색·생성·QA 평가와 완성된 multi-size GraphRAG 시스템은 아직 구현하지 않았다.

## 현재까지 확인한 것

- GraphRAG-Bench Novel의 2문서에서 각 1,024토큰을 사용했다. E128/E256/E512, seed 17 및 E256 seed 29 대조군으로 **36회** 실행했다.
- 모델은 **Qwen3-VL-4B-Instruct Q4_K_M**이다. 텍스트만 입력했다. 일반 Qwen3-4B 모델과 구분한다.
- 최종 36회에서 JSON 파싱 실패·출력 잘림은 0회, 자동 오류가 있는 호출은 17회였다. 문자열 근거 검사 통과가 관계의 의미적 정확성을 보장하지 않는다.
- 같은 사실도 별칭·표현·방향 때문에 다르게 집계된다. Novel-44557 E256/E512 관계 Jaccard는 문자열 비교 0.000에서 부분 정렬 후 **0.160**이 되었다.
- 작은 청크와 큰 청크에서만 나온 원문 지지 사례가 있지만, 잘못된 연결도 발견됐다. **다중 크기의 효과나 KET-RAG 대비 우위를 입증한 결과는 아니다.**

|문서|E128 entity / relation|E256 entity / relation|E512 entity / relation|
|---|---:|---:|---:|
|Novel-10146|62 / 38|58 / 42|61 / 50|
|Novel-44557|15 / 17|17 / 11|20 / 18|

seed 17, 문서·조건 내 중복을 제거한 문자열 집합 수다. relation은 ID 연결이 가능한 항목이며 유효 fact 수가 아니다. 두 표본은 도시·호텔 기록과 여행 서술이다.

## 결과 읽기

**E1024 추가:** 동일한 1,024토큰 구간 전체를 한 번에 추출하는 조건 2회를 추가했다. 기존 36회와 합쳐 38회이며, [128/256/512/1024 비교](experiments/pilot/E1024_비교.md)에서 확인할 수 있다. 원시 기록은 `e1024_raw`에 별도로 보존한다. 여행 문서 E1024는 9 entity / 7 relation이었다. 호텔 문서는 출력 4,096토큰에서 잘렸고, 별도 6,144토큰 상한 진단에서는 49 / 30으로 완료됐다. 이 진단은 동일 설정 비교에 합치지 않는다.

- [36회 원래 실험 결과](experiments/pilot/청크별_추출_실험결과.md)
- [별칭·관계 방향의 부분 정렬 후 비교](experiments/pilot/정렬후_비교.md)
- [이름 기반 endpoint 재실험](experiments/pilot/이름기반_재실험.md)
- [실험 설계와 후속 통제 조건](experiments/pilot/실험설계.md)
- [진단 사례 12개](experiments/pilot/diagnostic_examples.json): assistant가 원문과 대조한 의도적 사례 선택. 독립적인 사람 주석이나 precision 표본이 아니다.

`pilot_raw`는 최종 36회, `pilot_raw_initial`은 반복 생성 때문에 중단한 초기 실행이다. `name_endpoint_raw`와 `name_endpoint_schema_raw`는 이름 기반 추출 및 스키마 제약을 시험한 별도 진단 실험이다. 서로 섞어서 청크 크기 효과를 계산하지 않는다.

## 설치와 재현

Python 3.9 이상, Python 패키지 설치 불필요. 추출에는 `/tokenize`와 `/v1/chat/completions`를 제공하는 로컬 `llama-server` 및 해당 GGUF 모델이 필요하다. 모델 가중치·실행기 바이너리는 포함하지 않는다. 모델과 실행기 SHA256은 [environment.json](experiments/pilot/environment.json)에 기록했다. 해시·seed가 같아도 실행 환경이 달라지면 출력이 달라질 수 있다.

저장된 결과만 재계산할 때는 모델이 필요 없다.

```bash
python3 -m unittest discover -s tests -v
python3 experiments/pilot/compare_pilot.py
python3 experiments/pilot/align_relations.py
python3 experiments/pilot/check_name_diagnostic.py
python3 experiments/pilot/write_pilot_report.py
```

새 추출은 기존 결과와 다른 폴더에 저장한다.

```bash
python3 experiments/pilot/download_corpus.py --output data/novel.json
llama-server --model /path/to/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 11439 --ctx-size 8192 --gpu-layers 99 \
  --parallel 1 --alias qwen-extraction --no-webui
# 다른 터미널에서 실행
python3 experiments/pilot/run_chunk_pilot.py --corpus data/novel.json --out runs/pilot
python3 experiments/pilot/compare_pilot.py --raw runs/pilot --out runs/comparison.json
```

다운로더는 기록된 corpus SHA256과 다르면 중단한다. `main`의 upstream 데이터가 바뀌면 동일 버전의 corpus를 확보해야 한다. 원 실험의 선택 구간과 문자·토큰 offset은 manifest에 보존되어 있다. 새 실행 시 모델 tokenizer를 다시 사용하므로 같은 tokenizer가 필요하다.

## 코드가 하는 일

1. `download_corpus.py`: 공식 corpus 다운로드와 SHA256 검증.
2. `run_chunk_pilot.py`: 고정 seed로 문서·구간을 선택하고 모델 토큰 piece를 UTF-8 원문으로 검증한다. 동일 구간을 128/256/512토큰으로 나누고 각 청크의 offset·해시를 저장한다. Overlap은 0이다. 문장이나 고유명사를 자를 수 있다.
3. 각 청크를 같은 모델·프롬프트에 보내 entity/relation JSON을 추출한다. 원시 요청·응답·시간·오류를 파일별 보존한다. 입력/설정이 다른 체크포인트나 미완료 체크포인트가 있으면 조용히 합치지 않고 중단한다.
4. `compare_pilot.py`: 문자열 정규화와 집합 비교. 근거의 원문 포함 여부와 endpoint 존재 여부를 별도 기록한다.
5. `align_relations.py`: 두 구간을 확인해 작성한 명시적인 별칭/관계 규칙으로 부분 정렬하고 모든 원래 주장·근거를 보존한다. 일반화된 entity linker나 의미 판정기가 아니다. 잘못된 주장을 올바른 주장으로 고치지 않는다.
6. `extract_names.py`: 별도 v3/v4 진단. v3는 JSON object만 제한하고 v4는 필수 필드 스키마를 추가한다. 관계 양 끝을 entity 이름으로 추출하고 코드에서 ID를 할당한다. 알 수 없는 이름을 임의로 연결하지 않는다. **틀린 이름을 선택하는 의미 오류까지 막지는 못한다.**

이름 기반 스키마 진단(v4)을 재실행하려면 아래 명령을 사용한다. v3 재현에는 `--plain-json`을 추가한다. 두 진단 모두 출력 검증 실패가 발생해 개선된 추출기로 채택하지 않았다.

```bash
python3 experiments/pilot/extract_names.py --job Novel-10146_e256_s17_c00 --out runs/names_s17.json
python3 experiments/pilot/extract_names.py --job Novel-10146_e256_s29_c00 --out runs/names_s29.json
```

## 연구상 한계와 다음 실험

현재 비교에는 seed 변동, 잘린 고유명사, 잘못된 ID, 별칭·관계 표현 차이, 환각이 섞여 있다. 추가 관계 수만으로 상호보완성을 주장하면 안 된다. 다음 단계는 별도 문서에서 규칙을 고정한 정렬, 조건을 가린 사람 검토, 각 크기의 여러 seed, 문장 경계 대조군, 동일 추출 비용의 단일 크기 반복 대비 다중 크기 비교다. 이후 retrieval/QA와 비용을 함께 평가해야 한다. Re-DocRED의 gold relation 평가는 아직 실행하지 않았다.

교수님께는 “실제 추출에서 크기별 차이를 확인했고 일부 상호보완 사례가 있지만, 추출기 오류와 표현 차이를 분리하는 중”이라고 보고할 수 있다.

## 데이터 출처

[GraphRAG-Bench / GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark), `Datasets/Corpus/novel.json`. 전체 corpus는 재배포하지 않으며, 기록에는 실험에 사용한 두 구간·청크와 모델 응답이 포함되어 있다. upstream 저장소의 MIT 고지는 [UPSTREAM_LICENSE.txt](experiments/pilot/UPSTREAM_LICENSE.txt)에 보존했다. 원문·모델의 권리는 각 원저작자에게 있다. 사용자의 논문 미팅 PPT/PDF와 개인 로컬 경로는 포함하지 않았다.
