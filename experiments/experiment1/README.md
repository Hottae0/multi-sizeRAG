# Experiment 1 — Multi-granularity Graph Complementarity

## 현재 설계: 5조건·2가지 분석·1→5→20문서

|구분|실행 내용|
|---|---|
|추출 조건 5개|128/256/512/1024 seed17과 256 seed29를 사용합니다. Boundary=0입니다.|
|핵심 분석 ①|4크기의 6쌍·12방향에서 신규 관계·근거·구조를 비교합니다.|
|핵심 분석 ②|256+256과 256+512의 유효 신규 관계와 추가 비용을 비교합니다.|
|규모 확장|전체 원문 1문서 → 총 5문서 → 총 20문서로 확장합니다.|
|문서별 흐름|추출·오류 점검·literal 그래프 저장 후 다음 문서로 진행하며, 의미 정렬·semantic 그래프·원문 판정은 별도 수행합니다.|
|실행 환경|CPU 4코어·RAM 128GB·A100 80GB 1개·저장공간 500GB의 CUDA 서버를 기준으로 합니다.|

**새 실행은 [FULL_DOCUMENTS.md](FULL_DOCUMENTS.md)를 따라 주세요.**
문서당 5조건이므로 최종 20문서에서 기본 조건별 문서 그래프는 100개입니다.
동일 조건의 literal/semantic 표현은 별도 추출 조건으로 세지 않습니다.
실제 A100 추론 및 실제 의미 정렬·원문 판정은 아직 수행하지 않았습니다.

## 기존 region CLI와 과거 실행 재현

아래 명령은 과거 region manifest 재분석/호환용입니다.
기존 prepare의 seeds/boundaries는 Cartesian 조합을 만들므로 새 study의 명시적 5조건과 다릅니다.
과거 5문서·15구간 입력을 전체 원문 입력으로 간주하지 않습니다.

2026-10-19. Python 3.10+ 표준 라이브러리만 필요하다. 저장소 루트에서 실행한다.
기존 `pilot/`, `expanded/` 원시 결과를 수정하지 않는다. `main_novel/` 등 로컬 미커밋 작업도 포함하거나 덮어쓰지 않았다.
연구 흐름과 해석 범위는 [연구 설계](../../../docs/Experiment1_연구설계.md), 실제 점검 결과는 [검증 기록](results/검증기록.md)에 있다.

## 바로 실행: 기존 1,140개 출력 분석

```bash
python3 -m unittest discover -s tests -v
python3 -m experiments.experiment1 plan --run experiments/expanded/results
python3 -m experiments.experiment1 analyze \
  --run experiments/expanded/results --out runs/exp1_legacy --per-stratum 3
```

모델 호출 없이 기존 `manifest.json`과 `raw/*.json`을 읽는다. 새 분석 디렉터리가 필요하다.
원본 응답의 content를 다시 파싱하므로 저장된 `parsed`/기존 오류 판정에 의존하지 않는다.
출력 예산을 바꾼 `retry_results`를 자동 혼합하지 않는다. 이는 새 추출 실험이 아니라 **선행 결과의 사후 재분석**이다.

## 완전 오프라인 의미 정렬 데모

```bash
python3 -m experiments.experiment1.demo --out runs/exp1_demo
```

20개 모의 호출 → literal 분석 → 별칭·역관계 정렬 → semantic 분석 → 독립 그래프 → 평가표까지 실행한다.
**문자 단위 가짜 tokenizer와 작성된 응답·정렬 정답을 쓰는 소프트웨어 fixture**이며 모델 품질 실험이 아니다.
E128→E256 관계 Jaccard는 literal 0, semantic 1/3, 정렬 후 신규 edge 1개다.

## 실제 모델 추출

같은 모델의 `/tokenize`(token pieces 반환)와 `/v1/chat/completions`를 제공하는 llama.cpp 호환 서버가 필요하다.
이 패키지는 서버를 설치하거나 시작하지 않는다. 기존 서버 실행 절차는 `experiments/expanded/README.md`를 참고한다.
서버 root URL에는 `/v1`을 붙이지 않는다. 토크나이저와 추출 endpoint는 같은 서버다.
서버의 실제 모델 파일·tokenizer·backend 해시와 context 설정을 확인하고 아래 식별자에 기록한다.
`server-id`는 선언된 식별자이며 서버 상태를 원격으로 증명하지 않는다. 서버 설정을 바꾸면 새 run이다.

```bash
python3 -m experiments.experiment1 prepare \
  --regions experiments/expanded/results/manifest.json \
  --out runs/exp1_new \
  --url http://127.0.0.1:11439 \
  --model qwen-extraction \
  --model-revision 'REPLACE_WITH_MODEL_FILE_SHA256' \
  --tokenizer-id 'REPLACE_WITH_TOKENIZER_HASH' \
  --server-id 'REPLACE_WITH_BACKEND_HASH_AND_CONTEXT_CONFIG' \
  --max-tokens 4096 --temperature 0.2
python3 -m experiments.experiment1 plan --run runs/exp1_new
python3 -m experiments.experiment1 extract --run runs/exp1_new --limit 4
python3 -m experiments.experiment1 extract --run runs/exp1_new
python3 -m experiments.experiment1 analyze --run runs/exp1_new --out runs/exp1_new_analysis
```

`prepare`는 원본 5문서·15구간의 **동일 문자 범위**를 재사용한다. 새 tokenizer의 pieces를 byte-for-byte 검증하고 tokenization 해시를 저장한다.
모델을 바꾸면 구간의 토큰 수가 4096과 달라질 수 있다. `source_tokens`와 새 `tokens`를 구분한다.
128/256/512/1024는 같은 token stream의 목표 폭이다. UTF-8 안전 경계까지 이동하므로 작은 초과가 가능하며 실제 token_start/end를 기록한다.
마지막 짧은 청크를 포함하고 모든 조건이 동일 원문을 누락·중복 없이 덮는지 검사한다.

- Gleaning 0, chunk당 한 번, 자동 재시도 0. 실패도 한 번의 시도로 보존한다.
- 요청 전 started 기록, 수신 즉시 wire response 저장, 그 뒤 JSON 파싱. HTTP 오류 본문도 보존한다.
- 재개 시 성공·실패·결과 불명 started 모두 재요청하지 않는다. 설정이나 raw job이 다르면 거부한다.
- `.extract.lock`은 동시 실행을 막는다. 강제 종료 후 실제 프로세스가 없음을 확인하고 stale lock만 삭제한다. raw 기록은 삭제하지 않는다.
- 미완료 요청을 다시 추출하려면 별도 run으로 수행하고 원본·추가 비용을 함께 보고한다. 최초 실행의 single-pass 결과에 대체하지 않는다.
- API key는 필요하면 `EXPERIMENT1_API_KEY` 환경변수로 제공한다. manifest/raw에는 저장하지 않는다.
- 실제 새 Qwen 추론은 이번 코드 검증에서 실행하지 않았다. 출력 예산·context·반복 억제 설정은 개발군에서 검증한 뒤 고정해야 한다.

## 본실험 seed/boundary 대조군

`prepare`에 `--seeds 17 29 --boundaries 0 64`를 지정하면 4크기 × 2seed × 2경계 조건을 만든다.
경계 offset=64는 첫 청크 길이가 64이고 이후 목표 폭으로 분할하며 마지막 꼬리까지 유지한다.
같은 seed/boundary끼리 크기를 비교하고, 같은 크기에서 seed만 또는 boundary만 바꾼 비교를 별도로 해석한다.
동시에 두 요소가 달라지는 쌍은 순수 요인 대조가 아니다. 모든 쌍을 내보내므로 사전 지정된 비교를 주 분석으로 선택한다.
최초 파이프라인은 기본 seed 17/boundary 0만으로 가능하지만 **granularity 효과 본실험에는 같은 크기 대조군이 필요**하다.
추가 호출 수·출력량이 다르므로 이 설정만으로 동일 비용 비교가 되지는 않는다. temperature=0에서 seed 차이가 의미 없을 수 있다.

## 의미 정렬: 추출 뒤, 그래프 비교 전

자동 fuzzy/embedding 유사도를 동치라고 확정하지 않는다. 실제 semantic 경로는 **검토된 occurrence class**를 적용한다.
동명이인은 동일 표면명이어도 별도 class, 별칭은 같은 class, 역관계는 명시적 방향 전환으로 표현한다.
정렬되지 않은 occurrence는 singleton이며 임의 이름 병합하지 않는다. 판정 누락이 있으면 semantic 결과는 `partial_do_not_claim_final`이다.
판정 파일이 없으면 semantic Jaccard는 unavailable이며 literal 결과를 semantic으로 표시하지 않는다.

1. `analyze`의 `occurrences.json`, `regions.json`과 원문 chunk를 읽는다. `alignment_template.json`을 복사한다.
2. entity_classes의 모든 member는 entity occurrence ID다. 문서 내부 동일 개체만 묶고, singleton도 검토 기록을 남긴다.
3. relation_classes는 같은 방향의 **상호 함의**가 확인된 사실만 묶는다. `entails`, `related`, `contradictory`, `uncertain`은 equivalent로 넣지 않는다.
4. 각 relation member에 `label: "equivalent"`, 역관계만 `reverse: true`를 넣는다. 극성·시간·조건이 다르면 병합을 거부한다.
5. 모든 판정에는 reviewer와 reason을 남긴다. raw/manifest가 바뀌면 input_hash가 달라져 재사용을 거부한다.

형식 예시(실제 ID는 생성 파일에서 복사):

```json
{
  "input_hash": "FROM_ALIGNMENT_TEMPLATE",
  "entity_classes": [
    {"id":"doc1:alice","members":["JOB:n:0","OTHER:n:2"],"reviewer":"reviewer1","reason":"원문이 별칭 동일성을 명시"},
    {"id":"doc1:bob","members":["JOB:n:1","OTHER:n:0"],"reviewer":"reviewer1","reason":"동일 인물"}
  ],
  "relation_classes": [
    {"id":"doc1:knows","reviewer":"reviewer1","reason":"방향을 정렬하면 같은 사실",
     "canonical":{"source":"doc1:alice","predicate":"knows","target":"doc1:bob","polarity":"positive","time":"","condition":""},
     "members":[{"id":"JOB:r:0","label":"equivalent"}, {"id":"OTHER:r:0","label":"equivalent","reverse":true}]}
  ]
}
```

```bash
python3 -m experiments.experiment1 analyze \
  --run runs/exp1_new --out runs/exp1_aligned --alignment reviewed_alignment.json
```

`--alignment-pairs`는 동일 region의 모든 관계 쌍에서 공통 endpoint 이름 또는 동일 predicate를 가진 쌍을 검토 queue로 출력한다.
차단된 나머지 쌍에서 seed 17 reservoir 표본 100개와 포함확률을 내보내 후보 누락을 감사한다.
이 queue는 의미 판정기가 아니다. pair 판정은 전체 class로 정리해 적용한다. 단순 연결요소식 전이 병합을 하지 않는다.
별칭 후보가 lexical blocking에서 누락될 수 있으므로 원문·entity 전체 목록과 제외 표본을 함께 검토한다.
시간 복잡도는 region당 O(관계 수²): 큰 corpus에는 후보 생성 방식을 추가해야 한다.
기존 응답의 극성/시점/조건은 `unspecified`이며 predicate 원문을 유지한다. 미기록 qualifier를 positive로 추정하지 않는다.

## 출력과 평가

|파일|내용|
|---|---|
|`call_audit.json`|호출별 완료·빈 출력·잘림·schema/endpoint/evidence 오류, usage·시간|
|`occurrences.json`, `regions.json`|원문, 모든 유지 mention/fact, job·chunk·크기·seed·raw hash·문자 offset|
|`graphs/*.json`|각 region/조건별 방향성 multigraph, literal/semantic 분리, node mention·edge occurrence 역참조|
|`comparisons.json`|동일 공통 구간의 A→B 방향별 node/edge Jaccard·신규량·구조·evidence 지표|
|`document_macro.json`|문서 내 region 평균 후 문서 동일 가중 평균; 단순 region 독립 추론 방지|
|`review_blind.json`|크기/seed/신규·공통 stratum을 가린 원문 판정표|
|`review_key.json`|별도 분석자용 condition/stratum/모집단/표본크기/포함확률|
|`summary.json`|재현 hash, 공통/제외 구간, 비용·오류·정렬 상태; 미판정 valid count는 null|

raw 원문에 없거나 endpoint가 잘못된 항목은 격리한다. 정상 생성된 응답의 일부 항목 오류는 call_audit에 남기고 나머지 후보를 사용한다.
따라서 `complete_generation`은 오류 없는 정답 그래프라는 뜻이 아니다. 파싱 실패·잘림은 응답 전체를 제외한다.
반복 relation occurrence는 보존하고 같은 edge로 중복 제거하며 개수를 보고한다.
같은 evidence 문자열이 chunk에 여러 번 등장하면 가능한 모든 위치를 provenance에 보존하지만,
그중 하나를 임의 선택하거나 모두 독립 근거로 세지 않는다. graph의 ambiguous 목록에 넣고 evidence 지표에서는 제외한다.

원문 평가는 `regions.json`과 함께 `review_blind.json`을 사용한다. `supported`, `semantically_novel`을 true/false로,
`annotator`, `rationale`, `error_type`을 작성한다. baseline_claims 전체와 원문을 비교해야 의미 신규성을 판정할 수 있다.
원문 substring 일치는 entailment 판정이 아니다. source/target·방향·부정·시점·발화자 오류를 구분한다.
공통 stratum도 표집하므로 잘못된 정렬로 숨겨진 신규 사실을 감사할 수 있다. 다른 annotator의 독립 판정 후 합의된 최종 파일 하나를 입력한다.

```bash
python3 -m experiments.experiment1 evaluate \
  --analysis runs/exp1_aligned --annotations reviewed_labels.json --out runs/exp1_evaluation.json
```

모든 표집 항목의 판정이 완료된 비교만 HT 추정 유효 신규 수/비율과 유효 신규 edge당 비용을 낸다.
일부 판정만 있으면 관찰된 유효 수만 보고 추정치는 null이다. raw string 신규가 아니라 지지+의미 신규 동시 true가 필요하다.
표본 추정은 기술 통계이며 gold recall·유의성 검정이 아니다. 비용은 제외 구간·실패를 포함한 추가 조건 전체 비용이다.
`analyze --rates INPUT_PER_M OUTPUT_PER_M`을 명시한 경우에만 사용자가 정한 화폐 비용을 계산한다.
usage 누락이 있으면 금액은 null이며 알려진 토큰 합계는 하한이다. 로컬 모델에 API 단가를 임의 적용하지 않는다.

그래프 정렬은 비교 목적의 공통 identity space다. 각 그래프에 다른 크기의 fact나 evidence를 추가하지 않는다.
다만 cross-size 판정에서 다른 크기의 문맥을 봤으므로 이 그래프를 후속 QA의 정보 누출 없는 단일 크기 기준선이라고 부르면 안 된다.
후속 QA에는 개발군에서 동결된 독립 정렬 정책으로 별도 구축하는 대조가 필요하다.
