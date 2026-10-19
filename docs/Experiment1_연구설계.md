# Multi-size GraphRAG 연구 흐름 1–6 및 Experiment 1

2026-10-19. 목표는 **추출 granularity의 상호보완성을 활용한 새로운 GraphRAG Graph Fusion 방법론**이다.
차이가 존재한다는 것, 유효한 지식이 추가된다는 것, 검색·QA가 개선된다는 것을 차례로 검증한다.
현재 신규 fusion 규칙이나 QA 개선을 입증한 단계는 아니다.

## 1. 연구 질문과 고정 corpus/region 설계

질문: 같은 크기의 독립 추출을 추가하는 것보다 다른 크기의 그래프를 추가할 때 유효 신규 관계와 근거 경로가 더 늘어나는가?
선행 corpus의 고정 5문서 × 3구간 × 4096 원래 모델 토큰을 재사용한다. 모든 크기에서 동일 문자 범위를 처리한다.
새 모델 tokenizer를 사용할 경우 토큰 수 변화와 모델·정밀도·backend를 별도 run으로 기록한다.
기본 크기는 128/256/512/1024, overlap=0, 마지막 꼬리 포함이다. 문서·구간·chunk 문자/토큰 offset과 hash를 보존한다.
개발 구간에서 prompt·출력 예산·정렬 정책을 정한 뒤 미관찰 평가 문서에 적용한다. 기존 5문서는 이미 분석한 개발/탐색군이다.

## 2. Single-pass extraction과 오류·비용 보존

모든 chunk에서 동일 prompt/model/decoding/output cap, gleaning=0, 한 번의 요청만 실행한다.
raw request/response, 종료 사유, usage, 시간, 원문과 hash를 보존한다. 출력을 고치거나 잘린 JSON을 억지 복구하지 않는다.
실패는 빈 graph가 아니다. 정상 JSON의 빈 arrays, schema 오류, endpoint 오류, quote 불일치, 반복 생성, 잘림을 구분한다.
원문 인용이 존재해도 관계가 틀릴 수 있다. 자동 검사 통과와 의미적 유효성을 분리한다.

기존 Qwen3-VL-4B-Instruct Q4_K_M의 1,140회에는 잘림 15건과 자동 오류 호출 475건이 있다.
후속 15건 재추출은 출력 상한/컨텍스트가 달라 원본과 자동 혼합하지 않는다.
43,120 출력 토큰/596 관계 항목/고유 ID triple 18종의 반복 사례는 출력 상한 확대만으로 해결되지 않음을 보여준다.
새 프롬프트·모델은 별도 프로토콜이며 이전 결과와 통제된 직접 비교라고 부르지 않는다.

## 3. Extraction 이후 literal/semantic 정렬과 중복 제거

literal 집합은 문서 내 정규화 entity 이름+type과 `(source,predicate,target,polarity,time,condition)`으로 만든다.
먼저 literal Jaccard를 계산하고, 원문 기반 entity identity와 사실 동치 판정을 적용한 semantic Jaccard를 따로 계산한다.
J(A,B)=|A∩B|/|A∪B|이며 둘 다 비면 null이다. 관계 방향을 포함하고 역관계는 명시적 변환만 허용한다.
별칭·동명이인·부정·시점·조건·발화자를 확인한다. 동치와 한쪽 함의·관련성·모순·불확실을 구분한다.
현재 구현은 검토된 occurrence class 방식이다. 임베딩 유사도나 LLM 판정을 자동 정답으로 사용하는 모듈은 없다.
판정 파일이 없으면 semantic 점수를 생성하지 않는다. 불완전 판정 점수는 탐색용으로 표시한다.
정렬은 유효성 검사와 다르다. 서로 동치인 두 추출이 모두 틀릴 수도 있다.

## 4. Provenance를 보존한 독립 방향성 E–R–E 그래프

각 조건의 entity가 node, 방향성 relation이 edge인 multigraph를 만든다. 같은 두 node 사이 여러 predicate를 유지한다.
중복 edge의 모든 occurrence와 quote 위치, 문서·구간·chunk·크기·seed·boundary·raw hash를 보존한다.
다른 크기의 fact/evidence를 단일 graph에 추가하지 않는다. 분석 단위는 region이며 문서 간 동일 이름을 합치지 않는다.
의미 비교용 공통 identity space는 cross-size 정보를 사용하므로 후속 QA용 단일 크기 구축 정책과 구분한다.

## 5. Graph complementarity와 유효 신규 관계 평가 — Experiment 1

모든 계획 조건에서 생성이 완료된 동일 region 교집합을 주 비교 cohort로 사용한다. 제외 목록·실패율·비용도 함께 보고한다.
item 오류는 격리하고 남은 후보 graph를 분석한다. 교집합 자체가 쉬운 구간으로 편향될 수 있어 전체 corpus 성능을 대표하지 않는다.
모든 방향 A→B를 보고하되 핵심 대조는 사전에 고정한다. 문서 내 region 평균 후 문서 macro 평균을 제시한다.

|축|계산/검증|해석 제한|
|---|---|---|
|Node|Jaccard, B−A 신규 node|동명이인/별칭 정렬 민감|
|Edge|Jaccard, B−A, 신규/추가 graph edge 수, union 크기|관계 수는 유효 사실 수가 아님|
|Structural|weak components·고립 node, A의 두 성분을 잇는 B의 edge, union에만 있는 simple directed 2-hop node path|그래프 연결 가능성 proxy; 정답 추론 경로 아님|
|Evidence|공통 edge에서 새 exact span 수와 A에 덮이지 않은 B의 문자 수|인용 길이 차이·위치 모호성, 의미 신규 근거와 구분|
|Validity|원문 지지 AND baseline 대비 의미 신규, 오류 유형|정렬과 별도 독립 주석 필요|
|Cost/error|모든 호출/토큰/시간, missing usage, 잘림/빈 출력/반복/검증 오류|로컬 단가 미지정 시 화폐 비용 미보고|

평가 데이터는 `(region, A→B, novel/shared)` 층별 무작위 표본이며 seed와 포함확률을 저장한다.
annotator는 condition/stratum을 가린 claim·baseline 전체 claim과 원문을 본다. 분석자는 별도 key를 보관한다.
새 후보 중 지지되는 관계의 비율과 HT 유효 신규 개수 추정, 추가 indexing cost/유효 신규 edge를 보고한다.
표본 미판정은 false로 채우지 않고 null로 둔다. 독립 두 판정자와 합의 절차를 권장하며 현재 코드가 판정자를 대신하지 않는다.
정답 관계·경로 annotation이 없으므로 relation recall, Novel Evidence Path Coverage/Recall은 계산하지 않는다.
최대 2-hop 구조 proxy를 path recall이라고 이름 붙이지 않는다. 3-hop 이상·QA별 경로 검증은 후속 구현이다.

**반복 대조군:** 첫 파이프라인 실행에는 필수 아님. 그러나 본실험에서는 같은 크기의 seed 반복과 경계 이동을 포함한다.
`--seeds 17 29 --boundaries 0 64`로 실행 가능하며 각 chunk는 여전히 single-pass다.
기본 4크기 15×4096이면 크기별 480/240/120/60, 합계 900회다. E256 seed29만 추가한 선행 설계가 1140회다.
전체 2seed×2boundary 설계는 경계 꼬리 때문에 단순히 900×4와 같지 않을 수 있다. `plan`으로 실제 수를 계산한다.
동일 비용 분석은 별도로 호출/토큰/시간 budget을 맞추고 반복에서 추가되는 유효 지식과 비교해야 한다.
2seed만으로 안정적 통계 추론을 보장하지 않으며 본실험 반복 수는 개발군 변동·예산으로 사전 결정한다.

## 6. 상호보완성 기반 Fusion과 GraphRAG 후속 평가

Experiment 1에서 유효한 차이를 확인한 다음 fusion 정책을 설계한다. 다음은 비교할 후보이며 이번 구현의 성능 결과가 아니다.

- Naive union: 동일 정렬 후 단순 합집합 기준선.
- Evidence-aware fusion: 근거 지지·방향·극성·시점을 고려한 통합.
- Complementarity-aware fusion: 고유하지만 유효한 관계와 연결 근거를 보존하는 제안 방법.

cross-size 반복 발견만 가중하면 특정 크기에만 있는 유효 관계를 잃을 수 있으므로 데이터로 설계를 결정한다.
단일 크기, 동일 크기 반복 union, multi-size union, 제안 fusion, Text-RAG를 동일 검색/생성 모델·문맥 budget으로 비교한다.
정답 annotation이 있는 evidence/path recall, retrieval 품질, QA F1/사실 충실도, 비용을 문서 단위로 평가한다.
전체 크기를 모두 추출한 fusion은 비용 절감을 자동으로 뜻하지 않는다. selective indexing/refinement는 효용 확인 후 다음 실험이다.

## 구현 완료와 아직 필요한 연구 작업

완료: 별도 표준 라이브러리 파이프라인, 고정 region 청킹/단일 추출/재개, raw 보존, 검토 기반 semantic 정렬,
독립 graph/provenance, 공통 cohort 지표, blind 평가표와 판정 집계, 원본 1140개 재분석, 모의 의미 정렬 통합 테스트.
미수행: 새 모델의 실추론, 실제 corpus 전체의 의미 정렬·독립 원문 판정, 동일 비용 반복 본실험, 새로운 fusion 및 QA 검증.
실행 명령과 파일 스키마는 [Experiment 1 README](../experiments/experiment1/README.md)를 따른다.
