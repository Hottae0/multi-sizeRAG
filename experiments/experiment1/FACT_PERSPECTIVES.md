# Experiment 1: 새로운 사실과 다른 측면의 포착

2026-10-22. 새 측정 명령은 `python3 -m experiments.experiment1.perspectives`다.
기존 `study` / `analyze` / `evaluate` v1은 이전 실험 재현용으로 유지한다.
**새 사실과 관점 측정에는 아래 명령을 사용한다.** Python 3.10+ 표준 라이브러리만 필요하다.

## 질문과 평가 단위

질문은 **같은 원문의 청크 크기에 따라 새 주장, 기존 주장의 구체화, 다른 측면이 포착되는가?** 이다.
비용 효율성이나 QA 성능은 주평가지표가 아니다. 기록된 토큰·시간만 참고로 보존한다.

문서 전체를 하나의 성공/실패 단위로 두지 않고, manifest의 최대 크기·boundary=0 청크의 **실제 token/char 경계**를 기준으로 블록을 만든다.
명목상 1024토큰 블록 안에서 G128은 8개, G256은 4개, G512는 2개 청크를 합쳐 비교한다.
UTF-8 안전 경계 때문에 token 경계도 확인한다. 경계를 가로지르는 청크는 자르거나 중복 배정하지 않고 해당 조건/블록을 제외 사유와 함께 기록한다.
마지막 짧은 블록도 표시하여 유지한다. 원문·manifest·raw·기존 analyses는 수정하지 않는다.

기본 비교는 G256 seed17 기준의 G256 seed29와 G512 seed17이다. 기본 공통 성공 cohort는 이 **세 조건**이다.
G128/G1024의 실패 때문에 이 비교를 제외하지 않는다. 최대 크기 결정에는 전체 manifest의 청크 크기를 사용한다.
모든 계획 블록은 `comparisons.json`에 남으며, common_complete=false도 사라지지 않는다.
리뷰와 의미 추정은 세 조건의 공통 생성 성공 블록에서만 수행한다. 문서 전체의 추정치로 확대하지 않는다.
기존 자료에 적용한 결과는 사후 탐색이며 신규 평가 전에는 정책과 조건을 동결한다.

## 서버에서 기존 978회 결과 재분석

기존 서버 checkout은 실행 중일 수 있고 v1 study는 구현 hash를 고정한다. 아래처럼 별도 checkout을 사용한다.
원본 run 경로는 이전 실행 기록의 경로다. 다르게 보관했다면 `--run`만 실제 경로로 바꾼다.

```bash
cd /home/work/multi-sizeRAG
git clone --branch codex/fact-perspective-blocks --single-branch \
  https://github.com/Hottae0/multi-sizeRAG.git repo-fact-perspective-v1
cd repo-fact-perspective-v1
python3 -m unittest discover -s tests -v

python3 -m experiments.experiment1.perspectives analyze \
  --run /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1/documents/Novel-10146 \
  --out runs/Novel-10146-facts-strict-v1
```

이 명령은 모델 서버나 GPU 없이 실행된다. 재추출하지 않으며 리뷰 후보가 자동 정답으로 바뀌지 않는다.
같은 출력 폴더가 있으면 덮어쓰지 않고 중단한다. 다시 실행하려면 새 `--out`을 지정한다.

누락 qualifier 감도분석은 별도 폴더에서 실행한다.

```bash
python3 -m experiments.experiment1.perspectives analyze \
  --run /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1/documents/Novel-10146 \
  --out runs/Novel-10146-facts-unknown-v1 \
  --qualifier-policy missing-unknown
```

- strict: 원래 parser를 그대로 적용한다.
- missing-unknown: `time`/`condition` **누락만** 미기록 값으로 보존하여 다른 검사에 통과하면 후보에 포함한다.
- 미기록은 출력에서 null + missing_qualifiers로 표시하고, 명시적 빈 문자열과 절대 병합하지 않는다.
- invalid polarity, explicit null, 잘못된 endpoint/evidence는 자동 수리하지 않는다. JSON 파싱 실패와 length 응답도 복구하지 않는다.
- call_audit는 strict 원래 판정을 유지한다. 두 정책의 결과와 리뷰는 별도 snapshot이며 라벨을 자동 혼합하지 않는다.

5조건 모두 성공한 범위로 민감도를 확인하려면 다음을 추가한다.

```bash
--cohort-conditions e128_s17_b0 e256_s17_b0 e256_s29_b0 e512_s17_b0 e1024_s17_b0
```

G256 기준으로 다른 크기들을 모두 검토하려면 다음을 추가한다. 이때 기본 cohort도 네 added 조건과 baseline으로 확대된다.

```bash
--added e128_s17_b0 e256_s29_b0 e512_s17_b0 e1024_s17_b0
```

역방향은 `--baseline e512_s17_b0 --added e256_s17_b0`처럼 별도 출력 폴더로 실행한다.
선택한 조건은 모두 원래 manifest에 있어야 한다. 여러 비교를 직접 대조하려면 같은 cohort 설정을 사용한다.

## 검토 기준: 겹칠 수 있는 두 축 + 원문 지지

|필드|값|뜻|
|---|---|---|
|support|supported / unsupported / uncertain|원문이 해당 방향·술어·귀속·극성·조건을 지지하는가?|
|information|equivalent|기준 그래프에 이미 같은 의미가 있다. 표현 차이 포함.|
|information|refinement|기준 주장에 시점·조건·대상 등 추가 명세가 있다.|
|information|new_claim|해당 블록 기준 그래프에서 포착하지 못한 주장이다.|
|information|broader|기존 주장보다 포괄적이다. 자동 신규 사실로 세지 않는다.|
|information|uncertain|판단 보류. 미판정은 거짓으로 취급하지 않는다.|
|aspects|same|다른 측면을 포착했다고 볼 근거가 없다.|
|aspects|participant_role|관련 사건의 다른 참여자·역할을 포착한다.|
|aspects|action_state|관련 사건의 다른 행동·상태를 포착한다.|
|aspects|cause_purpose_result|관련 사건의 원인·목적·결과를 포착한다. 반드시 원문 지지 필요.|
|aspects|time_condition|관련 사건의 시간·조건 측면이 달라진다.|
|aspects|attribution|발화자·관점의 귀속 등을 추가로 포착한다.|
|aspects|decomposition_aggregation|같은 사건/내용의 관계를 분해하거나 통합한다.|
|aspects|other / uncertain|그 외(설명 필수) / 판단 보류.|

aspects는 복수 선택할 수 있으나 same/uncertain과 다른 값을 섞을 수 없다.
새 주장과 다른 측면은 겹칠 수 있으므로 두 수치를 합쳐 총 신규 사실 수로 만들지 않는다.
관련 없는 새 사실을 곧바로 다른 관점으로 표시하지 않는다. 다른 측면은 `baseline_matches`에 관련 기준 주장 ID를 연결해야 한다.
`equivalent`는 aspects=same이어야 하며, 단순 표현 차이를 관점 변화로 세지 않는다.

1. `review_blind.json`을 다른 이름으로 복사하여 **items의 라벨 필드만** 작성한다. version/input_hash/measurement_hash는 그대로 둔다.
2. 원문은 `review_blocks.json`, 기준 그래프의 전체 주장은 `review_baselines.json`에서 확인한다. 크기·seed·표집 층은 `review_key.json`에 분리되어 있다. 텍스트 자체로 조건을 추측할 가능성까지 제거한 완전한 맹검은 아니다.
3. support_spans에는 원문 절대 문자 [시작, 끝) 범위를 기입한다. evidence가 여러 위치에 나오면 임의 위치를 정답으로 가정하지 않는다.
4. baseline_matches는 같은 baseline_ref의 claim ID 목록이다. 여러 관계를 연결할 수 있어 분해/통합을 검토할 수 있다. equivalent/refinement/broader에는 필수다. 다른 측면을 주장할 때도 필수다.
5. supported 판정에는 information, aspects, support_spans, semantic_fact_id, annotator, rationale가 필요하다. semantic_fact_id는 검토자가 문서 내 의미상 같은 사실에 같은 ID를 부여한다. 원문 지원 여부를 ID의 문자열로 자동 추론하지 않는다.
6. unsupported는 annotator/rationale와 support 판정으로 완료할 수 있다. uncertain 또는 빈 필드는 미결정이다.
7. 전체 표집 항목 검토가 끝나야 후보 수의 모집단 추정치를 제공한다. 부분 검토는 관찰값만 기록한다.

```bash
cp runs/Novel-10146-facts-strict-v1/review_blind.json \
  runs/Novel-10146-facts-strict-labels.json
# labels 파일을 원문/기준 그래프와 함께 검토해 작성한 다음:
python3 -m experiments.experiment1.perspectives evaluate \
  --analysis runs/Novel-10146-facts-strict-v1 \
  --annotations runs/Novel-10146-facts-strict-labels.json \
  --out runs/Novel-10146-facts-strict-evaluation-v1.json
```

빈 리뷰 파일로 평가해도 실행은 되지만 의미 결과는 null이다. 자동 라벨링은 구현하지 않았다.
`analysis_integrity.json`은 분석 파일 변경을 감지한다. review_blind를 직접 수정하지 말고 복사본으로 판정한다.

## 집계의 의미와 한계

- 블록마다 literal_added_only / literal_shared 두 층에서 최대 `--per-stratum 20`개씩 표집한다. 공통 관계도 오병합 감사를 위해 검토한다.
- 같은 added 조건 안에서 문서별 후보 수를 추정하고 문서 동일 가중 평균을 낸다. 블록 독립 신뢰구간은 만들지 않는다.
- estimated_candidate_counts는 포함확률을 반영한 **지지된 후보 관계** 추정이다. 고유한 의미 사실 수가 아니다.
- 같은 사실이 여러 literal 후보나 블록에 반복될 수 있다. observed_reviewed_fact_classes는 검토된 표본에서 확인된 class 수일 뿐 전체 수 추정이 아니다.
- 전수 검토(census)와 문서 내 모든 사실 class의 중복 검토 후 `document_deduplication.complete=true`, reviewer/rationale를 작성해야 unique_supported_facts를 낸다. 필요하면 충분히 큰 per-stratum으로 새 분석을 만들어 전수 queue를 생성한다.
- `unique_fact_classes_with_block_local_new_label`은 **블록 기준 신규** 라벨을 가진 class 수다. 다른 블록 기준 그래프에 이미 있는 사실일 수 있어 문서 전체 신규성으로 주장하지 않는다.
- 같은 정보 축·서로 다른 측면의 수를 별도로 제시한다. 새 사실이 많은가와 같은 사건을 다르게 표현하는가는 독립적으로 해석한다.
- 공통 성공 블록 0개면 결과는 unavailable/null이다. 정상 생성된 빈 그래프는 후보 0개로 구분한다.
- 이 실험은 추출의 정보 차이를 설명한다. GraphRAG QA 성능, 전체 문서 recall, 비용 효율성, 크기의 인과 효과를 주장하지 않는다.

## 산출물

|파일|내용|
|---|---|
|report.md / summary.json|블록 수, 후보 수, 검토 대기 상태, 정책·hash, 참고 비용|
|blocks.json|계획 블록 전체와 조건별 경계 일치·실패·공통 성공 상태|
|comparisons.json|실패/비교 제외 블록도 포함한 literal 진단|
|review_blind.json|두 축 + 원문 지지 판정 템플릿|
|review_baselines.json / review_blocks.json|판정용 기준 주장과 원문|
|review_key.json|조건·층·포함확률·occurrence 역참조|
|occurrences.json / call_audit.json|후보·원본 감사 기록|
|analysis_integrity.json|분석 파일 변경 검증용 SHA-256|
