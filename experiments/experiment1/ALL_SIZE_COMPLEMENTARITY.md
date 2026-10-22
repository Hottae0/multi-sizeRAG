# 전체 크기 결합 분석

2026-10-23. `256 + 512`만 비교하지 않고, 128·256·512·1024의 전체 합집합과 각 크기의 기여를 분석한다. Python 3.10+ 표준 라이브러리만 사용한다. 모델 서버, GPU, API 호출, 재추출은 필요 없다. 기존 manifest/raw와 frozen study 계약을 변경하지 않는다.

## 완료된 문서들을 한 번에 분석

저장소 루트에서 실행한다. `--study`는 `documents/<문서ID>/manifest.json`이 있는 study 폴더다. TAR를 `restore`로 복원한 경우 그 구조가 있는 `source`를 지정한다.

```bash
python3 -m experiments.experiment1.complementarity batch \
  --study /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1 \
  --out runs/all_sizes_v1 \
  --docs Novel-10146 Novel-30752
```

`--docs`를 생략하면 study의 documents 아래 모든 문서를 분석한다. 아직 호출이 끝나지 않은 문서도 읽을 수 있지만, 실패/미기록 구간은 공통 성공 범위에서 제외되므로 완료 후 새 output으로 다시 분석한다. output은 새 폴더여야 한다. 재실행 시 `runs/all_sizes_v2`처럼 새 이름을 쓴다. 원본 문서 폴더 내부에는 결과를 쓰지 못하도록 제한한다.

각 문서를 `strict`와 `missing-unknown` 정책으로 모두 분석한다. 두 정책의 개수를 더하지 않는다. 후자는 누락된 time/condition을 unknown으로 보존하며, 잘린 JSON·없는 endpoint·잘못된 evidence를 복구해 주지 않는다.

한 문서만 분석하거나 조건을 지정할 수도 있다.

```bash
python3 -m experiments.experiment1.complementarity analyze \
  --run /path/to/study/documents/Novel-30752 \
  --out runs/novel30752_all_sizes \
  --sizes 128 256 512 1024 --seed 17 \
  --qualifier-policy missing-unknown
```

요청한 크기가 manifest에 없으면 명시적으로 실패한다. G256의 다른 seed가 존재하면 반복 대조로 별도 포함한다. 네 크기의 seed17 전체 조합과 반복을 섞어서 같은 실험으로 취급하지 않는다.

## 자동 결과

각 문서/정책 폴더:

- `summary.json`: 모든 비어 있지 않은 크기 조합(4크기이면 15조합), 전체 합집합, 크기를 하나씩 뺄 때 사라지는 수, 최고 단일 크기 대비 증가, 블록별 최고 단일 크기를 선택한 합계, 블록 하나씩 제외한 민감도, G256 seed 반복 대조.
- `report.md`: 자동 집계의 간단한 안내.
- `call_audit.json`: 완료 상태, 파싱/항목 오류 및 원시 응답 출처.
- `snapshot.json`: manifest/raw/code 식별 해시, 원문 블록, 조건별 관계와 occurrence ID.
- `evidence.html`: 관계·조건·근거·원문을 펼쳐 보는 로컬 검토표.
- `review_template.json`: **공통 성공 구간**의 판정 양식. 처음에는 모두 pending.
- batch 루트의 `index.json`: 문서별 분석 경로. 문서들을 한 큰 그래프나 독립 블록 표본으로 합산하지 않는다.

세 범위를 구분한다:

1. `common_cohort`: 모든 크기와 seed 반복이 성공한 동일 최대 청크 블록. 반복과 조합을 비교할 주 분석 범위.
2. `common_sizes`: 선택한 크기들만 모두 성공한 블록. 반복의 실패 때문에 제외된 블록도 포함할 수 있음.
3. `whole_available`: 기록에서 사용 가능한 전체 블록. 실패/미정렬 구간이 섞인 기술 통계이며 우열 판정의 주 근거가 아님.

자동 literal은 타입·극성·time/condition·unknown을 구분한다. name_triple은 정규화한 주어/관계/목적어만 비교하는 진단이다. 둘 다 **의미적 동일성이나 사실의 정확성 판정이 아니다.** 집계 단위는 `(블록, 관계)`이며 문서 전체에서 반복되는 사실을 완전히 중복 제거한 수가 아니다. 공통 범위가 비어 있으면 “신규 사실이 0개”라고 해석하지 않는다.

## 검토 후 핵심 관계의 상호보완성 계산

템플릿을 복사하여 판정한다. 기존 snapshot은 편집하지 않는다.

```bash
cp runs/all_sizes_v1/Novel-30752/strict/review_template.json runs/novel30752_reviews.json
```

각 항목의 `id`는 변경하지 않는다. 입력 해시가 다른 결과에 판정을 재사용할 수 없다. 동일 literal이라도 조건별 evidence가 다를 수 있어 **각 조건을 따로 검토**한다.

판정 예시(실제 근거를 직접 읽고 작성):

```json
{
  "id": "템플릿에 있는 ID",
  "verdict": "supported",
  "family_id": "shelly-pioneer-employment",
  "canonical_claim": "Capt. Shelly worked at the Pioneer",
  "support_quotes": ["Capt. Shelly's first engagement was in the office of the Register, but he soon changed to the Pioneer"],
  "reviewer": "실제 판정자",
  "reviewer_kind": "human",
  "rationale": "주체와 근무처가 원문에서 직접 지지된다.",
  "alignment_rationale": "같은 블록에서 worked at과 was employed by를 동일한 핵심 근무 관계로 판정했다."
}
```

- verdict: `pending`, `supported`, `unsupported`, `unresolved`.
- `supported`는 **핵심 관계**의 원문 지지다. 독립 역사적 진실, 모든 qualifier의 정확성, 신규 사실 여부를 인증하지 않는다.
- supported에는 family_id, canonical_claim, 실제 원문에 존재하는 support_quotes, reviewer/reviewer_kind, rationale, alignment_rationale가 모두 필요하다.
- 같은 블록의 동등한 관계에는 같은 family_id와 canonical_claim을 쓴다. 다른 극성의 관계는 같은 family에 넣을 수 없다. 화자·사건·대상·시간이 다르면 무리하게 합치지 않는다. unknown은 명시적으로 빈 값과 다르다.
- 가족 ID는 블록 내부에서만 유효하다. 다른 문서·블록의 동명이인이나 반복을 자동 병합하지 않는다.
- 모델이 판정했으면 reviewer_kind는 `model`로 기록한다. 이 도구가 모델을 호출하지는 않는다.
- 부분 검토 및 unresolved는 전체 모집단의 확정 지표로 승격하지 않는다.

```bash
python3 -m experiments.experiment1.complementarity evaluate \
  --analysis runs/all_sizes_v1/Novel-30752/strict \
  --annotations runs/novel30752_reviews.json \
  --out runs/novel30752_reviewed_v1
```

출력의 `reviewed_subset_metrics`는 검토된 supported 핵심 관계의 집계다. `complete_review_metrics`는 모든 공통 구간 후보가 supported/unsupported로 판정된 경우만 채워진다. `families.json`에는 검토자가 동등하다고 묶은 관계와 제공 조건을 저장한다. 사람 gold 정답이 없는 상태를 감추지 않도록 reviewer_kinds도 기록한다.

## 이 도구가 판정하지 않는 것

- 합집합 증가를 새로운 사실·관점·QA 성능으로 자동 판정하지 않는다.
- 서로 다른 관계를 이어 붙여 원문에 없는 인과관계를 만들지 않는다.
- 신규 사실 / 속성 세분화 / 관점 / 추출 교정 / 필터 손실 회복의 상대적 판정은 별도 작업이다. 기존 `judgments`의 쌍별 평가와 이 도구의 전체 union 평가를 대체 관계로 혼동하지 않는다.
- 같은 문서의 블록을 독립 문서처럼 취급하여 유의성 검정을 하지 않는다.
- 이전의 문서별 수동 판정 숫자를 새 문서의 정답으로 하드코딩하지 않는다. 과거 core-only 스크리닝을 자동으로 완전 검증된 사실로 가져오지도 않는다.

원본 추출 조건은 유지하면서 문서를 추가할 수 있다. 먼저 자동 지표로 점검하고, 근거를 판정한 뒤 전체 조합 및 크기별 고유 기여를 해석한다.
