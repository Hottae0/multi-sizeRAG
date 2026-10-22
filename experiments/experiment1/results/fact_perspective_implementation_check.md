# 사실·측면 블록 측정 구현 검증

2026-10-22. 기준 commit `49d4c99`에서 별도 successor 모듈을 추가했다.
서버 실행 당시 frozen study를 보존하기 위해 core.py/analysis.py/study.py/cuda_server.py는 변경하지 않았다.

## 의도와 구현 대응

|사용자 의도|구현 및 확인|
|---|---|
|최대 chunk_size 단위 비교|manifest의 최대 size, boundary=0 실제 token/char 구간 사용. 작은 청크는 전체 포함·연속 coverage일 때만 합침. 교차 청크를 자르지 않음.|
|새 사실과 다른 관점을 구분|support + information + aspects 세 필드. new_claim과 different_perspective는 겹칠 수 있음. equivalent는 same만 허용.|
|표현 차이를 신규 정보로 세지 않기|literal 후보와 사람의 의미 판정 분리. 미판정 결과는 null.|
|구체화·분해·통합 평가|refinement/broader, decomposition_aggregation 및 다수 baseline_matches 지원.|
|비용이 주목적이 아님|참고 토큰/시간만 저장. 비용으로 결과를 나누거나 예산 일치를 강제하지 않음.|
|원본 결과 보존|read-only 재분석, 새 output만 허용, raw hash 유지.|
|문서 전체 실패 때문에 분석이 사라지지 않기|모든 계획 블록을 기록하고, 선택 비교 조건의 공통 생성 성공 블록에서 리뷰 생성.|
|지나친 결론 방지|후보 HT 추정과 고유 의미 사실 수 분리. 전수/문서 중복 검토 전에는 unique 결과 null. 블록 신규성을 문서 전체 신규성으로 확대하지 않음.|

## 실제 Novel-10146 978회 검증

원본 ZIP SHA-256 및 파일별 SHA-256은 별도 원본 감사 산출물에 보관했다.
재분석 전후 해제된 source 내 1,023개 파일의 SHA-256이 전부 일치했다.
새 모델 호출 0회, 자동 의미 라벨 0개. 전체 원문이나 raw 응답은 이 commit에 포함하지 않는다.

|분석|계획 블록|공통 성공 블록|유지/검토 후보 relation occurrence|표집 리뷰 항목|
|---|---:|---:|---:|---:|
|strict, 주비교 3조건|52|30|1,119|248|
|missing-unknown, 주비교 3조건|52|30|2,983|670|
|strict, 공통 5조건|52|22|1,119|167|

주비교는 G256 seed17 기준으로 seed29와 G512 seed17을 비교한다.
5조건 설정에서도 비교 방향은 동일하고 cohort만 5조건으로 제한했다.
missing-unknown의 추가 후보 1,864개는 원래 감사의 감도분석 결과와 일치한다.
원본의 qualifier 2,431개를 전부 복구한 것이 아니다. 다른 규약 위반은 계속 제외된다.
세 조건 공통 성공 블록 문자 범위는 139,306/244,216자다. 전체 자료의 의미 결과로 외삽하지 않는다.

빈 review template로 evaluate를 실제 실행하여 모든 문서 macro 의미 추정이 null임을 확인했다.
이는 “새 사실 0개”가 아니라 “아직 검토하지 않음”이다.

## 테스트

`python3 -m unittest discover -s tests -v`: 47개 통과.
기존 38개 회귀 테스트 + 새 9개 테스트:

- 최대 크기 블록, tail, 3조건/5조건 cohort 차이
- 실패한 블록 보존, 공통 성공 0개는 unavailable
- shifted boundary 교차 청크 제외
- qualifier 미기록과 명시적 없음 분리, raw 불변
- 정보 축과 측면 축의 중첩, 전수·중복 검토 조건
- unsupported/uncertain 처리
- snapshot/content/baseline link/span/label 충돌 및 분석 파일 변조 거부
- 표본 포함확률 보정, 후보 추정을 고유 사실 수로 오해하지 않도록 제한
- 성공한 빈 그래프의 0과 비교 불가능 null 구분

`git diff --check` 통과. 실제 LLM 재추출·사람의 의미 검토·QA 성능 측정은 수행하지 않았다.
실행 명령과 라벨 설명은 [FACT_PERSPECTIVES.md](../FACT_PERSPECTIVES.md)에 있다.
