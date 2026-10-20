# 전체 문서 실행 구현 검증 — 2026-10-21

기준 main은 `debd4b155cb5d53a89200fbf58bea6f463826208`입니다.
기존 사전실험 데이터와 과거 결과 기록은 수정하지 않았습니다.

## 확인한 내용

- `python -m unittest discover -s tests -q`: 38개 테스트가 통과했습니다.
- `python -m compileall -q experiments/experiment1`, `git diff --check`를 확인했습니다.
- 모의 20문서와 모의 tokenizer/추출 응답으로 1→5→20 누적 실행, 기본 5조건, 전체 원문 꼬리·UTF-8 coverage,
  먼저 완료한 문서의 raw 재사용, 다음 문서 추출 전 그래프 저장, freeze 전 평가군 추출 차단을 검증했습니다.
- 변경된 프로토콜·원문·동결 정책 거부, 실패 요청 재시도 방지, 실패 문서 제외와 비용 보존을 검증했습니다.
- 의미 정렬·compact baseline 참조·독립 판정 파일 집계 후 256+512 대 256+256의 합성 데이터 기대 delta=1을 확인했습니다.
  미판정은 null이며, 검토된 빈 그래프는 0으로 구분했습니다.
- CUDA token offset/서버 identity 검사 및 큰 2-hop 경로 계산 상한의 null 처리를 검증했습니다.
- 기존 1,140개 실제 raw 응답을 재분석하여 저장된 `legacy_summary.json`의 주요 집계와 일치함을 확인했습니다.
  총 호출 1,140, 잘림 15, 자동 오류 호출 475, 빈 출력 380, entity occurrence 6,471,
  relation occurrence 4,517 및 공통 완료 6구간이 동일했습니다.

## 검증하지 않은 내용

실제 A100 장치에서 모델 적재·CUDA 추론·처리량·VRAM 사용량은 아직 검증하지 않았습니다.
새 모델과 프롬프트의 추출 품질, 실제 전체 문서 의미 정렬·원문 판정, fusion·QA 효과도 아직 검증하지 않았습니다.
모의 테스트는 코드 동작 검증이며 연구 가설을 지지하는 실험 결과가 아닙니다.
