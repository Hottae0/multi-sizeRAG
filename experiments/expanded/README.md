# 범위만 확대한 추출 실험

2026-10-15 착수. 기존 v2 추출 조건을 고정하고 표본을 5문서 × 3구간 × 4096토큰(61440토큰)으로 확대했다. 총 1140회 요청. 이 문서는 실행 계획이며 결과 완료를 뜻하지 않는다.

- 문서: Novel-10146, Novel-30752, Novel-41603, Novel-44557, Novel-54537.
- 조건: E128/E256/E512/E1024 seed 17, E256 seed 29. Overlap=0.
- 기존과 동일: Qwen3-VL-4B-Instruct Q4_K_M, v2 prompt, temperature .2, top_p .9, repeat_penalty 1.1, max_tokens 4096, cache_prompt false, context 8192, parallel 1.
- 범위 변경에 따른 표본 구성: seed 20261015로 5문서를 선택하고, 각 문서의 10–80% 문자 위치 범위에서 문장 시작점 뒤 연속 구간 3개를 겹치지 않게 선정한다. 기존 2문서도 포함되므로 독립 holdout 평가가 아니다. 모든 조건은 같은 원문 구간을 덮는다.
- 원문 토큰 총량은 기존 2048의 30배다. 요청 수 1140은 독립 표본 수가 아니다. 독립적인 문서 수는 5개다.
- 출력 잘림을 피하려고 일부 조건만 설정을 바꾸지 않는다. 잘림·파싱 실패는 기록하고 해당 구간/조건의 완전한 그래프 개수 비교에서 제외한다.
- 이 실행에서는 overlap, 이름 기반 프로토콜, 의미 정렬 규칙, QA 평가를 추가하지 않는다. 원시 요청·응답 및 문자열 기준 집계를 보존한다.

## 실행

저장소 루트에서 실행한다. 이전 파일럿과 같은 corpus 파일이 필요하다.

```bash
python3 experiments/expanded/run_expanded.py \
  --corpus data/novel.json --out runs/expanded \
  --server /path/to/llama-server \
  --model /path/to/Qwen3VL-4B-Instruct-Q4_K_M.gguf
```

실행기가 자신의 서버를 시작하고 종료한다. 기본 포트 11439가 사용 중이면 시작하지 않으며 `--port`로 변경할 수 있다. macOS에서는 실행 중 `caffeinate -i`로 유휴 절전을 막고 종료 시 해제한다. 기기가 종료되거나 수면에 들어가면 실행이 지속된다고 보장할 수 없다.

`status.json`은 현재 요청과 기록 수, `raw/`는 호출별 결과다. 10회마다 `summary.json`과 `결과.md`를 갱신하고 최종 종료 시 다시 생성한다. 보고서는 lexical entity/relation 수이며 의미적 유효성이나 precision이 아니다. `completed`는 요청을 모두 기록했다는 뜻이고, 모든 응답이 올바르다는 뜻은 아니다.

같은 명령으로 재개하면 입력·요청 설정을 검증한 뒤 기록된 모델 응답을 건너뛴다. 파싱 실패·잘린 응답도 보존한다. 통신 실패가 생기면 중단하고 기록을 남긴다. 통신 실패 기록은 원인을 점검하고 별도 보존한 뒤 재개해야 한다. 실행 설정과 다른 기록을 덮어쓰지 않는다.

표본 구간 위치·해시는 [selection.json](selection.json)에 공개했다. 대규모 추출 결과는 완료 및 검토 전까지 기존 파일럿 결과와 구분한다.
