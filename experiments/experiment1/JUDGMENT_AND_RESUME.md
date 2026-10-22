# 같은 설정으로 다른 문서 추출 + 원문 기반 판정

2026-10-22. 원본 데이터는 GitHub에 올리지 않는다. 코드·테스트·설명만 추가한다.
기존 `core.py`, `analysis.py`, `study.py`, `cuda_server.py`, `perspectives.py`를 변경하지 않아
기존 study의 구현 hash 및 이미 만든 248개 리뷰의 측정 hash를 보존한다.

## 지금 먼저 할 일: 다른 개발 문서 하나

세션 재접속 후 서버 파일과 모델 서버가 살아 있으면 ZIP을 다시 풀 필요가 없다.
새 명령은 study.json의 모델·프롬프트·tokenizer·seed·크기·출력 한도·온도 등을 그대로 읽는다.
모델 다운로드나 서버 실행은 자동으로 하지 않는다. 서버 identity가 다르면 기존 검증기가 거부한다.

```bash
cd /home/work/multi-sizeRAG/repo-fact-perspective-v1
git pull --ff-only

# 새 문서 20회로 먼저 정상 실행 여부 확인 (tokenize 요청도 필요)
python3 -m experiments.experiment1.next_document \
  --study /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1 \
  --doc Novel-30752 --limit 20

# 같은 설정으로 남은 호출 실행. 세션 연결이 끝나도 실행이 이어지도록 nohup 사용.
mkdir -p runs
nohup python3 -u -m experiments.experiment1.next_document \
  --study /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1 \
  --doc Novel-30752 > runs/Novel-30752-extract.log 2>&1 &
```

위 두 추출 명령은 동시에 실행하지 않는다. 먼저 20회 명령이 끝난 뒤 나머지를 실행한다.
첫 20회가 이미 기록돼 있으면 나머지 명령은 중복 실행하지 않는다. 실패/중단 당시 결과 불명 호출도 자동 재시도하지 않는다.
새 문서의 호출 수는 원문 길이에 따라 달라지며 978회로 고정되지 않는다. 기존 Novel-10146 raw는 수정하지 않는다.
실행 중인 프로세스가 없는 것이 확인된 경우에만 강제 종료가 남긴 `.study.lock`/`.extract.lock`을 정리한다. 자동 삭제 기능은 없다.
평가 문서는 이 간편 명령으로 실행할 수 없다. 기존 평가군 동결 절차를 사용한다.

모두 끝난 뒤 새 문서 결과 분석:

```bash
python3 -m experiments.experiment1.perspectives analyze \
  --run /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1/documents/Novel-30752 \
  --out runs/Novel-30752-facts-strict-v1
```

그다음 개발 문서는 `Novel-41603`, `Novel-44557`, `Novel-54537`이다.
문서 하나가 끝나면 JSON 완료율·qualifier 누락·유지 관계 수를 확인하고 같은 설정으로 진행한다.
프롬프트를 바꾸면 같은 조건 반복이 아니므로 별도 실험으로 구분한다.

## 판정 코드가 추가하는 것

|단계|구현|
|---|---|
|판정 자료|블록 원문 + 대상 관계 + 같은 블록의 전체 기준 관계를 packet으로 저장. 크기·seed는 노출하지 않음.|
|주장 지지|subject/object/relation/direction/polarity/time/condition/attribution 8개 항목별 상태와 이유.|
|실제 근거|exact quote를 원문에서 찾아 절대 문자 범위를 계산. 인용이 없거나 다르면 거부. 여러 위치에 매칭되면 보류.|
|기준 그래프 검사|대응하는 기준 관계도 독립적으로 8개 항목 검토. 여러 관계를 연결할 수 있음.|
|정보 차이|equivalent/refinement/broader/new_claim/uncertain. 기존 정보·추가 정보를 설명.|
|다른 측면|같은 사건인지 확인하고 이전 측면·추가 측면을 설명. 단순 동의 표현은 same.|
|오류 수정|대상 관계가 지지되고 관련 기준 관계가 미지지일 때 correction. 신규 사실/관점 집계에서 별도 분리.|
|1차 판정|HTTP chat-completions 호환 모델의 제안. author_kind=model, 요청·응답·usage·시간·실패 기록.|
|재검토|미판정/실패/불확실/오류 수정 전부 + 나머지 확정 제안의 기본 20% 무작위 점검.|
|집계|모델 제안과 실제 재검토 라벨을 분리. 부분 재검토를 전체의 검증 완료로 표시하지 않음.|

중요: 코드는 정확한 인용 위치·필드·논리적 일관성을 검증한다. 문장 의미 자체의 진실성을 자동 증명하지 않는다.
모델 판정은 임시 제안이며, 두 번째 모델을 사용하더라도 사람 판정으로 표시하지 않는다.
신규성은 같은 블록의 기준 그래프에 대한 신규성이다. 문서 전체의 고유 신규 사실 수를 자동 산출하지 않는다.

## 첫 20개로 판정 절차 점검

이미 서버에서 만든 Novel-10146 분석 결과를 그대로 사용한다. 경로가 다르면 --analysis만 수정한다.
prepare는 오프라인이며 모델을 호출하지 않는다. **draft부터 모델 호출이 발생**한다.
원래 추출 모델과 같은 모델로 판정할 수도 있지만 같은 오류를 반복할 수 있어 재검토가 필요하다.

```bash
python3 -m experiments.experiment1.judgments prepare \
  --analysis runs/Novel-10146-facts-strict-v1 \
  --out runs/Novel-10146-judgment-packets-v1

python3 -m experiments.experiment1.judgments draft \
  --packets runs/Novel-10146-judgment-packets-v1 \
  --out runs/Novel-10146-judge-drafts-v1 \
  --url http://127.0.0.1:11439/v1 --model extraction-cuda \
  --identity-file /home/work/multi-sizeRAG/repo/runs/novel_nohup_v1/study.json \
  --limit 20 --max-tokens 2048
```

`--url`은 `/chat/completions` 바로 앞까지의 주소다. 모델 이름은 실제 실행 중인 서버에 맞춘다. 이 저장소의 CUDA 서버는 identity-file이 필요하며 응답의 server_id도 대조한다. 다른 호환 서버를 쓰면 해당 옵션을 생략할 수 있다. 입력+출력 한도가 서버 context를 넘으면 실패로 기록하며 임의로 원문을 자르지 않는다.
API key가 필요한 호환 서버는 기존 `EXPERIMENT1_API_KEY` 환경변수를 사용하며 파일에 key를 저장하지 않는다.
명령을 반복하면 미기록 항목만 최대 limit개 진행한다. 시작 기록만 남은 호출·JSON 실패·길이 제한·검증 실패도 자동 재전송하지 않는다.
동일 출력 폴더에서 모델/프롬프트/출력 한도 등을 바꾸면 거부한다. 수정할 때는 새 packet/draft 폴더를 만들어 버전을 구분한다.
이전 설정의 failed 기록을 지워 성공 결과로 대체하지 않는다.

첫 20개 출력에서 다음을 확인한 뒤 범위를 확대한다:

- source에 있는 인용만 사용했는가?
- 행위 성공과 시도, 부정과 긍정, 발화 귀속을 구분했는가?
- 대응 기준 관계가 실제로 같은 사건인가?
- 새로운 주장/구체화/표현 차이와 오류 수정을 구분했는가?
- 확인할 수 없는 것을 uncertain으로 남기는가?

추가 판정을 진행하려면 같은 명령의 limit만 늘릴 수 있다. 예를 들어 `--limit 248`은 최대 248개의 **미기록** 항목을 시도한다.
원문 978회 추출을 다시 수행하는 것이 아니라 생성된 리뷰 항목을 판정한다.

## 재검토와 집계

```bash
python3 -m experiments.experiment1.judgments review-queue \
  --packets runs/Novel-10146-judgment-packets-v1 \
  --drafts runs/Novel-10146-judge-drafts-v1 \
  --out runs/Novel-10146-review-queue-v1.json
```

각 항목의 packet과 model_proposal을 읽고 다음만 작성한다:

- `final_judgment`: 동일 template 구조의 최종 판정. 모델 제안을 무조건 복사하지 말고 원문과 확인한다.
- `reviewer`, `reviewer_kind` (`human` 또는 `model`), `rationale`.
- 원래 packet/id/packet_set_hash는 수정하지 않는다.

모델 제안을 보고 진행하는 검토이므로 독립 맹검이나 평가자 간 일치도를 주장하지 않는다.
전체 검토가 필요하면 `--audit-fraction 1`로 모든 항목을 queue에 넣는다.

```bash
python3 -m experiments.experiment1.judgments evaluate \
  --packets runs/Novel-10146-judgment-packets-v1 \
  --drafts runs/Novel-10146-judge-drafts-v1 \
  --analysis runs/Novel-10146-facts-strict-v1 \
  --reviews runs/Novel-10146-review-queue-v1.json \
  --out runs/Novel-10146-judgment-evaluation-v1.json
```

reviews를 생략하면 모델 임시 제안의 관찰 집계만 확인한다.
`model_provisional_observed_candidates`와 `reviewed_observed_candidates`를 구분한다.
모든 표집 항목의 재검토가 확정되지 않으면 `fully_reviewed_candidate_estimates`는 null이다.
이 값도 고유한 의미 사실 수가 아닌 후보 edge의 포함확률 보정 추정이다. 관점과 신규성 축은 겹칠 수 있다.
오류 수정은 별도 카운트이며 신규/관점 증가로 합산하지 않는다.

## 서버 파일이 없을 때만 ZIP 복원

세션 재접속과 파일 삭제는 다르다. 서버 파일이 남아 있으면 이 절차는 생략한다.
다운로드했던 ZIP에는 원시 결과가 있으므로 저장소에 업로드하지 않고 로컬/서버의 새 폴더로 해제한다.

```bash
python3 -m experiments.experiment1.restore \
  --archive /path/to/vfolder-multi-sizeRAG-20261022T093939Z.zip \
  --out runs/restored-v1
```

복원 명령은 ZIP→tar.gz 중첩과 직접 tar.gz/일반 ZIP을 지원한다.
경로 이탈·링크·특수파일·중복 경로·확장 크기를 검사하고 새 폴더에만 쓴다.
`restore_manifest.json`에 원본 ZIP 해시, 파일별 해시, manifest/raw 위치와 누락 호출 수를 기록한다.
이 ZIP의 study 경로는 `runs/restored-v1/source/novel_nohup_v1`이다. 이후 --study/--run에 복원 경로를 사용한다.
복원된 응답에 적힌 예전 서버 절대 경로는 provenance이며, 새 위치의 manifest/raw를 읽어 재분석한다.

## 검증 범위

56개 표준 라이브러리 테스트 통과. 실제 ZIP 복원 후 978/978회 기록, 공통 성공 30블록, 리뷰 248개 및 기존 measurement_hash 일치.
248개 판정 packet 생성은 실제 원본으로 검증했고 모델 호출은 0회였다.
모델 통신·재개·실패 기록·오류 수정 집계는 작성한 fixture 응답으로 검증했다.
실제 서버의 새 문서 추출/실제 모델 의미 판정 품질은 이 로컬 검증에서 실행하지 않았다.
