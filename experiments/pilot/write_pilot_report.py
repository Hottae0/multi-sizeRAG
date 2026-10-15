import json
from pathlib import Path
from collections import defaultdict
root=Path(__file__).parent
r=json.loads((root/'pilot_comparison.json').read_text())
assert r['completed']==r['planned'], 'Do not publish a partial experiment as complete'
m=json.loads((root/'pilot_raw/manifest.json').read_text());a=json.loads((root/'diagnostic_examples.json').read_text())
lines=['# Qwen 청크별 추출 파일럿 결과','', '## 실행 범위','',f"공식 GraphRAG-Bench Novel corpus 20개 중 seed=20261015로 문서 2개를 선정하고, 각 1,024 모델 토큰의 연속 구간을 사용했다. 선택 문서: {', '.join(x['doc_id'] for x in m['regions'])}. 두 구간은 호텔·도시 기록과 여행 서술이며, 전형적인 소설 전체를 대표한다고 가정하지 않는다.",'', 'Qwen3-VL-4B-Instruct Q4_K_M, llama-server, context 8192, temperature 0.2, top_p 0.9, repeat_penalty 1.1, max_tokens 4096. 세 크기(128/256/512)의 기본 seed는 17, E256 반복 seed는 29. Overlap=0. 모든 조건이 동일한 원문을 정확히 복원하며 청크별 token/char offset을 저장했다. 총 36건 완료.','', '첫 설정에서 반복 생성·출력 잘림을 확인하여 중단했다. `pilot_raw_initial`에 초기 기록을 보존하고, 간결한 출력·중복 억제 설정을 추가한 뒤 같은 표본의 36건을 모두 다시 실행했다. 아래 표에는 재실행 결과만 포함한다. 이 과정은 개발용 파일럿이므로 확증 실험으로 해석하지 않는다.','', '## 크기별 추출량','', '**아래 개수는 유효 fact 수가 아니다.** 이름과 predicate를 NFKC·소문자·공백으로 정규화한 문자열 기준이다. 관계는 source/target ID가 존재하는 경우만 비교 집합에 포함한다. 의미적 별칭 통합·동의어 통합·관계 역방향 통합은 수행하지 않았다.','', '| 문서 | E | seed | 호출 | Entity 명칭 | 연결 가능한 관계 | 원시 관계 수 | 빈 출력 | 자동 오류 호출 | 잘림 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in r['groups']:
 lines.append(f"| {g['doc']} | {g['size']} | {g['seed']} | {g['calls']} | {g['entities']} | {g['relations']} | {g['raw_relations']} | {g['empty_calls']} | {g['calls_with_errors']} | {g['truncated']} |")
lines+=['','Entity 명칭과 관계는 문서·조건 내 중복 제거 수다. 원시 관계 수는 각 청크 응답의 relation 항목 수 합계다. 자동 오류는 근거 문자열 불일치·필드 누락·잘못된 endpoint 등을 포함하며, 의미 오류 전체를 포착하지 않는다. 잘린 응답이나 파싱 실패가 있다면 해당 조건은 완전한 개수 비교에서 제외해야 한다.','', '## 얼마나 겹치는가','', 'Jaccard = 교집합 / 합집합. 값이 낮다고 곧바로 유효한 상호보완성이 크다는 뜻은 아니다. 별칭·문구·주체/대상 방향 차이도 값을 낮춘다.','', '| 문서 | 조건 A | 조건 B | Entity Jaccard | Relation Jaccard |','|---|---|---|---:|---:|']
for doc in [x['doc_id'] for x in m['regions']]:
 for aa,bb in [([128,17],[256,17]),([128,17],[512,17]),([256,17],[512,17]),([256,17],[256,29])]:
  ps={p['kind']:p['jaccard'] for p in r['pairs'] if p['doc']==doc and p['a']==aa and p['b']==bb}
  fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
  lines.append(f"| {doc} | E{aa[0]} / seed{aa[1]} | E{bb[0]} / seed{bb[1]} | {fmt(ps['entities'])} | {fmt(ps['relations'])} |")
lines+=['','## 각 크기에서만 나온 관계 문자열','', '기본 seed17끼리 비교. 의미적으로 고유하거나 유효하다는 판정이 아니다.','', '| 문서 | E128 only | E256 only | E512 only |','|---|---:|---:|---:|']
for doc in [x['doc_id'] for x in m['regions']]:
 d={u['size']:u['lexically_unique_relations'] for u in r['unique'] if u['doc']==doc};lines.append(f"| {doc} | {d[128]} | {d[256]} | {d[512]} |")
lines+=['','## 원문 대조 사례','', '다음은 assistant가 입력 원문과 대조한 진단 사례다. 독립적인 사람 주석 또는 무작위 precision 표본이 아니므로 이 사례들의 정답 비율을 전체 정확도로 보고하지 않는다. 모든 사례는 원시 JSON에서 해당 triple을 확인했다.','']
for i,c in enumerate(a,1):
 lines +=[f"### {i}. {c['head']} / {c['predicate']} / {c['tail']}",f"- 파일: `{c['job']}.json`",f"- 원문 대조: **{c['label']}**",f"- 근거 문자열 완전 일치: {c['evidence_exact_match']}",f"- 판단: {c['reason_ko']}",'']
lines+=['## 경계 효과','', 'Novel-10146의 E256/seed17/chunk01에서는 청크가 `ominie hotel`로 시작하고, 모델도 이를 entity 이름으로 반환했다. 전체 표본의 명칭은 `Sintominie hotel`이다. 이는 fixed-token 경계에 따른 이름 손상이다. 의미적 상호보완성과 분리하여 판단해야 하며, 후속 실험에 문장 경계 분할 또는 경계 이동 대조군이 필요하다.','', '## 비용과 실행 품질','']
secs=sum(x['seconds'] for x in r['runs']);prompt=sum(x['usage'].get('prompt_tokens',0) for x in r['runs']);out=sum(x['usage'].get('completion_tokens',0) for x in r['runs']);cached=sum(x['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) for x in r['runs'])
lines +=[f'- 완료 요청: {r["completed"]}/{r["planned"]}',f'- 재실행 요청 시간 합계: {secs:.1f}초 ({secs/60:.1f}분). 모델 로드·초기 실패 실행·분석 시간 제외.',f'- 입력 토큰: {prompt:,}; 출력 토큰: {out:,}; 서버 보고 cached input tokens: {cached:,}.',f'- 잘린 응답: {sum(x["finish_reason"]=="length" for x in r["runs"])}건.',f'- 자동 오류가 있는 호출: {sum(bool(x["errors"]) for x in r["runs"])}건.','', '## 현재 결론','', '여행 구간에서는 E128만 소녀의 Board School 미경험 관계를, E512만 Devil’s Frying Pan과 Cadgwith의 근접 관계를 정확한 pair로 추출했다. 기본 seed17의 출력에 대한 진단적 사례이며, scale별 고유한 유효 관계가 존재할 수 있다는 소규모 관찰이다. 반복으로 확정한 효과나 모집단 precision 추정은 아니다.','', '청크 크기를 바꾸면 추출 결과는 달라진다. 그러나 현재 차이에는 이름 손상, 관계 표현·방향 차이, endpoint 오류와 의미 오류가 함께 섞여 있다. 원시 관계 수 증가나 낮은 문자열 Jaccard만으로 multi-scale fusion의 유효성을 주장할 수 없다.','', '다음 단계는 entity 별칭과 역관계를 정렬하고, 조건을 숨긴 독립 원문 검증으로 유효한 고유 관계를 세는 것이다. 같은 크기 반복 추출 대조군을 모든 크기에 확대하고 경계 이동 대조군을 추가한다. 명시적인 entity 이름으로 관계를 출력한 뒤 코드에서 ID를 부여하는 추출 schema도 별도 개발 표본에서 시험할 가치가 있다.','', '## 재현 파일','', '- `실행방법.md`: 실행 순서와 코드 설명.', '- `pilot_raw/manifest.json`: 표본·모든 청크·offset·입력 해시.', '- `pilot_raw/*.json`: 요청·원시 응답·자동 검사.', '- `pilot_comparison.json`: 집합 비교와 관계별 근거.', '- `diagnostic_examples.json`: 위 진단 사례의 추적 가능한 기록.', '- 원본: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Datasets/Corpus/novel.json','']
(root/'청크별_추출_실험결과.md').write_text('\n'.join(lines))
print('report written')
