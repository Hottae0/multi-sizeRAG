import json,collections,unicodedata,datetime
from pathlib import Path
base=Path(__file__).resolve().parent;root=base/'retry_results';out=root/'추가실험_확인결과.md'
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' .,;:!?')
def calc(p, overlay=False):
 m=json.loads((p/'manifest.json').read_text());su=json.loads((root/'combined_summary.json' if overlay else p/'summary.json').read_text());look={(r['region'],r['size'],r['seed']):r for r in su['groups']};regs=[r['region_id'] for r in m['regions'] if all(look[(r['region_id'],*c)]['complete'] for c in [(256,17),(256,29),(512,17)])];g=collections.defaultdict(set)
 for j in m['jobs']:
  if j['region_id'] not in regs or (j['size'],j['seed']) not in [(256,17),(256,29),(512,17)]:continue
  record_path=p/'raw'/(j['job_id']+'.json')
  retry_path=root/'raw'/record_path.name
  if overlay and retry_path.exists():
   rr=json.loads(retry_path.read_text())
   if rr.get('parsed') is not None and rr['response']['choices'][0]['finish_reason']=='stop':record_path=retry_path
  r=json.loads(record_path.read_text());ids={e['id']:e['name'] for e in r['parsed']['entities']}
  for x in r['parsed']['relations']:
   if x.get('source') in ids and x.get('target') in ids:g[(j['region_id'],j['size'],j['seed'])].add((norm(ids[x['source']]),norm(x['predicate']),norm(ids[x['target']])))
 totals=collections.Counter()
 for reg in regs:
  a,b,c=[g[(reg,*k)] for k in [(256,17),(256,29),(512,17)]]
  totals.update({'base':len(a),'repeat_added':len(b-a),'multi_added':len(c-a),'repeat_union':len(a|b),'multi_union':len(a|c)})
 return regs,dict(totals),{str(c):sum(r['complete'] for r in su['groups'] if (r['size'],r['seed'])==c) for c in [(128,17),(256,17),(256,29),(512,17),(1024,17)]}
old=calc(base/'results');new=calc(base/'results',overlay=True);retry=json.loads((root/'retry_results.json').read_text());status=json.loads((root/'status.json').read_text());dt=datetime.datetime.fromtimestamp(status['updated_at_unix'],datetime.timezone(datetime.timedelta(hours=9)))
lines=['# 추가 재추출 실험 확인 결과','',f'완료 시각: {dt:%Y-%m-%d %H:%M} KST. 15건 중 13건 종료·파싱 성공, 2건 미해결. 요청 시간 합계 {retry["seconds"]/3600:.2f}시간.','', '모델은 기존 Qwen3-VL-4B-Instruct Q4_K_M이다. 7B 실험이 아니다. 기존 length 종료 15건만 context 65,536, max_tokens=-1로 재실행했다. 무제한 생성 설정이어도 context 한계는 남는다.','', '|조건|기존 완전 구간|재추출 병합 후|','|---|---:|---:|']
for k in old[2]:lines.append(f'|{k}|{old[2][k]}/15|{new[2][k]}/15|')
lines+=['','## 256+256 / 256+512 문자열 비교','', '|항목|기존 공통 11구간|재추출 포함 공통 14구간|','|---|---:|---:|']
for k,label in [('base','기준 256 관계'),('repeat_added','256 반복 추가'),('multi_added','512 추가'),('repeat_union','256+256 합집합'),('multi_union','256+512 합집합')]:lines.append(f'|{label}|{old[1][k]}|{new[1][k]}|')
lines+=['','구간 내 정규화 문자열 중복 제거 후 합산. 유효 fact 수가 아니다. 재추출은 선택적으로 설정을 바꾼 복구 결과이며 동일 설정의 새 독립 실험이 아니다. 기존 11구간은 재추출 대상이 아니므로 원래 비교 결과는 그대로다.','', '## 출력 반복 확인','', '|응답|출력 relation 항목|고유 ID triple|','|---|---:|---:|']
for f in sorted((root/'raw').glob('*.json')):
 r=json.loads(f.read_text());p=r.get('parsed')
 if p:
  rel=p['relations'];c=collections.Counter((x.get('source'),x.get('predicate'),x.get('target')) for x in rel)
  if len(rel)>100:lines.append(f'|{f.stem}|{len(rel)}|{len(c)}|')
lines+=['','예: 512토큰 입력 한 건에서 출력 43,120토큰, 관계 596항목 중 ID triple은 18종뿐이었다. 따라서 길이 제한을 풀어 파싱을 복구해도 추출 품질이 해결되지는 않는다. 위 고유 ID triple 통계는 반복 진단용이며 이름 기반 그래프 집계와 다르다.','', '미해결 응답은 각각 64,795 / 64,283 출력 토큰에서 length 종료했다.','', '본 사전 실험은 일부 잘림이 단순 출력 공간 부족 때문이고, 일부는 반복 생성 때문임을 보여준다. 7B 모델 교체 실험에서는 반복 생성과 주체·대상 연결 오류 감소를 함께 확인하는 것이 유용하다. 새 결과의 의미 유효성 표본 평가는 이번 확인에서 수행하지 않았다.','', '## 기록 해석 주의','', 'combined/결과.md의 기존 v2 설정 유지 및 completed 문구는 자동 보고서 템플릿 문구다. 실제 재시도는 context와 출력 상한을 변경했으며 status.json은 completed_with_unresolved다. 원본 보고서는 수정하지 않고 이 확인 보고서에 차이를 명시했다.','']
out.write_text('\n'.join(lines));print(json.dumps({'old':old,'new':new,'time':str(dt),'hours':retry['seconds']/3600},ensure_ascii=False))
