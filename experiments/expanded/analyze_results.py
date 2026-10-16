import json,hashlib,collections,statistics,itertools,datetime
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent/'results');args=parser.parse_args()
root=args.root;m=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
graphs={};agg=collections.defaultdict(collections.Counter);error_types=collections.Counter();records=[]
def norm(s):
 import unicodedata
 return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' .,;:!?')
for j in m['jobs']:
 r=json.loads((root/'raw'/(j['job_id']+'.json')).read_text());assert r['job']==j
 assert hashlib.sha256(j['text'].encode()).hexdigest()==j['sha256']
 assert r['request']['messages']==[{'role':'system','content':m['prompt']},{'role':'user','content':'PASSAGE:\n'+j['text']}]
 assert r['request']['seed']==j['seed'] and r['request']['max_tokens']==4096
 key=(j['size'],j['seed']);a=agg[key];a['calls']+=1;a['errors']+=bool(r['automatic_errors']);a['parse_fail']+=r['parsed'] is None;a['truncated']+=r['response']['choices'][0]['finish_reason']=='length';a['seconds']+=r['elapsed_seconds']
 for e in r['automatic_errors']:error_types[e.split(':')[-1].strip() if ':' in e else e]+=1
 p=r['parsed'];a['empty']+=isinstance(p,dict) and not p.get('entities') and not p.get('relations');usage=r['response'].get('usage',{});a['input_tokens']+=usage.get('prompt_tokens',0);a['output_tokens']+=usage.get('completion_tokens',0)
 g=graphs.setdefault((j['region_id'],*key),{'entities':set(),'relations':set()})
 if isinstance(p,dict):
  ids={x['id']:x['name'] for x in p['entities']}
  g['entities'].update(norm(v) for v in ids.values())
  g['relations'].update((norm(ids[x['source']]),norm(x['predicate']),norm(ids[x['target']])) for x in p['relations'] if x.get('source') in ids and x.get('target') in ids)
 records.append(r)
rows=summary['groups'];lookup={(r['region'],r['size'],r['seed']):r for r in rows};regions=[r['region_id'] for r in m['regions']];common=[reg for reg in regions if all(lookup[(reg,s,17)]['complete'] for s in [128,256,512,1024])]
matched=[]
for size in [128,256,512,1024]:
 rr=[lookup[(reg,size,17)] for reg in common];matched.append({'size':size,'regions':len(rr),'entities_sum':sum(r['entities'] for r in rr),'relations_sum':sum(r['relations'] for r in rr),'relations_mean':statistics.mean(r['relations'] for r in rr)})
pairs=[]
for ca,cb in [((128,17),(256,17)),((128,17),(512,17)),((256,17),(512,17)),((512,17),(1024,17)),((256,17),(256,29))]:
 eligible=[reg for reg in regions if lookup[(reg,*ca)]['complete'] and lookup[(reg,*cb)]['complete']]
 for kind in ['entities','relations']:
  vals=[]
  for reg in eligible:
   a,b=graphs[(reg,*ca)][kind],graphs[(reg,*cb)][kind]
   if a|b:vals.append(len(a&b)/len(a|b))
  pairs.append({'a':ca,'b':cb,'kind':kind,'paired_regions':len(eligible),'mean_jaccard':statistics.mean(vals) if vals else None})
result={'completed':len(records),'total_seconds':sum(r['elapsed_seconds'] for r in records),'groups':[{'size':k[0],'seed':k[1],**a,'complete_regions':sum(r['complete'] for r in rows if (r['size'],r['seed'])==k)} for k,a in sorted(agg.items())],'matched_regions':common,'matched_counts':matched,'pairs':pairs,'error_types':dict(error_types),'limitations':['Lexical counts, not valid facts','Complete-case comparison selects 6/15 regions from 3/5 documents; selection bias','Same fixed output cap causes size-dependent censoring','No new human semantic annotations or QA evaluation']}
(root/'점검통계.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# Qwen 청크 크기 확대 실험 결과', '', '완료: 2026-10-16 09:31 KST. GraphRAG-Bench Novel corpus 20문서에서 고정 seed로 선택한 5문서의 비중복 15구간을 사용했다. 원문 총량은 최초 파일럿의 30배다. 문서 전체 추출이나 독립 holdout 평가는 아니다.', '', '## 연구 질문과 조건', '', '같은 원문을 다른 크기로 나누었을 때 entity·relation 추출 결과가 얼마나 달라지는지 확인했다. 기존 설정을 유지하고 데이터 범위만 확대했으며, overlap 실험은 수행하지 않았다.', '', '모델: Qwen3-VL-4B-Instruct Q4_K_M(텍스트 입력), context 8192, temperature 0.2, top_p 0.9, repeat_penalty 1.1, max_tokens 4096, cache_prompt false, parallel 1. 조건: E128/E256/E512/E1024 seed 17 및 E256 seed 29. Overlap=0. 원문 구간당 각 조건의 청크 수는 32/16/8/4개이며, 256 반복 16개를 더해 76회 × 15구간 = 1140회다.', '', '문서: Novel-10146, Novel-30752, Novel-41603, Novel-44557, Novel-54537. 원문·프롬프트·seed·출력 상한과 응답별 기록의 일치 여부를 점검했다. 생성 결과의 의미적 정답 여부를 전수 검증한 것은 아니다.', '', '## 실행 완료와 오류', '', '5문서 × 3구간 × 4096토큰 = 61440토큰. 1140회 응답 모두 저장 완료. 요청 시간 합계 %.2f시간. 기존 v2 설정과 원문·프롬프트·seed·출력 상한을 점검했다.'%(result['total_seconds']/3600), '', '|크기|seed|호출|빈 출력|출력 잘림/파싱 실패|자동 오류 호출|완전한 구간/15|', '|---|---:|---:|---:|---:|---:|---:|']
for r in result['groups']:lines.append(f"|{r['size']}|{r['seed']}|{r['calls']}|{r['empty']}|{r['truncated']} / {r['parse_fail']}|{r['errors']}|{r['complete_regions']}|")
lines+=['', '자동 오류는 근거 문자열·endpoint·파싱 등의 점검이며 의미적 정확도 지표가 아니다. 빈 출력은 원문에 관계가 없다는 판정이 아니다. 출력 잘림 때문에 파싱 실패한 결과를 빈 그래프로 취급하지 않았다.', '', '## 모든 크기가 정상 완료된 동일 6구간', '', '15구간 중 공통 6구간(3문서)만 포함한다. 어려운 구간이 제외된 편향이 있어 전체 5문서로 일반화할 수 없다. 아래 합계는 구간별 중복 제거 수를 합친 것이며 구간 간 중복은 제거하지 않았다. 유효 fact 수가 아니다.', '', '|크기|entity 합계|relation 합계|구간당 relation 평균|','|---|---:|---:|---:|']
for r in matched:lines.append(f"|{r['size']}|{r['entities_sum']}|{r['relations_sum']}|{r['relations_mean']:.1f}|")
lines+=['','포함 구간: '+', '.join(common), '', '## 크기 차이와 seed 차이', '', '각 비교에서 양쪽 모두 정상 완료된 구간만 사용한 문자열 Jaccard 평균. 비교마다 구간 수가 다르므로 효과 크기를 직접 대조하면 안 된다. 부분 정렬 규칙은 옛 두 표본용이어서 새 구간에 적용하지 않았다.', '', '|조건 A|조건 B|구간 수|대상|평균 Jaccard|','|---|---|---:|---|---:|']
for r in pairs:lines.append(f"|{r['a']}|{r['b']}|{r['paired_regions']}|{r['kind']}|{r['mean_jaccard']:.3f}|")
lines+=['', '## 현재 해석', '', '- 큰 청크는 동일한 4096 출력 상한에서 잘림이 더 자주 발생했다. 이것은 현재 추출 설정의 한계이며 큰 청크가 원래 열등하다는 증거가 아니다.', '- 완전한 공통 구간에서는 E512의 문자열 relation 합계가 가장 많았지만, precision과 유효한 상호보완성은 아직 검증하지 않았다.', '- E128은 빈 출력이 많다. 문맥 부족 또는 추출 누락 가능성과 실제 관계 부재를 원문 대조로 구분해야 한다.', '- 기존 설계를 변경하지 않은 확대 실행은 끝났다. 후속 실험은 출력 상한·스키마 안정화 여부를 별도로 정하고 원 결과와 섞지 않아야 한다.', '- 독립 표본은 1140개가 아니라 5문서 안의 15구간이다. 사람 검토·동일 비용 대조·QA 평가는 아직 없다.']
lines += ['', '## 후속 검증 제안', '', '1. 출력 상한·스키마 안정화는 별도 실험으로 수행하고, 큰 청크에만 조건을 바꾼 결과를 원 비교에 합치지 않는다.', '2. 크기와 seed를 가린 원문 검토로 supported / unsupported / ambiguous를 구분한다. 명칭·관계 방향을 정렬한 뒤 유효한 고유 관계를 센다.', '3. 동일 비용의 단일 크기 반복 추출과 다중 크기 통합을 비교하고, 경계 손실을 구분하기 위한 overlap 대조군을 후속 실행한다.', '4. 이후 검색·QA까지 검증해야 다중 크기 통합의 효용을 주장할 수 있다.', '', '## 보고 시 전달할 결론', '', '> 5문서·15구간으로 확대해도 청크 크기에 따라 추출 결과가 달라졌다. 모든 크기가 정상 완료된 공통 6구간에서는 E512의 관계 문자열 합계가 가장 컸다. 다만 큰 청크의 출력 잘림과 작은 청크의 빈 출력이 발생했고, 표현 차이·추출 오류를 분리하지 못했으므로 유효한 상호보완성이나 성능 우위는 아직 입증하지 않았다.', '', '## 출처와 재현 기록', '', '- 데이터: [GraphRAG-Bench 공식 저장소](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark), Datasets/Corpus/novel.json.', '- [실험 저장소 및 연구 배경](https://github.com/Hottae0/multi-sizeRAG)', '- [확대 실험 코드·표본 선정](https://github.com/Hottae0/multi-sizeRAG/tree/main/experiments/expanded)', '- [원시 요청·응답 및 집계](https://github.com/Hottae0/multi-sizeRAG/tree/main/experiments/expanded/results)', '', '코드 재현: 저장소 루트에서 `python3 experiments/expanded/analyze_results.py`를 실행하면 저장된 응답으로 통계와 본 보고서를 재생성한다. 모델 재실행은 필요 없다.']
(root/'확대실험_점검보고서.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ['matched_regions','matched_counts','pairs']},ensure_ascii=False,indent=2))
