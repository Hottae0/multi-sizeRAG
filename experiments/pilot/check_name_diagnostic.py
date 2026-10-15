"""Audit a purposively selected hotel/location list, not corpus-level precision."""
import json
from pathlib import Path
from compare_pilot import norm

ROOT = Path(__file__).parent

def main():
    manifest = json.loads((ROOT/'pilot_raw/manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['job_id']=='Novel-10146_e256_s17_c00')
    # The source starts with nineteen explicitly enumerated hotel/address pairs.
    # Restrict the reference to this list; no pronoun resolution or world knowledge.
    listing = job['text'].split('The first hotel fire',1)[0]
    reference = []
    for item in listing.split('.   '):
        item = item.strip().rstrip('.')
        head, tail = item.split(', ',1)
        reference.append({'head':head, 'tail':tail, 'evidence':item,
                          'reviewer':'assistant source inspection; not independent human gold'})
    assert len(reference)==19
    assert all(r['evidence'] in job['text'] for r in reference)
    gold = {(norm(r['head']),norm(r['tail'])) for r in reference}
    heads = {h for h,t in gold}
    rows=[]
    for folder, label in [('pilot_raw','v2-ID'),('name_endpoint_raw','v3-name/plain JSON'),('name_endpoint_schema_raw','v4-name/schema')]:
        for seed in [17,29]:
            path = ROOT/folder/f'Novel-10146_e256_s{seed}_c00.json'
            record=json.loads(path.read_text())
            data=record['parsed'] if folder=='pilot_raw' else record['linked']
            ids={e['id']:e['name'] for e in data['entities']}
            facts=set()
            for r in data['relations']:
                if r.get('source') not in ids or r.get('target') not in ids:
                    continue
                h,t=norm(ids[r['source']]),norm(ids[r['target']])
                if h in heads and norm(r.get('predicate','')) in {'located on','located at','is located on','is located at','is located in','located in'}:
                    facts.add((h,t))
            errors=record.get('automatic_errors',data.get('errors',[]))
            rows.append({'protocol':label,'seed':seed,'reference_pairs':len(gold),
                         'matched_pairs':len(facts&gold),'wrong_location_pairs':len(facts-gold),
                         'missing_pairs':len(gold-facts), 'accepted_entities':len(data['entities']),
                         'accepted_relations':len(data['relations']),
                         'automatic_error_count':len(errors),
                         'seconds':record['elapsed_seconds'],
                         'finish_reason':record['response']['choices'][0]['finish_reason'],
                         'matched':sorted(facts&gold),'wrong':sorted(facts-gold),
                         'missing':sorted(gold-facts)})
    out={'scope':'Same deliberately selected 256-token chunk, two seeds; 19 explicit hotel/address pairs only. Post-hoc assistant reference, NOT corpus precision or general efficacy.', 'reference':reference,'rows':rows}
    (ROOT/'name_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    lines=['# 이름 기반 endpoint 진단', '',
           '기존 출력에서 다른 호텔을 도로로 연결하던 Novel-10146의 첫 E256 청크를 의도적으로 선택했다. 동일 원문·seed 17/29에서 세 프로토콜을 비교했다. 크기 비교나 독립 평가가 아니다.', '',
           '- v2: 모델이 entity ID와 관계 ID 참조를 생성한다.',
           '- v3: head/tail 이름을 출력하고 코드가 ID를 붙인다. JSON object만 제한했다. 두 출력 모두 evidence를 누락하여 스키마 검증에서 항목을 제외했다.',
           '- v4: v3와 같은 프롬프트에 필수 evidence가 있는 JSON 스키마를 추가했다. 모르는 endpoint는 추정하지 않는다.', '',
           '|프로토콜|seed|명시적 호텔 위치 19쌍 중 일치|잘못된 위치 쌍|검증 후 entity/relation|자동 오류 항목|',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['protocol']}|{r['seed']}|{r['matched_pairs']}/19|{r['wrong_location_pairs']}|{r['accepted_entities']}/{r['accepted_relations']}|{r['automatic_error_count']}|")
    lines += ['', '이번 v4 두 출력은 entity 20개를 생성했으나 관계 7개 모두 미등록 endpoint를 사용하여 연결 검증에서 제외됐다. 스키마 준수만으로 의미적 추출이나 연결 완전성이 보장되지 않으며, 이 방식의 개선 효과는 확인하지 못했다.', '', 'v3의 0/19는 원시 모델 출력에 위치 정보가 전혀 없다는 뜻이 아니다. evidence 필드가 빠져 검증 후 사용할 수 있는 관계가 0개라는 뜻이다. 오류 항목 수는 의미 오류 수가 아니다.', '',
              '19개 기준 쌍은 입력의 명시적 목록에서 assistant가 확인했고, 명시적인 located 계열 predicate만 비교했다. 이 제한된 진단을 전체 precision/recall 또는 일반적인 개선율로 보고하면 안 된다. 그 밖의 relation 및 source-grounding은 별도로 검토해야 한다.', '',
              '프롬프트와 출력 제약을 함께 바꿨으므로 ID 사용 여부만의 인과효과는 분리하지 못한다. 표본 확대 전에 같은 스키마 제약 아래 ID 방식과 이름 방식을 비교해야 한다.', '',
              '재현: `python3 experiments/pilot/check_name_diagnostic.py`. 이름 추출의 `--plain-json` 옵션으로 v3를 재현한다. 기본값은 v4다. 요청 전문·응답·seed·시간은 각 raw 폴더에 보존했다. llama.cpp의 schema 전달 형식은 [공식 structured-output 예제](https://github.com/ggml-org/llama.cpp/blob/master/scripts/server-test-structured.py)를 참고했다.']
    (ROOT/'이름기반_재실험.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['matched','wrong','missing']} for r in rows],indent=2))

if __name__=='__main__':
    main()
