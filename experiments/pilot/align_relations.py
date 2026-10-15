"""Conservative, inspectable post-hoc alignment. This is NOT an entailment judge."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from compare_pilot import norm

# These aliases were inspected in the two sampled source regions. They are not
# a learned/general resolver. Truncated names (ominie, low house) stay unchanged.
ALIASES = {
    'Novel-10146': {
        'the new england': 'new england', 'the empire saloon': 'empire saloon',
        'the winslow': 'winslow house',
        "st. paul's principal hotel": 'rice house',
    },
    'Novel-44557': {
        'mary': 'mary mundy', 'the old inn': 'old inn',
        "mary's brother": "mary mundy's brother",
        'the tiny farm-house on the hill-side near the frying-pan': 'tiny farm-house',
        'scotch professor': 'professor', 'greek professor': 'professor',
        'the greek professor': 'professor',
        'the enthusiastic scotch professor': 'professor',
        'the noted greek professor': 'professor',
    },
}
# value = canonical predicate, reverse endpoints. Never drop negation or repair
# a malformed claim (e.g. person "constructed by" hotel remains reversed).
PREDICATES = {
    'located on': ('located at', False), 'located at': ('located at', False),
    'was located on': ('located at', False), 'was located at': ('located at', False),
    'was located in': ('located at', False), 'situated at': ('located at', False),
    'is near': ('near', False), 'is located near': ('near', False),
    'located near': ('near', False),
    'constructed': ('constructed', False), 'constructed hotel': ('constructed', False),
    'was constructed by': ('constructed', True), 'constructed by': ('constructed', True),
    'was among firemen who assisted in saving furniture': ('helped save furniture at', False),
    'was a fireman who assisted in saving the furniture': ('helped save furniture at', False),
    'described in guide-book': ('described in guide-book', False),
    'is described in guide-book': ('described in guide-book', False),
    'is brother of': ('brother of', False), "is mary's brother": ('brother of', False),
    "has poem in visitors' book and portrait in album": ('has poem and portrait by', False),
    "has a poem in her visitors' book and portrait in her album": ('has poem and portrait by', False),
}

def align(doc, fact):
    head, predicate, tail = map(norm, fact)
    aliases = ALIASES.get(doc, {})
    head, tail = aliases.get(head, head), aliases.get(tail, tail)
    predicate, reverse = PREDICATES.get(predicate, (predicate, False))
    if reverse:
        head, tail = tail, head
    return head, predicate, tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    baseline = json.loads((args.root / 'pilot_comparison.json').read_text())
    if baseline['completed'] != baseline['planned']:
        raise ValueError('Incomplete pilot; refusing to report a full comparison')
    graphs = defaultdict(set)
    provenance = defaultdict(list)
    for row in baseline['relation_details']:
        fact = align(row['doc'], row['fact'])
        graphs[(row['doc'], row['size'], row['seed'])].add(fact)
        provenance[(row['doc'], fact)].append(row)
    pairs = []
    for row in baseline['pairs']:
        if row['kind'] != 'relations':
            continue
        a = graphs[(row['doc'], *row['a'])]
        b = graphs[(row['doc'], *row['b'])]
        pairs.append({**row, 'lexical_jaccard': row['jaccard'],
                      'intersection': len(a & b), 'union': len(a | b),
                      'jaccard': len(a & b) / len(a | b) if a | b else None})
    result = {
        'status': 'post-hoc partial alignment; not semantic validity or precision',
        'limitations': ['Rules inspected on these two regions; may over/under-merge.',
                       'All claims retained, including invalid claims; no endpoint repair.',
                       'Chunk-truncated names deliberately not restored.',
                       'No independent human annotations; diagnostic labels are purposive.'],
        'aliases': ALIASES, 'predicates': PREDICATES, 'pairs': pairs,
        'facts': [{'doc': doc, 'fact': fact, 'provenance': rows}
                  for (doc, fact), rows in sorted(provenance.items())],
    }
    (args.root / 'aligned_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# 부분 정렬 후 비교', '',
             '두 원문을 확인하여 작성한 사후 규칙이다. 유효성 평가나 일반화된 entity linking 결과가 아니다. 잘못된 관계도 남겨 두며, 부정·방향·잘린 이름을 임의로 교정하지 않는다.', '',
             '|문서|비교 (크기/seed)|문자열 Jaccard|부분 정렬 Jaccard|정렬 후 교집합/합집합|',
             '|---|---|---:|---:|---:|']
    for p in pairs:
        lines.append(f"|{p['doc']}|{p['a']} ↔ {p['b']}|{p['lexical_jaccard']:.3f}|{p['jaccard']:.3f}|{p['intersection']}/{p['union']}|")
    lines += ['', 'Winslow house → was constructed by → J.M. Winslow와 반대 방향의 constructed hotel 표현은 같은 사실로 정렬된다. 반면 person → constructed by → hotel처럼 잘못 추출된 관계는 올바른 사실로 고치지 않는다.', '',
              '정렬 후 차이에도 추출 오류와 미처 정렬하지 못한 표현 차이가 섞여 있다. 12개 진단 사례의 supported/unsupported는 AI의 원문 대조이며, 표본이 의도적으로 선택되어 precision을 계산할 수 없다.', '',
              '다음 확증 단계: 정렬 규칙을 고정하고 별도 문서에서 blind 이중 검토, seed 반복, 문장 경계 조건, 동일 추출 비용의 단일 크기 반복과 다중 크기를 비교한다.']
    (args.root / '정렬후_비교.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(pairs, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
