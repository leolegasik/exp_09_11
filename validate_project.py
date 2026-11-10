"""Итоговая проверка фактических результатов, а не только структуры кода."""
import json
import subprocess
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import nbformat
import experiment as e

root = Path(__file__).resolve().parent
docs, pred = e.corpus(), e.predictions()
assert len(docs) == 1000
assert len(pred) == 1000, f'Incomplete inference: {len(pred)}/1000'
assert {d['id'] for d in docs} == set(pred)
reasoning_tokens, generated_tokens = 0, 0
for doc in docs:
    result = pred[doc['id']]
    assert result['protocol_sha256'] == e.protocol_hash()
    assert result['text_sha256'] == doc['text_sha256']
    assert result['model'] == e.MODEL
    e.validate_prediction(result['prediction'], doc['text'])
    indices = result['evidence_fragment_ids']
    parts = e.evidence_fragments(doc['text'])
    assert len(indices) == len(result['prediction']['triples'])
    assert all(t[3] == parts[i] for i, t in zip(indices, result['prediction']['triples']))
    raw = e.load_json(root/f"results/raw_responses/{doc['id']}_{e.protocol_hash()[:12]}_0_{result['attempts']-1}.json")
    assert raw['choices'][0]['finish_reason'] == 'stop'
    reasoning_tokens += result['stats'].get('completion_tokens_details', {}).get('reasoning_tokens', 0)
    generated_tokens += result['stats'].get('completion_tokens', 0)
assert reasoning_tokens == 0, f'Unexpected reasoning tokens: {reasoning_tokens}'
stability = e.stability()
assert len(stability) == 10, 'Need 3 runs for all 10 pilot texts'
for doc in e.group_docs('pilot'):
    for run in range(3):
        e.validate_prediction(e.load_json(e.cache_path(doc, run))['prediction'], doc['text'])

notebook = nbformat.read(root/'experiment.ipynb', as_version=4)
nbformat.validate(notebook)
code_cells = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells), 'Notebook not fully executed'
assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']

test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                      cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace')
(root/'results/test_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
assert test.returncode == 0, test.stderr
test_count = int(re.search(r'Ran (\d+) tests?', test.stderr).group(1))

human = {}
for group in ['pilot', 'audit']:
    annotations = e.load_annotations(group)
    human[group] = len(annotations) if annotations else 0
    review = root/f'annotation/imported/review_{group}.json'
    if review.exists():
        e.validate_review(e.load_json(review), group)

summary = e.load_json(root/'results/summary.json')
assert summary['processed'] == 1000 and len(summary['stability']) == 10
result = {'validated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'machine_documents': len(pred), 'pilot_total_runs': 30, 'reasoning_tokens': reasoning_tokens,
          'completion_tokens_first_pass': generated_tokens, 'notebook_code_cells_executed': len(code_cells),
          'unit_tests': test_count, 'unit_test_status': 'passed', 'real_human_annotations': human,
          'pilot_relation_review': e.relation_metrics('pilot')['status'],
          'transfer_test': summary['transfer']['status'], 'protocol_sha256': e.protocol_hash(),
          'corpus_sha256': e.load_json(root/'data/manifest.json')['corpus_file_sha256'],
          'openie_documents': len(e.read_jsonl(root/'results/openie.jsonl')),
          'openie_triples': len(e.pd.read_csv(root/'results/openie_triples.csv')),
          'note': 'Validation checks technical integrity. Semantic correctness is measured by real human annotations, not guaranteed.'}
e.save_json(root/'results/validation.json', result)
print(json.dumps(result, ensure_ascii=True, indent=2))
