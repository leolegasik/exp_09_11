import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import experiment as e
from forms import js_json


class DataIntegrityTests(unittest.TestCase):
    def test_dataset_partition_and_provenance(self):
        docs = e.corpus()
        manifest = e.load_json(e.ROOT/'data/manifest.json')
        self.assertEqual(len(docs), 1000)
        self.assertEqual(len({d['id'] for d in docs}), 1000)
        self.assertEqual(len({d['text_sha256'] for d in docs}), 1000)
        self.assertEqual(sum(d['split'] == 'pilot' for d in docs), 10)
        self.assertEqual(sum(d['split'] == 'bulk' for d in docs), 990)
        self.assertEqual(sum(d['audit'] for d in docs), 20)
        self.assertTrue(all(d['split'] == 'bulk' for d in docs if d['audit']))
        self.assertTrue(all(150 <= len(d['text']) <= 650 for d in docs))
        self.assertTrue(all(d['source_field'] == 'summary' and d['url'].startswith('https://www.gazeta.ru/') for d in docs))
        self.assertEqual(hashlib.sha256((e.ROOT/'data/processed/corpus.jsonl').read_bytes()).hexdigest(), manifest['corpus_file_sha256'])
        self.assertTrue(all(e.sha(d['text'].lower().replace('ё', 'е')) == d['text_sha256'] for d in docs))

    def test_fragments_cannot_invent_evidence(self):
        for doc in e.corpus():
            parts = e.evidence_fragments(doc['text'])
            self.assertEqual(parts[0], doc['text'])
            self.assertTrue(all(p in doc['text'] for p in parts))

    def test_blind_form_contains_no_predictions(self):
        for group in ['01_pilot', '02_audit']:
            text = (e.ROOT/f'annotation/{group}_blind.html').read_text(encoding='utf-8')
            self.assertNotIn('const pred=', text)
            self.assertNotIn('prediction', text)
            self.assertNotIn('stats', text)
            self.assertNotIn('<script src=', text)

    def test_dictionary_baseline_has_only_labels(self):
        docs = {d['id']: d for d in e.corpus()}
        base = e.read_jsonl(e.ROOT/'results/baselines.jsonl')
        self.assertEqual({x['id'] for x in base}, set(docs))
        for item in base:
            self.assertEqual(set(item), {'id', 'labels'})
            self.assertEqual(item['labels'], e.lexical_labels(docs[item['id']]['text']))

    def test_openie_export_matches_evaluated_relations(self):
        docs, pred = e.corpus(), e.predictions()
        output = e.read_jsonl(e.ROOT/'results/openie.jsonl')
        self.assertEqual({x['id'] for x in output}, {d['id'] for d in docs})
        by_id = {d['id']: d for d in docs}
        for item in output:
            expected = e.openie_document(by_id[item['id']], pred[item['id']])
            self.assertEqual(item, expected)
        csv = e.pd.read_csv(e.ROOT/'results/openie_triples.csv')
        self.assertEqual(len(csv), sum(len(x['triples']) for x in output))
        for item in csv.to_dict('records'):
            triple = output[[x['id'] for x in output].index(item['id'])]['triples'][item['triple_index']-1]
            self.assertEqual({k: item[k] for k in triple}, triple)

    def test_openie_rejects_stale_protocol_or_different_text(self):
        doc = e.corpus()[0]
        record = e.predictions()[doc['id']]
        for key in ['id', 'text_sha256', 'protocol_sha256']:
            bad = copy.deepcopy(record); bad[key] = 'wrong'
            with self.assertRaises(ValueError):
                e.openie_document(doc, bad)

    def test_openie_export_never_calls_model(self):
        doc = e.corpus()[0]
        record = e.predictions()[doc['id']]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(e, 'ROOT', Path(directory)), patch.object(e, 'corpus', return_value=[doc]), \
                 patch.object(e, 'extract', side_effect=AssertionError('Unexpected model call')):
                result = e.export_openie({doc['id']: record})
                self.assertEqual(result['documents'], 1)
                self.assertEqual(result['additional_model_calls'], 0)
                result = e.export_openie({})
                self.assertEqual(result['documents'], 0)
                self.assertEqual(result['missing_documents'], 1)
                self.assertEqual(len(e.pd.read_csv(e.ROOT/'results/openie_triples.csv')), 0)


class MetricTests(unittest.TestCase):
    def test_known_confusion(self):
        y = np.array([[1, 0], [1, 0]])
        p = np.array([[1, 1], [0, 0]])
        m = e.score_arrays(y, p)
        self.assertAlmostEqual(m['micro_f1'], .5)
        self.assertAlmostEqual(m['precision_micro'], .5)
        self.assertAlmostEqual(m['recall_micro'], .5)
        self.assertAlmostEqual(m['hamming_error'], .5)
        self.assertAlmostEqual(m['exact_match'], 0)

    def test_perfect_bootstrap(self):
        y = np.ones((10, 6), dtype=int)
        self.assertEqual(e.bootstrap_f1(y, y, n=50), [1., 1.])

    def test_empty_positive_convention(self):
        y = np.zeros((3, 6), dtype=int)
        m = e.score_arrays(y, y)
        self.assertEqual(m['micro_f1'], 0)
        self.assertEqual(m['exact_match'], 1)

    def test_bootstrap_is_reproducible(self):
        y = np.array([[1, 0], [0, 1], [0, 0]])
        p = np.array([[1, 1], [0, 1], [1, 0]])
        self.assertEqual(e.bootstrap_f1(y, p, 30), e.bootstrap_f1(y, p, 30))

    def test_paired_bootstrap_identical_methods(self):
        y = np.array([[1, 0], [0, 1], [1, 1]])
        p = np.array([[1, 1], [0, 1], [1, 0]])
        result = e.paired_bootstrap_difference(y, p, p, n=50)
        self.assertEqual(result['qwen_minus_lexical_micro_f1'], 0)
        self.assertEqual(result['paired_bootstrap95'], [0., 0.])

    def test_transfer_counts_and_holm_correction(self):
        h1 = {'p1': {'labels': ['cause']}, 'p2': {'labels': []}}
        h2 = {'a1': {'labels': ['cause']}, 'a2': {'labels': []}, 'a3': {'labels': []}}
        pred = {k: {'prediction': {'labels': []}} for k in list(h1)+list(h2)}
        pred['p2']['prediction']['labels'] = ['change']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'results').mkdir()
            with patch.object(e, 'ROOT', root), patch.object(e, 'load_annotations', side_effect=[h1, h2]), patch.object(e, 'predictions', return_value=pred):
                result = e.transfer_test()
        cause_fn = next(r for r in result['rows'] if r['label'] == 'cause' and r['direction'] == 'FN')
        self.assertEqual((cause_fn['pilot_count'], cause_fn['audit_count']), (1, 1))
        self.assertTrue(cause_fn['same_direction_observed'])
        self.assertEqual(len(result['rows']), 12)
        self.assertTrue(all(r['holm_p'] >= r['fisher_p_exploratory'] for r in result['rows']))
        self.assertTrue(all(0 <= r['holm_p'] <= 1 for r in result['rows']))


class ValidationTests(unittest.TestCase):
    def test_fresh_request_bypasses_cache_and_does_not_save(self):
        doc = e.corpus()[0]
        raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'triples': [{'s': 'А', 'r': 'связано с', 'o': 'Б', 'e': 0}], 'labels': []})}}]}
        with patch.object(e.requests, 'post') as post, patch.object(e, 'load_json', side_effect=AssertionError('Cache read')), \
             patch.object(e, 'save_json', side_effect=AssertionError('Write')):
            post.return_value.json.return_value = raw
            result = e.extract(doc, use_cache=False, save=False)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result['prediction']['triples'][0][3], doc['text'])

    def test_invalid_or_fabricated_quote_rejected(self):
        good = {'triples': [['Кот', 'спит на', 'ковре', 'Кот спит на ковре.']], 'labels': []}
        self.assertEqual(e.validate_prediction(good, 'Кот спит на ковре.'), good)
        bad = copy.deepcopy(good); bad['triples'][0][3] = 'Кот играет.'
        with self.assertRaises(ValueError):
            e.validate_prediction(bad, 'Кот спит на ковре.')

    def test_duplicate_or_unknown_labels_rejected(self):
        for labels in [['cause', 'cause'], ['fabricated']]:
            with self.assertRaises(ValueError):
                e.validate_prediction({'triples': [], 'labels': labels}, 'Тест')

    def test_protocol_change_invalidates_cache(self):
        doc = e.corpus()[0]
        old = e.cache_path(doc)
        with patch.object(e, 'MODEL', 'different-model'):
            self.assertNotEqual(e.cache_path(doc), old)
        with patch.object(e, 'PROMPT', e.PROMPT+' изменён'):
            self.assertNotEqual(e.cache_path(doc), old)
        self.assertNotEqual(e.cache_path(doc, 1), old)

    def test_import_requires_real_complete_answers(self):
        docs = e.group_docs('pilot')
        data = {'group': 'pilot', 'corpus_sha256': e.load_json(e.ROOT/'data/manifest.json')['corpus_file_sha256'],
                'answers': [{'id':d['id'], 'labels': [], 'triples': [['А', 'связано с', 'Б']], 'complete': True} for d in docs]}
        e.validate_annotations(data, 'pilot')
        incomplete = copy.deepcopy(data); incomplete['answers'][0]['complete'] = False
        with self.assertRaises(ValueError):
            e.validate_annotations(incomplete, 'pilot')
        duplicate = copy.deepcopy(data); duplicate['answers'][1] = copy.deepcopy(duplicate['answers'][0])
        with self.assertRaises(ValueError):
            e.validate_annotations(duplicate, 'pilot')
        wrong = copy.deepcopy(data); wrong['corpus_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            e.validate_annotations(wrong, 'pilot')

    def test_partial_audit_excludes_unanswered_documents(self):
        doc = e.group_docs('audit')[0]
        data = {'group':'audit', 'partial':True,
                'corpus_sha256':e.load_json(e.ROOT/'data/manifest.json')['corpus_file_sha256'],
                'answers':[{'id':doc['id'], 'labels':[], 'triples':[['А','связано с','Б']], 'complete':True}]}
        e.validate_annotations(data, 'audit')
        complete_required = copy.deepcopy(data); complete_required.pop('partial')
        with self.assertRaises(ValueError):
            e.validate_annotations(complete_required, 'audit')
        unanswered = copy.deepcopy(data); unanswered['answers'][0]['complete'] = False
        with self.assertRaises(ValueError):
            e.validate_annotations(unanswered, 'audit')
        empty = copy.deepcopy(data); empty['answers'] = []
        with self.assertRaises(ValueError):
            e.validate_annotations(empty, 'audit')

    def test_one_to_one_semantic_review(self):
        human = {'doc': {'id':'doc','triples':[['А','делает','Б']]}}
        pred = {'doc': {'prediction': {'triples': [['А','делает','Б','цитата'], ['А','делает','Б','цитата']]}}}
        review = {'group':'review_pilot','protocol_sha256':e.protocol_hash(),'answers':[{'id':'doc','links':[
            {'supported':True,'human_index':0}, {'supported':True,'human_index':0}]}]}
        with patch.object(e, 'load_annotations', return_value=human), patch.object(e, 'predictions', return_value=pred):
            with self.assertRaises(ValueError):
                e.validate_review(review, 'pilot')
            review['answers'][0]['links'][1]['human_index'] = -1
            e.validate_review(review, 'pilot')

    def test_no_human_metrics_without_annotations(self):
        with patch.object(e, 'load_annotations', return_value=None):
            self.assertEqual(e.evaluate('pilot')['status'], 'awaiting_human')
            self.assertEqual(e.transfer_test()['status'], 'awaiting_human')

    def test_html_embedded_json_does_not_close_script(self):
        escaped = js_json({'x':'</script><script>alert(1)</script>'})
        self.assertNotIn('</script>', escaped)
        self.assertEqual(json.loads(escaped)['x'], '</script><script>alert(1)</script>')

    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'a.json'
            e.save_json(path, {'тест': [1, 2]})
            self.assertEqual(e.load_json(path), {'тест': [1, 2]})
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
