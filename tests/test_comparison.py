import hashlib
import unittest

import compare_methods as c
import experiment as e


class ComparisonTests(unittest.TestCase):
    def test_known_word_coverage_and_empty_reference(self):
        self.assertEqual(c.word_coverage('кот спит дома', 'кот спит'), 2/3)
        self.assertEqual(c.word_coverage('Ёж спит', 'еж'), .5)
        self.assertIsNone(c.word_coverage('и в на', 'кот'))

    def test_evidence_cannot_inflate_word_coverage(self):
        text = c.triple_text([['кот', 'спит', 'дома', 'кот спит дома птица летит']])
        self.assertEqual(text, 'кот спит дома')
        self.assertEqual(c.word_coverage('птица летит', text), 0)

    def test_reconstructed_file_and_counts(self):
        output = c.OUT
        self.assertEqual(hashlib.sha256((output/'ngrams.jsonl').read_bytes()).hexdigest(), c.LEGACY_NGRAM_SHA256)
        stats = e.load_json(output/'statistics.json')
        docs = e.pd.read_csv(output/'per_document.csv')
        pilot = e.pd.read_csv(output/'human_word_coverage.csv')
        self.assertEqual(len(docs), 1000)
        self.assertEqual(len(pilot), 10)
        self.assertEqual(stats['ngram_units'], int(docs.ngram_units.sum()))
        self.assertEqual(stats['openie_relations'], int(docs.openie_units.sum()))
        self.assertAlmostEqual(stats['human_word_coverage']['ngram_mean'], pilot.ngram_word_coverage.mean())
        self.assertAlmostEqual(stats['human_word_coverage']['openie_mean'], pilot.openie_word_coverage.mean())
        self.assertTrue(((pilot.ngram_word_coverage >= 0) & (pilot.ngram_word_coverage <= 1)).all())

    def test_no_fabricated_quality_comparison(self):
        stats = e.load_json(c.OUT/'statistics.json')
        self.assertIsNone(stats['manual_relation_evaluation']['ngrams'])
        for group in ['pilot', 'audit']:
            self.assertEqual(stats['facet_metrics'][group]['difference'], 0)
        self.assertEqual(stats['additional_model_calls'], 0)
        self.assertEqual(c.bootstrap_difference([0, 0, 0]), [0, 0])


if __name__ == '__main__':
    unittest.main()
