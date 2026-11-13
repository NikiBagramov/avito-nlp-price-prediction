"""Checks target-leakage protection, group isolation and evidence admission."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare import clean_text,normalize,fingerprint,UnionFind
from llm import parse
from common import data,metrics
import numpy as np

class Invariants(unittest.TestCase):
    def test_currency_and_exact_price_removed_but_area_retained(self):
        text=clean_text('Цена 12 500 000 рублей. 43.3 м2, 2022 год. 12500000. 4 т. р. ипотека',12500000)
        self.assertNotIn('12 500 000',text)
        self.assertNotIn('12500000',text)
        self.assertIn('43.3',text)
        self.assertIn('2022',text)
    def test_homoglyph_and_template(self):
        self.assertEqual(normalize('Пpoдается'),normalize('Продается'))
        self.assertEqual(fingerprint('Квартира 25.4 м2 этаж 2'),fingerprint('Квартира 35.6 м2 этаж 8'))
    def test_transitive_components(self):
        u=UnionFind(3);u.union(0,1);u.union(1,2)
        self.assertEqual(u.find(0),u.find(2))
    def test_hallucinated_quote_rejected(self):
        raw='{"r":["designer","дизайнерский ремонт"],"f":["unknown",""],"v":["unknown",""],"u":["unknown",""],"l":["unknown",""]}'
        p=parse(raw,'квартира без отделки')
        self.assertTrue(p['parse_valid']);self.assertFalse(p['evidence_valid'])
        self.assertEqual(p['renovation'],'unsupported')
    def test_final_data_isolation(self):
        d=data()
        self.assertEqual(d.listing_id.nunique(),len(d))
        self.assertEqual(set(d.split),{'train','dev','test'})
        for c in ['group','address_key','template_key']:
            self.assertTrue((d.loc[d[c].notna() & d[c].ne('')].groupby(c).split.nunique()==1).all(),c)
        self.assertEqual(d.loc[d.llm_subset].split.value_counts().to_dict(),{'train':1400,'test':400,'dev':300})
    def test_metric_units(self):
        m=metrics(np.log([100.,200.]),np.log([110.,180.]))
        self.assertAlmostEqual(m['mae_rub'],15)
        self.assertAlmostEqual(m['median_ape'],.1)
    def test_fewshot_examples_are_training_only(self):
        d=data().set_index('listing_id')
        for key in ['7929104599','7866165689']:
            self.assertEqual(d.loc[key,'split'],'train')
            self.assertTrue(d.loc[key,'llm_subset'])

if __name__=='__main__':unittest.main()
