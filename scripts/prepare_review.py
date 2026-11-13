"""Prepare a blind, optional human validation sample, not fabricated gold labels."""
import json
from common import ROOT,data
d=data();sample=d.loc[d.llm_subset & d.split.eq('test')].sample(100,random_state=731)
dest=ROOT/'data/annotations';dest.mkdir(parents=True,exist_ok=True)
records=[{'listing_id':str(r.listing_id),'text':r.text} for r in sample.itertuples()]
(dest/'review_100.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'artifacts/review_sample_ids.json').write_text(json.dumps(sample.listing_id.tolist(),indent=2))
print('100 blind test texts prepared locally; no labels, prices, source renovation fields or model predictions are shown.')
