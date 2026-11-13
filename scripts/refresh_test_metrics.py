"""Make released float64 CSV predictions the authoritative source for test scores."""
import json
from common import ROOT,metrics
import numpy as np
import pandas as pd
for scope in ['full','subset']:
    path=ROOT/'artifacts'/scope
    p=pd.read_csv(path/'predictions.csv',dtype={'listing_id':str,'group':str})
    records=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
    for r in records:r.update(metrics(np.log(p.price.to_numpy()),p[r['method']].to_numpy()))
    (path/'metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    manifest=json.loads((path/'run_manifest.json').read_text())
    manifest['test_metrics_authority']='float64 recomputation from released predictions.csv; dev selection unchanged'
    (path/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Test scores recomputed from the released CSV values; no retraining or model selection.')
