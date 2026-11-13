"""Independently recompute reported scores and check completeness of saved outputs."""
import hashlib
import json
from common import ROOT,data,metrics
import numpy as np
import pandas as pd
d=data();emb=np.load(ROOT/'cache/embeddings.npy',mmap_mode='r')
assert emb.shape==(len(d),1024)
assert np.isfinite(emb).all()
assert np.allclose(np.linalg.norm(emb,axis=1),1,atol=1e-4)
manifest=json.loads((ROOT/'artifacts/embedding_manifest.json').read_text())
assert manifest['data_ids_sha256']==hashlib.sha256('\n'.join(d.listing_id).encode()).hexdigest()
for scope in ['full','subset']:
    folder=ROOT/'artifacts'/scope
    rows=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    preds=pd.read_csv(folder/'predictions.csv',dtype={'listing_id':str,'group':str})
    expected=d.loc[d.split.eq('test') & (d.llm_subset if scope=='subset' else True)]
    assert set(preds.listing_id)==set(expected.listing_id)
    assert preds.listing_id.nunique()==len(preds)
    assert np.isfinite(preds.iloc[:,2:].to_numpy()).all()
    assert len({r['method'] for r in rows})==len(rows)
    for row in rows:
        computed=metrics(np.log(preds.price.to_numpy()),preds[row['method']].to_numpy())
        for k,value in computed.items():
            assert np.isclose(value,row[k],atol=1e-8,rtol=1e-7),(scope,row['method'],k,value,row[k])
    choice=json.loads((folder/'run_manifest.json').read_text())['selected_by_dev']
    assert choice==min(rows,key=lambda r:r['dev_log_mae'])['method']
    print(scope,'verified:',len(rows),'methods,',len(preds),'predictions per method; all metrics recomputed; selection uses dev.')
for suffix in ['', '_fewshot']:
    f=pd.read_json(ROOT/f'artifacts/llm_features{suffix}.jsonl',lines=True,dtype={'listing_id':str})
    assert len(f)==2100 and f.listing_id.nunique()==2100
    assert set(f.listing_id)==set(d.loc[d.llm_subset,'listing_id'])
print('Both LLM feature caches and all 30,775 normalized embeddings checked.')
