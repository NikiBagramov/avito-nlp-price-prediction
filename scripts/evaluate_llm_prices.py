import json
from common import ROOT,data,metrics
import numpy as np
import pandas as pd
def main():
    d=data();d=d.loc[d.llm_subset]
    raw=pd.read_json(ROOT/'artifacts/llm_prices.jsonl',lines=True,dtype={'listing_id':str})
    assert len(raw)==1400 and not raw.duplicated(['listing_id','shots']).any(), 'Expected 2 prompts for 700 dev/test examples'
    train_ids=set(d.loc[d.split.eq('train'),'listing_id'])
    assert all(set(ids)<=train_ids for ids in raw.retrieved_ids), 'Retrieval leakage'
    trainmedian=float(d.loc[d.split.eq('train'),'price'].median())
    raw['valid']=raw.prediction.notna() & raw.prediction.gt(0)
    raw['prediction_filled']=raw.prediction.where(raw.valid,trainmedian)
    joined=raw.merge(d[['listing_id','price','group']],on='listing_id',validate='many_to_one')
    results=[]
    for shots,frame in joined.groupby('shots'):
        dev=frame.loc[frame.split.eq('dev')];test=frame.loc[frame.split.eq('test')]
        results.append({'method':'llm_price_'+str(shots)+'shot',**metrics(np.log(test.price),np.log(test.prediction_filled)),
                        'dev_log_mae':metrics(np.log(dev.price),np.log(dev.prediction_filled))['log_mae'],
                        'invalid_test':int((~test.valid).sum()),'invalid_dev':int((~dev.valid).sum()),'fallback':'train price median for invalid outputs only'})
    near=joined.loc[joined.shots.eq(3)].copy()
    price_map=d.loc[d.split.eq('train')].set_index('listing_id').price.to_dict()
    near['knn_price']=near.retrieved_ids.map(lambda ids:float(np.mean([price_map[i] for i in ids])))
    dev=near.loc[near.split.eq('dev')];test=near.loc[near.split.eq('test')]
    results.append({'method':'knn3',**metrics(np.log(test.price),np.log(test.knn_price)),
                    'dev_log_mae':metrics(np.log(dev.price),np.log(dev.knn_price))['log_mae'],
                    'parameters':{'neighbors':3,'selection':'same training examples as 3-shot LLM','aggregation':'arithmetic mean price'}})
    path=ROOT/'artifacts/subset'
    records=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
    records=[r for r in records if not r['method'].startswith('llm_price_') and r['method']!='knn3']+results
    (path/'metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    p=pd.read_csv(path/'predictions.csv',dtype={'listing_id':str,'group':str})
    for shots in [0,3]:
        sub=joined.loc[joined.shots.eq(shots)&joined.split.eq('test')].set_index('listing_id')
        p['llm_price_'+str(shots)+'shot']=np.log(sub.reindex(p.listing_id).prediction_filled.to_numpy())
    p['knn3']=np.log(test.set_index('listing_id').reindex(p.listing_id).knn_price.to_numpy())
    assert p.notna().all().all()
    p.to_csv(path/'predictions.csv',index=False)
    manifest=json.loads((path/'run_manifest.json').read_text())
    manifest['selected_by_dev']=min(records,key=lambda r:r['dev_log_mae'])['method']
    (path/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
