"""Exploratory incremental-text control using ALL available structured fields."""
import json
import time
import joblib
from common import ROOT,data,table,metrics
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
def main():
    d=data();masks={s:d.split.eq(s) for s in ['train','dev','test']};tr,va,te=[masks[s] for s in ['train','dev','test']]
    path=ROOT/'artifacts/full';y=np.log(d.price.to_numpy());x=table(d,extra=True)
    base=CatBoostRegressor();base.load_model(str(path/'models/tabular_rich.cbm'))
    devbase=base.predict(x.loc[va]);testbase=base.predict(x.loc[te]);residual=y[tr]-np.load(path/'rich_oof.npy')
    tfidf=joblib.load(path/'models/tfidf.joblib')['vectorizer']
    emb=np.load(ROOT/'cache/embeddings.npy',mmap_mode='r')
    representations={'tfidf':[tfidf.transform(d.loc[mask,'text']) for mask in [tr,va,te]],'embedding':[emb[mask] for mask in [tr,va,te]]}
    records=json.loads((path/'metrics.json').read_text(encoding='utf-8'));pred=pd.read_csv(path/'predictions.csv',dtype={'listing_id':str,'group':str})
    for kind,(ztr,zva,zte) in representations.items():
        started=time.perf_counter();candidates=[]
        for alpha in [1,10,100]:
            m=Ridge(alpha=alpha,solver='lsqr').fit(ztr,residual)
            candidates.append((np.mean(np.abs(y[va]-devbase-m.predict(zva))),alpha,m))
        _,alpha,m=min(candidates,key=lambda z:z[0]);dp=m.predict(zva);tp=m.predict(zte)
        weight=min(np.linspace(0,1,21),key=lambda w:np.mean(np.abs(y[va]-devbase-w*dp)))
        name='rich_'+kind+'_residual';p=testbase+weight*tp
        row={'method':name,**metrics(y[te],p),'dev_log_mae':float(np.mean(np.abs(y[va]-devbase-weight*dp))),'seconds':time.perf_counter()-started,'parameters':{'alpha':alpha,'residual_weight':float(weight),'crossfit_folds':3,'selection':'dev only'}}
        records=[r for r in records if r['method']!=name]+[row];pred[name]=p
        joblib.dump({'regressor':m,'weight':weight},path/'models'/f'{name}.joblib');print(json.dumps(row),flush=True)
    pred.to_csv(path/'predictions.csv',index=False);(path/'metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    manifest=json.loads((path/'run_manifest.json').read_text());manifest['selected_by_dev']=min(records,key=lambda r:r['dev_log_mae'])['method'];(path/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
