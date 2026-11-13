"""Train on frozen splits, choose variants on dev, evaluate once on test."""
import argparse
import json
import time
import platform
import joblib
from common import ROOT,NUM,CAT,EXTRA,data,table,metrics,rule_features
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subset',action='store_true');ap.add_argument('--embeddings',action='store_true');ap.add_argument('--llm',action='store_true');a=ap.parse_args()
    all_d=data(); d=all_d.loc[all_d.llm_subset].copy() if a.subset else all_d.copy()
    dest=ROOT/'artifacts'/('subset' if a.subset else 'full'); dest.mkdir(parents=True,exist_ok=True)
    prior_records=json.loads((dest/'metrics.json').read_text(encoding='utf-8')) if (dest/'metrics.json').exists() else []
    masks={s:d.split.eq(s).to_numpy() for s in ['train','dev','test']}; tr,va,te=[masks[s] for s in ['train','dev','test']]
    y=np.log(d.price.to_numpy()); records=[]; predictions={}; train_predictions={}; dev_predictions={}
    def save(name,p,dev,seconds,params):
        p=np.asarray(p,dtype=np.float64);dev=np.asarray(dev,dtype=np.float64)
        row={'method':name,**metrics(y[te],p),'dev_log_mae':metrics(y[va],dev)['log_mae'],'seconds':seconds,'parameters':params}
        records.append(row); predictions[name]=p; dev_predictions[name]=dev
        print(json.dumps(row,ensure_ascii=False),flush=True)
        pd.DataFrame(records).to_json(dest/'metrics.json',orient='records',force_ascii=False,indent=2)
        pd.DataFrame({'listing_id':d.loc[te,'listing_id'].to_numpy(),'group':d.loc[te,'group'].to_numpy(),'price':d.loc[te,'price'].to_numpy(),**predictions}).to_csv(dest/'predictions.csv',index=False)
    t=time.perf_counter(); median=float(np.median(y[tr])); save('median',np.full(te.sum(),median),np.full(va.sum(),median),time.perf_counter()-t,{})
    def cb(name,x,seeds=(42,),iterations=1500):
        cat=[c for c in x if x[c].dtype=='object']; candidates=[]; started=time.perf_counter()
        modelpath=dest/'models'/f'{name}.cbm'
        if modelpath.exists():
            old=next((r for r in prior_records if r['method']==name),None)
            if old is not None:
                m=CatBoostRegressor();m.load_model(str(modelpath))
                train_predictions[name]=m.predict(x.loc[tr])
                save(name,m.predict(x.loc[te]),m.predict(x.loc[va]),old['seconds'],old['parameters'])
                return m
        for depth in [6,8]:
            m=CatBoostRegressor(iterations=iterations,depth=depth,learning_rate=.04,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
            m.fit(x.loc[tr],y[tr],cat_features=cat,eval_set=(x.loc[va],y[va]),early_stopping_rounds=100)
            score=metrics(y[va],m.predict(x.loc[va]))['log_mae']; candidates.append((score,depth,m))
        score,depth,model=min(candidates,key=lambda z:z[0]); ntrees=model.tree_count_
        train_predictions[name]=model.predict(x.loc[tr])
        p=model.predict(x.loc[te]);dev=model.predict(x.loc[va]); variants=[p]
        for seed in seeds[1:]:
            ms=CatBoostRegressor(iterations=ntrees,depth=depth,learning_rate=.04,loss_function='RMSE',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
            ms.fit(x.loc[tr],y[tr],cat_features=cat)
            variants.append(ms.predict(x.loc[te]))
        (dest/'models').mkdir(exist_ok=True)
        model.save_model(str(dest/'models'/f'{name}.cbm'))
        np.save(dest/f'{name}_seed_predictions.npy',np.array(variants))
        pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(dest/f'{name}_importance.csv',index=False)
        save(name,p,dev,time.perf_counter()-started,{'depth':depth,'iterations':ntrees,'seeds':list(seeds),'learning_rate':.04,'selection':'dev log MAE','seed_test_log_mae':[metrics(y[te],q)['log_mae'] for q in variants]})
        return model
    x=table(d); base=cb('tabular',x,seeds=(42,43,44))
    cb('tabular_rich',table(d,extra=True),seeds=(42,43,44))
    rules=rule_features(d.text); cb('tabular_rules',pd.concat([x,rules],axis=1))
    t=time.perf_counter()
    vector=TfidfVectorizer(max_features=40000,ngram_range=(1,2),min_df=3,sublinear_tf=True)
    ztr=vector.fit_transform(d.loc[tr,'text']);zva=vector.transform(d.loc[va,'text']);zte=vector.transform(d.loc[te,'text'])
    models=[]
    for alpha in [1,10,100]:
        m=Ridge(alpha=alpha,solver='lsqr');m.fit(ztr,y[tr]);models.append((metrics(y[va],m.predict(zva))['log_mae'],alpha,m))
    _,alpha,m=min(models,key=lambda v:v[0]); p=m.predict(zte);dev=m.predict(zva)
    joblib.dump({'vectorizer':vector,'regressor':m},dest/'models/tfidf.joblib')
    save('tfidf',p,dev,time.perf_counter()-t,{'alpha':alpha,'vocabulary_size':len(vector.vocabulary_),'ngrams':[1,2]})
    # Convex blend selected only on dev, no target leakage into features.
    weights=np.linspace(0,1,21)
    w=min(weights,key=lambda w:np.mean(np.abs(y[va]-((1-w)*dev_predictions['tabular']+w*dev))))
    save('tabular_tfidf_blend',(1-w)*predictions['tabular']+w*p,(1-w)*dev_predictions['tabular']+w*dev,time.perf_counter()-t,{'text_weight':float(w),'selection':'dev only convex blend'})
    # Independently fit train residuals from group-out-of-fold predictions.
    t=time.perf_counter();oof=np.zeros(tr.sum());xt=x.loc[tr];yt=y[tr]
    for fit,hold in GroupKFold(3).split(xt,groups=d.loc[tr,'group']):
        fold=CatBoostRegressor(iterations=base.tree_count_,depth=base.get_param('depth'),learning_rate=.04,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        fold.fit(xt.iloc[fit],yt[fit],cat_features=CAT);oof[hold]=fold.predict(xt.iloc[hold])
    np.save(dest/'oof.npy',oof)
    residuals=yt-oof; candidates=[]
    for alpha in [1,10,100]:
        mres=Ridge(alpha=alpha,solver='lsqr').fit(ztr,residuals)
        candidates.append((np.mean(np.abs(y[va]-dev_predictions['tabular']-mres.predict(zva))),alpha,mres))
    _,alpha,mres=min(candidates,key=lambda v:v[0])
    joblib.dump({'vectorizer':vector,'regressor':mres},dest/'models/tfidf_residual.joblib')
    save('tabular_tfidf_residual',predictions['tabular']+mres.predict(zte),dev_predictions['tabular']+mres.predict(zva),time.perf_counter()-t,{'alpha':alpha,'crossfit_folds':3,'vocabulary_fit':'train only'})
    # Export high-magnitude residual vocabulary associations.
    words=np.array(vector.get_feature_names_out());idx=np.argsort(np.abs(mres.coef_))[-40:][::-1]
    pd.DataFrame({'term':words[idx],'coefficient_log_price':mres.coef_[idx]}).to_csv(dest/'text_associations.csv',index=False)
    if a.embeddings:
        emb=np.load(ROOT/'cache/embeddings.npy',mmap_mode='r')[d.index]
        t=time.perf_counter();candidates=[]
        for alpha in [1,10,100]:
            er=Ridge(alpha=alpha).fit(emb[tr],y[tr]);candidates.append((np.mean(np.abs(y[va]-er.predict(emb[va]))),alpha,er))
        _,alpha,er=min(candidates,key=lambda v:v[0]);save('embedding_ridge',er.predict(emb[te]),er.predict(emb[va]),time.perf_counter()-t,{'alpha':alpha})
        t=time.perf_counter();candidates=[]
        for alpha in [1,10,100]:
            er=Ridge(alpha=alpha).fit(emb[tr],residuals);candidates.append((np.mean(np.abs(y[va]-dev_predictions['tabular']-er.predict(emb[va]))),alpha,er))
        _,alpha,er=min(candidates,key=lambda v:v[0]);save('tabular_embedding_residual',predictions['tabular']+er.predict(emb[te]),dev_predictions['tabular']+er.predict(emb[va]),time.perf_counter()-t,{'alpha':alpha,'crossfit_folds':3})
        joblib.dump(er,dest/'models/embedding_residual.joblib')
    if a.llm:
        for filename,name in [('llm_features.jsonl','tabular_llm'),('llm_features_fewshot.jsonl','tabular_llm_fewshot')]:
            fp=ROOT/'artifacts'/filename
            if not fp.exists():continue
            f=pd.read_json(fp,lines=True,dtype={'listing_id':str}).set_index('listing_id')
            joined=f.reindex(d.listing_id)
            if joined.index.duplicated().any() or joined['raw'].isna().any():raise ValueError('Incomplete LLM features')
            llmx=x.copy()
            for c in ['renovation','furniture','water_view','urgency','legal']:
                llmx['llm_'+c]=joined[c].fillna('invalid').to_numpy()
            cb(name,llmx,seeds=(42,43,44))
    # Freeze the ranking selector from dev; no method choice based on test metrics.
    selected=min(records,key=lambda r:r['dev_log_mae'])['method']
    meta={'selected_by_dev':selected,'rows':len(d),'splits':{s:int(m.sum()) for s,m in masks.items()},'platform':platform.platform(),'python':platform.python_version()}
    (dest/'run_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('DONE',json.dumps(meta),flush=True)

if __name__=='__main__': main()
