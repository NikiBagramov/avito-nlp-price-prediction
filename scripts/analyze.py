import json
import time
from common import ROOT,data,table,metrics,NUM,CAT
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bootstrap(d,preds,reference='tabular'):
    y=np.log(preds.price.to_numpy());groups=preds.group.to_numpy();rng=np.random.default_rng(2026)
    unique,inv=np.unique(groups,return_inverse=True);counts=np.bincount(inv);rows=[]
    base=np.abs(y-preds[reference].to_numpy())
    for name in preds.columns[3:]:
        diff=np.abs(y-preds[name].to_numpy())-base
        sums=np.bincount(inv,weights=diff);draw=rng.integers(0,len(unique),size=(1000,len(unique)))
        boot=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        rows.append({'method':name,'reference':reference,'delta_log_mae':float(diff.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'bootstrap_groups':len(unique)})
    return pd.DataFrame(rows)

def main():
    d=data();(ROOT/'artifacts/figures').mkdir(exist_ok=True)
    for scope in ['full','subset']:
        path=ROOT/'artifacts'/scope
        if not (path/'predictions.csv').exists():continue
        p=pd.read_csv(path/'predictions.csv',dtype={'listing_id':str,'group':str})
        bootstrap(d,p).to_csv(path/'bootstrap.csv',index=False)
        bootstrap(d,p,reference='tabular_rich').to_csv(path/'bootstrap_rich.csv',index=False)
        joined=p.merge(d[['listing_id','Район','text','Ремонт']],on='listing_id',validate='one_to_one')
        selected=json.loads((path/'run_manifest.json').read_text())['selected_by_dev']
        joined['ape']=np.abs(np.exp(joined[selected])-joined.price)/joined.price
        joined['log_error']=np.abs(joined[selected]-np.log(joined.price))
        joined['price_band']=pd.cut(joined.price,[0,1e7,2e7,5e7,np.inf],labels=['<10m','10-20m','20-50m','>=50m'])
        joined.groupby('price_band',observed=True).agg(n=('ape','size'),median_ape=('ape','median'),log_mae=('log_error','mean')).to_csv(path/'price_slices.csv')
        joined.groupby('Район').agg(n=('ape','size'),median_ape=('ape','median'),log_mae=('log_error','mean')).to_csv(path/'district_slices.csv')
        # No raw descriptions/addresses in the releasable error table.
        joined[['listing_id','price',selected,'ape','log_error','Район']].sort_values('log_error',ascending=False).head(20).to_csv(path/'largest_errors.csv',index=False)
        ms=pd.read_json(path/'metrics.json')
        fig,ax=plt.subplots(figsize=(9,4.5));ms=ms.sort_values('log_mae')
        ax.barh(ms.method,ms.log_mae,color='#3b6785');ax.invert_yaxis();ax.set_xlabel('Mean absolute log-price error (lower is better)');ax.set_title(f'{scope.title()} benchmark: address/template holdout');fig.tight_layout();fig.savefig(ROOT/f'artifacts/figures/{scope}_metrics.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.6,4.5));ax.scatter(np.log10(joined.price),joined[selected]/np.log(10),s=6,alpha=.25,color='#315f78');ax.plot([5.8,9.5],[5.8,9.5],color='#ad6246');ax.set(xlabel='Observed asking price (log10 RUB)',ylabel='Predicted price (log10 RUB)',title=f'{selected}: held-out properties');fig.tight_layout();fig.savefig(ROOT/f'artifacts/figures/{scope}_scatter.png',dpi=180);plt.close(fig)
    # Local explanation of the strong tabular control on a fixed random test sample.
    scope=ROOT/'artifacts/full';x=table(d,extra=True);te=d.split.eq('test');tr=d.split.eq('train');va=d.split.eq('dev');y=np.log(d.price.to_numpy())
    model=CatBoostRegressor();model.load_model(str(scope/'models/tabular_rich.cbm'))
    sample=x.loc[te].sample(n=min(512,int(te.sum())),random_state=42)
    shap=model.get_feature_importance(Pool(sample,cat_features=CAT+[c for c in x if c not in NUM+CAT]),type='ShapValues')
    importance=pd.DataFrame({'feature':x.columns,'mean_absolute_shap_log_price':np.abs(shap[:,:-1]).mean(axis=0)}).sort_values('mean_absolute_shap_log_price',ascending=False)
    importance.to_csv(scope/'shap_importance.csv',index=False)
    # Cyrillic labels are rendered with DejaVu Sans, not an incompatible default font.
    fig,ax=plt.subplots(figsize=(8,5));imp=importance.head(15).iloc[::-1];ax.barh(imp.feature,imp.mean_absolute_shap_log_price,color='#315f78');ax.set_xlabel('Mean |SHAP| in log-price units');ax.set_title('Predictive associations; not causal effects');fig.tight_layout();fig.savefig(ROOT/'artifacts/figures/factors.png',dpi=180);plt.close(fig)
    groups={'without_area':['Общая площадь','Площадь кухни','Жилая площадь'],
            'without_location':['Район','Metro1','Metro2','Metro3','Time1','Time2','Time3','city'],
            'without_condition':['Ремонт','Мебель','Отделка'],
            'without_building':['Год постройки','Тип дома','floor_current','floor_total','Высота потолков']}
    ablationpath=scope/'ablations.json'
    ablations=json.loads(ablationpath.read_text(encoding='utf-8')) if ablationpath.exists() else []
    for name,cols in groups.items():
        if any(r['method']==name for r in ablations):continue
        xx=x.drop(columns=cols);cats=[c for c in xx if xx[c].dtype=='object'];started=time.perf_counter()
        m=CatBoostRegressor(iterations=model.tree_count_,depth=model.get_param('depth'),learning_rate=.04,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        m.fit(xx.loc[tr],y[tr],cat_features=cats)
        ablations.append({'method':name,**metrics(y[te],m.predict(xx.loc[te])),'removed':cols,'seconds':time.perf_counter()-started})
        ablationpath.write_text(json.dumps(ablations,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,ablations[-1]['log_mae'],flush=True)
    (scope/'ablations.json').write_text(json.dumps(ablations,ensure_ascii=False,indent=2),encoding='utf-8')
    # LLM admission/coverage diagnostics and an explicitly silver renovation check.
    for suffix in ['', '_fewshot']:
        featurepath=ROOT/f'artifacts/llm_features{suffix}.jsonl'
        if not featurepath.exists():continue
        f=pd.read_json(featurepath,lines=True,dtype={'listing_id':str}).merge(d[['listing_id','split','Ремонт']],on='listing_id',validate='one_to_one')
        status={'rows':len(f),'parse_valid_rate':float(f.parse_valid.mean()),'all_evidence_valid_rate':float(f.evidence_valid.mean()),'truncated_rate':float(f.truncated.mean()),'by_split':{}}
        f['silver']=f['Ремонт'].map({'дизайнерский':'designer','евро':'renovated','косметический':'renovated','требует ремонта':'needs'})
        for split,sub in f.groupby('split'):
            evaluated=sub.loc[sub.silver.notna() & ~sub.renovation.isin(['unknown','unsupported','invalid'])]
            status['by_split'][split]={'n':len(sub),'original_label_n':int(sub.silver.notna().sum()),'comparable_n':len(evaluated),'agreement_on_comparable':float((evaluated.renovation==evaluated.silver).mean()) if len(evaluated) else None,'coverage':{c:float((~sub[c].isin(['unknown','invalid','unsupported'])).mean()) for c in ['renovation','furniture','water_view','urgency','legal']}}
        (ROOT/f'artifacts/llm_diagnostics{suffix}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('Analysis done',flush=True)
if __name__=='__main__':main()
