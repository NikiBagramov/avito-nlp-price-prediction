import time
from common import ROOT,data,table
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
d=data();tr=d.split.eq('train');x=table(d,extra=True).loc[tr];y=np.log(d.loc[tr,'price'].to_numpy())
path=ROOT/'artifacts/full/rich_oof.npy'
if not path.exists():
    base=CatBoostRegressor();base.load_model(str(ROOT/'artifacts/full/models/tabular_rich.cbm'))
    oof=np.zeros(len(x));started=time.perf_counter()
    for i,(fit,hold) in enumerate(GroupKFold(3).split(x,groups=d.loc[tr,'group'])):
        m=CatBoostRegressor(iterations=base.tree_count_,depth=base.get_param('depth'),learning_rate=.04,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        m.fit(x.iloc[fit],y[fit],cat_features=[c for c in x if x[c].dtype=='object']);oof[hold]=m.predict(x.iloc[hold])
        print('rich OOF fold',i+1,'seconds',time.perf_counter()-started,flush=True)
    np.save(path,oof)
