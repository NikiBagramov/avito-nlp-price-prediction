import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np
import pandas as pd

NUM = ['Общая площадь','Площадь кухни','Жилая площадь','floor_current','floor_total','rooms_count','Год постройки','Высота потолков','Time1','Time2','Time3']
CAT = ['Район','Metro1','Metro2','Metro3','city','Тип дома']
EXTRA = ['Ремонт','Мебель','Отделка','Окна','Балкон или лоджия','Санузел','Парковка','Срок сдачи','Способ продажи','Вид сделки','Название новостройки']
def data():
    return pd.read_csv(ROOT/'data/processed/prepared.csv', dtype={'listing_id':str,'group':str},low_memory=False)
def table(d,extra=False):
    x = pd.DataFrame(index=d.index)
    for c in NUM: x[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(-1)
    for c in CAT+(EXTRA if extra else []): x[c]=d[c].fillna('[missing]').astype(str)
    return x
def metrics(ylog, pred):
    from sklearn.metrics import r2_score
    ylog=np.asarray(ylog,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    y=np.exp(ylog); p=np.exp(np.clip(pred,0,30))
    return {'log_mae': float(np.mean(np.abs(ylog-pred))),
            'mae_rub': float(np.mean(np.abs(y-p))),
            'rmse_rub':float(np.sqrt(np.mean((y-p)**2))),
            'median_ape':float(np.median(np.abs(y-p)/y)),
            'r2':float(r2_score(y,p))}
def rule_features(texts):
    import re
    patterns = {
        'renovation_design': r'дизайнерск\w*\s+(?:ремонт|отделк)',
        'renovation_euro': r'евро\s*ремонт',
        'renovation_needed': r'(?:требу\w*|нуж\w*)\s+(?:косметическ\w*\s+|капитальн\w*\s+)?ремонт|без\s+отделк|под\s+ремонт',
        'furnished': r'мебел\w*\s+(?:и\s+техник\w*\s+)?оста\w*|с\s+мебелью|полностью\s+меблир',
        'water_view': r'вид\w*\s+(?:из\s+окон\s+)?на\s+(?:нев\w*|вод\w*|залив\w*|рек\w*|канал\w*|озер\w*)',
        'urgent': r'срочн\w*\s+прода|срочно\s*!|срочная',
        'legal_clear': r'без\s+обременени|один\s+(?:взрослый\s+)?собственник|1\s+(?:взрослый\s+)?собственник',
        'developer': r'от\s+застройщика|застройщик\w*\s+представ',
    }
    return pd.DataFrame({c:[int(bool(re.search(p,t))) for t in texts] for c,p in patterns.items()},index=texts.index)
