"""Read-only audit, deterministic preparation and address/template-disjoint splits."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
HOMOGLYPHS = str.maketrans(dict(zip('ABCEHKMOPTXYabcehkmoptyx', 'АВСЕНКМОРТХУавсенкмортух')))
def normalize(s):
    s = unicodedata.normalize('NFKC', str(s)).replace('\xa0', ' ')
    # Transliterate lookalikes only inside tokens that contain Cyrillic.
    s = re.sub(r'[A-Za-zА-Яа-яЁё]+', lambda m: m[0].translate(HOMOGLYPHS) if re.search('[А-Яа-яЁё]', m[0]) else m[0], s)
    return re.sub(r'\s+', ' ', s).lower().replace('ё', 'е').strip()

def clean_text(s, price):
    s = normalize(s)
    # Remove currency-denominated amounts incl. rent, mortgage and price per sqm.
    s = re.sub(r'\d[\d\s.,]*\s*(?:млн\.?|миллион\w*|тыс\.?|т\s*\.\s*р\.?|руб\w*|₽)', ' [money] ', s)
    if np.isfinite(price):
        digits = str(int(price))
        s = re.sub(r'(?<!\d)' + r'[\s.,]*'.join(digits) + r'(?!\d)', ' [price] ', s)
    s = re.sub(r'(?:\+7|8)[\s()\-]*\d(?:[\s()\-]*\d){9}', ' [phone] ', s)
    return re.sub(r'\s+', ' ', s).strip()

def fingerprint(text):
    # Removing numbers and spaces detects repeated developer templates across units.
    return hashlib.sha256(re.sub(r'[^а-яa-z]', '', text).encode()).hexdigest()

class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, x, y): self.p[self.find(x)] = self.find(y)

def main():
    d = pd.read_csv(ROOT / 'data/raw/listings.csv', low_memory=False)
    audit = {'source_rows': len(d), 'source_columns': len(d.columns),
             'missing_fraction': d.isna().mean().round(6).to_dict(),
             'price_quantiles': d.price.quantile([0,.01,.5,.99,1]).to_dict()}
    d['listing_id'] = d.url.str.extract(r'_(\d+)(?:\?|$)', expand=False)
    audit['missing_listing_id'] = int(d.listing_id.isna().sum())
    audit['duplicate_id_rows'] = int(d.listing_id.duplicated().sum())
    # No percentile trim: keep the expensive tail and expose its error separately.
    valid = d.listing_id.notna() & d.price.gt(0) & d['Общая площадь'].gt(0) & d.description.fillna('').str.len().ge(80)
    audit['excluded_missing_invalid_or_short'] = int((~valid).sum())
    d = d.loc[valid].drop_duplicates('listing_id').reset_index(drop=True)
    d['text_raw'] = d.title.fillna('') + '. ' + d.description.fillna('')
    d['text'] = [clean_text(t, p) for t,p in zip(d.text_raw, d.price)]
    d['address_key'] = d.clean_address.fillna('').map(normalize).str.replace(r'[^а-яa-z0-9]', '', regex=True)
    d['template_key'] = d.text.map(fingerprint)
    uf = UnionFind(len(d))
    for column in ['address_key','template_key']:
        seen = {}
        for i,key in enumerate(d[column]):
            if key:
                if key in seen: uf.union(i,seen[key])
                else: seen[key] = i
    d['group'] = [str(uf.find(i)) for i in range(len(d))]
    # Stable group labels based on the minimum listing ID, independent of row order.
    groupnames = d.groupby('group').listing_id.min()
    d['group'] = d.group.map(groupnames)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=42)
    train_dev, test = next(splitter.split(d, groups=d.group))
    sub = d.iloc[train_dev]
    tr,va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=43).split(sub, groups=sub.group))
    d['split'] = 'train'
    d.loc[train_dev[va], 'split'] = 'dev'
    d.loc[test,'split'] = 'test'
    # Fixed, uniform row sample per split for expensive generation. No target stratification.
    d['llm_subset'] = False
    for split,n in [('train',1400),('dev',300),('test',400)]:
        idx = d.loc[d.split.eq(split)].sample(n=n, random_state=2026).index
        d.loc[idx,'llm_subset'] = True
    audit.update({'prepared_rows': len(d), 'groups': d.group.nunique(),
                  'largest_groups': d.groupby('group').size().nlargest(10).to_dict(),
                  'splits': d.split.value_counts().to_dict(),
                  'text_char_quantiles': d.text.str.len().quantile([0,.5,.9,.99,1]).to_dict(),
                  'modified_texts': int(d.text.str.contains(r'\[money\]|\[price\]',regex=True).sum()),
                  'duplicate_templates': int(d.template_key.duplicated().sum())})
    for col in ['group','address_key','template_key']:
        overlap = d.loc[d[col].ne('')].groupby(col).split.nunique().max()
        assert overlap == 1, f'Cross-split {col} leakage'
    dest = ROOT / 'data/processed'
    dest.mkdir(parents=True,exist_ok=True)
    d.to_csv(dest / 'prepared.csv',index=False)
    d[['listing_id','group','split','llm_subset']].to_csv(ROOT / 'artifacts/splits.csv',index=False)
    (ROOT / 'artifacts/audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='missing_fraction'},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
