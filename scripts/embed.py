import argparse
import hashlib
import json
import time
from common import ROOT,data
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args()
    torch.set_num_threads(4)
    d=data(); texts=d.text.tolist()
    fingerprint=hashlib.sha256(('\n'.join(d.listing_id)+'\n'+'\n'.join(texts)).encode()).hexdigest()
    dest=ROOT/'cache'; dest.mkdir(exist_ok=True)
    path=dest/'embeddings.npy'
    mmap=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(d),1024))
    progress=dest/'embedding_progress.json'
    saved=json.loads(progress.read_text()) if progress.exists() else {}
    if saved and saved.get('data_fingerprint')!=fingerprint:raise ValueError('Embedding cache belongs to different data; use a new experiment directory.')
    start=saved.get('rows',0)
    if start==len(d) and (ROOT/'artifacts/embedding_manifest.json').exists():
        print('All embeddings already exist; preserving original runtime manifest.',flush=True)
        return
    tokenizer=AutoTokenizer.from_pretrained(dest/'embedding_model',padding_side='left',local_files_only=True)
    model=AutoModel.from_pretrained(dest/'embedding_model',dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    t=time.perf_counter()
    for i in range(start,len(texts),a.batch_size):
        batch=texts[i:i+a.batch_size]
        enc=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            hidden=model(**enc).last_hidden_state[:,-1]
            emb=torch.nn.functional.normalize(hidden.float(),p=2,dim=1).cpu().numpy()
        mmap[i:i+len(batch)]=emb
        if i%256==0 or i+len(batch)==len(texts):
            mmap.flush();progress.write_text(json.dumps({'rows':i+len(batch),'seconds_this_session':time.perf_counter()-t,'data_fingerprint':fingerprint}))
            print(f'embeddings {i+len(batch)}/{len(texts)} elapsed {time.perf_counter()-t:.1f}s',flush=True)
    (ROOT/'artifacts/embedding_manifest.json').write_text(json.dumps({'rows':len(d),'dimension':1024,'max_length':512,'pooling':'last non-padding token (left padding)','normalization':'L2','data_ids_sha256':hashlib.sha256('\n'.join(d.listing_id).encode()).hexdigest(),'seconds_this_session':time.perf_counter()-t},indent=2))
if __name__=='__main__': main()
