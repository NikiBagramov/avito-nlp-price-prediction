from pathlib import Path
import json
import requests
from huggingface_hub import snapshot_download
root = Path(__file__).resolve().parents[1]
repo = 'Qwen/Qwen3-Embedding-0.6B'
metadata = requests.get('https://huggingface.co/api/models/'+repo, timeout=60)
metadata.raise_for_status()
revision = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
snapshot_download(repo, revision=revision, local_dir=root/'cache/embedding_model',
                  allow_patterns=['*.json','*.safetensors','*.txt','README.md','LICENSE'], max_workers=2)
info = {'repo':repo,'revision':revision,'license':'Apache-2.0',
        'generation_model': 'Qwen/Qwen3-1.7B',
        'generation_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'}
(root/'artifacts/model_provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info))
