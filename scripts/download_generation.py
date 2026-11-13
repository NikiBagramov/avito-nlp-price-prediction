from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1]
snapshot_download('Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                  local_dir=root/'cache/generation_model',max_workers=2,
                  allow_patterns=['*.json','*.safetensors','*.txt','README.md','LICENSE'])
