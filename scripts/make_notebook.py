from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
n=nbf.v4.new_notebook()
n.cells=[
nbf.v4.new_markdown_cell('''# Russian real-estate text benchmark

Команда: Никита Баграмов, Альфия Исхакова.

Это входная точка для **полного повторного прогона**. Результаты текущего прогона лежат в `artifacts/`, а не в выдуманных выводах ноутбука. Сначала включите GPU в Colab/Kaggle, распакуйте исходный архив проекта и установите рабочую папку в его корень (рядом с README.md).

Нужны интернет для публичных данных/весов, GPU от 6 GB, RAM примерно 16 GB и около 10 GB диска. Генеративное извлечение и эмбеддинги выполняются последовательно; полный прогон занимает существенно больше времени, чем чтение готового отчёта. Прочитайте DATA_CARD.md и PROTOCOL.md перед изменением данных или запросов.'''),
nbf.v4.new_code_cell('''from pathlib import Path
import os
root = Path.cwd()
assert (root / "scripts/run_pipeline.py").exists(), "Set the working directory to the extracted project root."
print(root)'''),
nbf.v4.new_code_cell('%pip install -r requirements.txt'),
nbf.v4.new_code_cell('''import torch
print(torch.__version__)
assert torch.cuda.is_available(), "Enable GPU before the full run."
print(torch.cuda.get_device_name())'''),
nbf.v4.new_markdown_cell('''## Full pipeline

The commands download the public source data and pinned open weights. No paid API key is needed. Keep the original split and do not tune prompts using test answers. To restart interrupted model inference, use the same files; to change the experiment, use a new clean project directory.'''),
nbf.v4.new_code_cell('''import subprocess, sys
subprocess.run([sys.executable, "scripts/run_pipeline.py"], check=True)'''),
nbf.v4.new_code_cell('''import pandas as pd, json
display(pd.read_json("artifacts/full/metrics.json")[["method","dev_log_mae","log_mae","mae_rub","median_ape"]])
display(pd.read_json("artifacts/subset/metrics.json")[["method","dev_log_mae","log_mae","mae_rub","median_ape"]])
print(json.load(open("artifacts/full/run_manifest.json"))["selected_by_dev"])'''),
nbf.v4.new_code_cell('''from IPython.display import Image, display
display(Image(filename="artifacts/figures/factors.png"))
display(Image(filename="artifacts/figures/full_metrics.png"))'''),
nbf.v4.new_markdown_cell('''## Before course submission

Publish the permitted source distribution, put the real repository URL into project.json, rebuild the PDF, and verify the author/contribution statement. Attach the PDF and paste the repository link into the course's solution field. A new split or a better number among these methods is not proof of global SotA or an awarded course bonus.''')]
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}}
nbf.write(n,root/'project.ipynb');nbf.validate(n)
print('Validated project.ipynb (entrypoint, not executed notebook outputs)')
