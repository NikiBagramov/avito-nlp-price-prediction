# Текст объявлений и прогноз цены квартиры

[Репозиторий](https://github.com/NikiBagramov/avito-nlp-price-prediction) · [PDF-отчёт](output/pdf/avito_nlp_report.pdf) · [Ноутбук запуска](project.ipynb)

**Вопрос:** помогают ли русскоязычные описания квартир прогнозировать запрашиваемую цену сверх площади, района и других табличных характеристик?

Команда: Никита Баграмов и Альфия Исхакова. Учебный проект NLP & LLMs, осень 2026. Подготовка кода, прогонов и текста выполнялась с помощью Codex; индивидуальные роли и авторскую проверку команда должна подтвердить перед сдачей.

Источник - [датасет В. Вялковой на Kaggle](https://www.kaggle.com/datasets/veronikavyalkova/saint-petersburg-avito-listings), объявления Avito за февраль 2026. Это цены предложений, а не сделок. Основная NLP-задача - сравнение представлений текста и извлечение его семантических признаков для последующей регрессии. Факторы цены интерпретируются как предиктивные связи.

## Что сравнивается

- Базовая и расширенная таблицы + CatBoost.
- TF-IDF + Ridge, смесь прогнозов и cross-fitted модель остатков.
- Qwen3-Embedding-0.6B + Ridge и residual-гибрид с таблицей.
- Словарные признаки и признаки Qwen3-1.7B с проверкой цитат.
- Прямой прогноз Qwen3-1.7B без примеров и с тремя похожими объявлениями из train.

В полном сравнении 30 775 объявлений. Генеративное сравнение ограничено фиксированными 1400 train / 300 dev / 400 test; все его обучаемые методы используют одинаковые train. Адреса и одинаковые текстовые шаблоны между train/dev/test не пересекаются. Это не устраняет все возможные повторы квартир.

## Быстрый просмотр результатов

- `output/pdf/avito_nlp_report.pdf` - академический отчёт со структурой шаблона курса.
- `artifacts/full/metrics.json`, `artifacts/subset/metrics.json` - фактические метрики каждого метода.
- `artifacts/*/bootstrap.csv` - парные групповые интервалы относительно базовой таблицы.
- `artifacts/full/shap_importance.csv`, `ablations.json` - анализ факторов.
- `artifacts/figures/` - графики; `artifacts/*/predictions.csv` - предсказания на test в **log RUB**, исходная колонка price в RUB.
- `PROTOCOL.md`, `DATA_CARD.md`, `MODEL_CARD.md`, `SOURCES.md` - методика, источники и ограничения.
- `VERIFICATION.md` - единый журнал фактических проверок.

Публичного подтверждения рекорда на этом датасете нет. Результат нового split сам по себе не является доказательством SotA.

## Воспроизведение

Нужны Python 3.10+, около 10 GB свободного диска, рекомендуемая RAM 16 GB и GPU от 6 GB VRAM. Веса моделей и исходные данные в архив кода не входят. На Windows используйте `python`, в Colab - его штатный Python. Проверялись версии из `artifacts/environment.json`; точная аппаратная детерминированность не гарантируется.

```bash
python -m venv .venv
# Активируйте .venv в своей оболочке.
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

Первый запуск скачает публичные данные Kaggle и открытые веса Hugging Face, последовательно выполнит GPU-этапы, сравнения, анализ, проверки и соберёт PDF. В Windows для отчёта используются системные Times New Roman; в Linux - DejaVu Serif. Веса закреплены конкретными revisions в коде. Сеть нужна только для загрузки.

Если уже есть именно закреплённые веса Qwen3-1.7B, задайте `GENERATION_MODEL_PATH` (путь без секретов) и используйте `--skip-generation-download`. При наличии исходных данных можно добавить `--skip-download`. Прерванные LLM/embedding-этапы продолжаются по сохранённому прогрессу. Не меняйте данные/промпты и не смешивайте кэш разных экспериментов: новый протокол запускается в новой папке без старых `cache`, `artifacts/*/models` и raw LLM-журналов. Модельные pickle/joblib-файлы загружайте только из собственного доверенного прогона.

Отдельные этапы:

```bash
python scripts/download_data.py
python scripts/read_workbook.py
python scripts/prepare.py
python scripts/download_embedding.py
python scripts/download_generation.py
python scripts/llm.py features
python scripts/llm.py features --prompt-version fewshot
python scripts/llm.py prices
python scripts/embed.py
python scripts/benchmark.py --embeddings
python scripts/benchmark.py --subset --embeddings --llm
python scripts/rich_oof.py
python scripts/augment_rich.py
python scripts/evaluate_llm_prices.py
python scripts/refresh_test_metrics.py
python scripts/analyze.py
python scripts/verify_results.py
python -m unittest discover -s tests -v
python scripts/build_report.py
```

Ссылка на этот репозиторий указана в `project.json` и в Abstract PDF-отчёта. При переносе проекта обновите URL и пересоберите отчёт.

## Данные, ограничения и права

Исходные описания и адреса не выкладываются в этот репозиторий. Данные автора: CC BY-NC-SA 4.0; веса Qwen: Apache-2.0; оригинальный код проекта: MIT. Лицензии независимы. Нет ручного gold набора атрибутов, коммерческих моделей, цен реальных сделок и причинной оценки стоимости ремонта. JSON/цитата, успешно прошедшие проверку, не гарантируют правильного понимания текста.

Основные опубликованные аналоги: [Zhang et al.](https://doi.org/10.1017/S1351324923000360), [Geerts et al.](https://arxiv.org/abs/2506.11812) и их [открытый код](https://github.com/margotgeerts/LLM4RealEstate). Чужие показатели нельзя напрямую сравнивать с нашими рублёвыми ошибками. Все методы в наших таблицах реализованы/адаптированы и запущены на общем split.
