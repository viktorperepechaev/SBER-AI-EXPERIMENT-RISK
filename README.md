# Запуск

Установите [uv](https://docs.astral.sh/uv/getting-started/installation/), затем выполните из корня репозитория.

Один эксперимент:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt experiment.py
```

Все 20 экспериментов:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt experiment.py --all
```

Сохранённые результаты 20 экспериментов — `results.csv`. Новые результаты записываются в `result.json` или `run_results.csv` соответственно.
