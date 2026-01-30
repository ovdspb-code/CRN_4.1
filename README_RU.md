# CRN — репозиторий воспроизводимости

Этот репозиторий упаковывает код, численные бэкенды (CSV) и фигуры, использованные в Core‑статье CRN и в Supplementary Materials.

Ключевая оговорка: **формализм GKSL/Линдблада используется как механистический прокси** для *кратковременной волновой (wave‑like) динамики* при дефазировке/«измерении». Это **не требует** (и не утверждает) микроскопическую «квантовую вычислительность» нейронов.

Покрытие:
- **C. elegans** — touch‑субконтур (малый реальный подграф; stress/temperature sweeps)
- **Drosophila larva** — реальный коннектом‑субсистемы (DES‑пик + зависимость от архитектуры: rewiring/lesion)
- **Mouse cortex proxy** — прокси‑модель микросхемы (permeability vs confinement; иерархический SBM)
- **Эволюционная игра / отбор стратегий** — companion dataset (границы trade‑off, стратегия‑каталог, пример репликаторной динамики)

## Быстрый старт (лучше — чистое окружение)

```bash
conda create -n crn python=3.10 -y
conda activate crn
python -m pip install -U pip
python -m pip install -r env/requirements.txt
```

Если в вашей базовой Anaconda «сломаны метаданные» пакетов (типичный симптом: `pyodbc 4.0.0-unsupported` и ошибки pip), **не пытайтесь чинить это в base** — создайте новое окружение, как выше.

## Что запускать (минимальные точки входа)

1) **Drosophila (пайплайн A7–A8)**

См. `experiments/drosophila/README.md`.

2) **Mouse cortex proxy + architecture trade‑off + evolutionary selection (Steps 2–6)**

```bash
cd experiments/step6_release_candidate/CRN_step6_release_candidate
python steps/step2_repro_bundle/CRN_step2_repro_bundle/step2_cortex_transport/run_cortex_transport.py
```

Дальше — по README внутри этого бандла.

3) **C. elegans (touch circuit sweeps)**

```bash
python experiments/elegans_touch_circuit/crn_nematode_pipeline.py
```

4) **Companion archive (эволюционная игра)**

Zenodo DOI: `10.5281/zenodo.18379851`.

Публичный снимок того же набора данных продублирован в:
- `datasets/zenodo_18379851_evolutionary_game/`

## Лицензии и данные

- Код: MIT (`LICENSE-CODE.txt`).
- Производные таблицы/фигуры этого проекта: CC BY 4.0 (`LICENSE-DATA.txt`).
- Сырые сторонние коннектомы имеют свои условия. См. `data/THIRD_PARTY_DATA.md`.
