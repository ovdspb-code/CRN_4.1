A8.4 / A8.5 — что делать и как запускать (простыми шагами)

Контекст
--------
У вас уже есть результаты A7 (папка A7_outputs). Это главное.
A8.4 и A8.5 — это следующий шаг:
  A8.4: "локализация" (Anderson-style) на реальном графе + связь с Selectivity_end из A7
  A8.5: "архитектурная зависимость" — мы берём реальный граф, делаем контролируемые
        in-silico модификации (rewire / lesion) и проверяем, сохраняется ли эффект.

Важно: НЕ нужно никакого "A7_OUT".
Скрипты принимают путь к вашей папке A7_outputs.


0) Минимальные зависимости
-------------------------
Нужно, чтобы в вашем python окружении были:
  networkx, numpy, scipy, matplotlib

Если вы на Anaconda (base), обычно уже есть.


1) Где должны лежать папки
-------------------------
Предположим, вы находитесь в вашей рабочей папке, где у вас уже есть:
  A7_outputs/
    A7_meta.json
    A7_gksl_sweep.csv
    A7_baselines.csv
    A7_energy_seeds.csv

И там же вы распаковали этот пакет A8_4_5_bundle_localization_arch/

Пример структуры:
  .../REAL/
      A7_outputs/
      A8_4_5_bundle_localization_arch/
          A8_4_localization_diagnostics.py
          A8_5_architecture_dependence.py
          README_A8_4_5_RU.txt


2) Запуск A8.4 (локализация)
---------------------------
Сделайте папку под вывод:
  mkdir -p A8_4_OUT

Запуск:
  python3 A8_4_5_bundle_localization_arch/A8_4_localization_diagnostics.py \
    --a7_dir A7_outputs \
    --out_dir A8_4_OUT \
    --kappa 0.001

Что получите:
  A8_4_OUT/A8_4_localization_trialwise.csv
  A8_4_OUT/A8_4_localization_by_epsilon.csv
  A8_4_OUT/A8_4_deltaPR_vs_selectivity.png
  A8_4_OUT/A8_4_deltaPR_vs_epsilon.png

Если хотите посмотреть не "почти когерентный" режим, а другой, например kappa=1.0:
  ... --kappa 1.0


3) Запуск A8.5 (архитектурная зависимость)
-----------------------------------------
Это более тяжёлый расчёт. Начните с дефолта.

Сделайте папку под вывод:
  mkdir -p A8_5_OUT

Запуск (пример):
  python3 A8_4_5_bundle_localization_arch/A8_5_architecture_dependence.py \
    --a7_dir A7_outputs \
    --out_dir A8_5_OUT

По умолчанию будет:
  epsilons = 0,3,5
  kappas   = 0.001,1.0
  n_trials = 5
  n_surrogates = 3 (для rewired_type и lesion)
  lesion_drop = 0.5 (доля KC->MBON, которую выкидываем)

Если на вашем ноутбуке долго, уменьшите, например:
  --n_trials 3 --n_surrogates 2

Что получите:
  A8_5_OUT/A8_5_arch_trialwise.csv
  A8_5_OUT/A8_5_arch_summary.csv
  A8_5_OUT/A8_5_selectivity_vs_epsilon.png


4) Если скрипт ругается на пути (самая частая проблема)
------------------------------------------------------
У вас в путях встречается папка "СRN" (возможно, кириллическая "С", не латинская "C").
Самый надёжный способ:
  - в Finder перетащите папку в окно Terminal (он сам вставит путь)
  - или используйте tab completion


5) Что именно мы проверяем научно (критерий успеха)
---------------------------------------------------
A8.4:
  Если при epsilon≈3 дифференциальная локализация (delta_PR_TminusD) максимальна
  и/или коррелирует с Selectivity_end (при малом kappa), это поддерживает механизм
  "distractors локализуются сильнее, чем target".

A8.5:
  Если в real/original графе есть пик (или заметный рост) Selectivity_end при epsilon≈3,
  а в rewired_type и/или lesion_KC_MBON он исчезает/сильно уменьшается, то эффект
  архитектурно-специфичен (не просто статистика степеней).

