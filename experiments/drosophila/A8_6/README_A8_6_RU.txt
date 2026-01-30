A8.6 — постобработка A8.5 (ratio-of-means, квантили, P_good, p-values)

Что нужно:
- один или два файла A8_5_arch_trialwise.csv (например, для kappa=0.001 и для kappa=1.0)

Рекомендуемый workflow:

(1) Прогон A8.5 для kappa=1.0 (если у вас уже есть kappa=0.001)
    mkdir -p A8_5_OUT_k1
    python3 A8_4_5_bundle_localization_arch/A8_5_architecture_dependence.py \
      --a7_dir A7_outputs \
      --out_dir A8_5_OUT_k1 \
      --n_trials 20 \
      --n_surrogates 20 \
      --epsilons "0,1,2,3,5" \
      --kappas "1.0" \
      --lesion_drop 0.5

(2) Постобработка (объединяем два trialwise-файла)
    mkdir -p A8_6_OUT
    python3 A8_6_postprocess_A85.py \
      --inputs "A8_5_OUT_big/A8_5_arch_trialwise.csv,A8_5_OUT_k1/A8_5_arch_trialwise.csv" \
      --out_dir A8_6_OUT \
      --pT_min 0.005 \
      --sel_thresh 2.0

Выходы:
- A8_6_cell_metrics.csv
  (variant, epsilon, kappa): ratio-of-means, mean-of-ratios, median/q25/q75/q10/q90, P_good + Wilson CI.
- A8_6_replicate_metrics.csv
  replicate = trial (для original) или surrogate_id (для rewired/lesion).
  Используется для error bars на графиках.
- A8_6_pairwise_tests_epsilon3.csv
  p-values на epsilon=3:
    * Fisher exact для P_good
    * Mann–Whitney U для log(Selectivity_end)
- PNG графики (если стоит matplotlib):
  * A8_6_Selectivity_ratioOfMeans_vs_epsilon_kappa*.png
  * A8_6_goodrun_prob_vs_epsilon_kappa*.png
  * A8_6_coverage_end_vs_epsilon_kappa*.png

Если matplotlib не нужен:
  добавьте флаг --no_plots
