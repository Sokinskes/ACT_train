Title: Add λ-sensitivity & robustness analyses; clarify method and add noise‑sensitivity proxy

Summary
-------
This PR completes the reviewer-driven hardening of the AdaStep paper and codebase:
- Clarifies the role of the safety coefficient \lambda as an explicit risk‑aversion knob in the Method section.
- Adds a formal Lipschitz-based proposition and its derivation in Appendix A.
- Replaces absolute/overgeneralized claims with conditionally qualified statements across reports and the manuscript.
- Adds and runs a noise‑sensitivity robustness proxy (offline) and λ sensitivity sweep; results and figures are included.
- Provides reproducible scripts and results in `experiments/sensitivity_results/`.

Why
---
These changes address reviewer concerns about theoretical grounding, robustness, and whether offline results can be misleading. The noise proxy and λ sweep turn an apparent contradiction (large k but high offline success) into a strength: AdaStep is a tunable risk–efficiency controller suitable for deployment.

Key files changed
-----------------
- experiments/latex_submission/main.tex (method, experiments, appendix, figures)
- experiments/noise_sensitivity_proxy.py (new)
- experiments/run_sensitivity_single_lam.py (new)
- experiments/run_sensitivity_experiment.py (updated)
- experiments/QUALITATIVE_ANALYSIS_REPORT.md (updated)
- experiments/FINAL_FOUR_TASK_REPORT.md (updated)
- experiments/COMPLETE_ANSWERS_TO_QUESTIONS.md (updated)
- experiments/paper_draft.md (updated)
- experiments/sensitivity_results/* (new experimental outputs: CSV + PNG)

Reproducibility (commands to reproduce key results)
--------------------------------------------------
# 1) λ sensitivity sweep (Square, offline)
cd experiments
conda activate act
python run_sensitivity_experiment.py \
  --data_path ../robomimic_data/square/mh/low_dim_v15.hdf5 \
  --output_dir ./sensitivity_results --epochs 5

# 2) Noise-sensitivity proxy (offline robustness proxy)
python noise_sensitivity_proxy.py

# 3) Extended noise sweep (multiple noise levels)
# (already run in this PR; generates noise_sensitivity_extended.csv/png)

What to review
---------------
- Manuscript text: `experiments/latex_submission/main.tex` (Method, Experiments §4.4, Appendix A)
- New experiments and plots in `experiments/sensitivity_results/` (verify figures and CSVs)
- Report wording: ensure the conditional phrasing is acceptable to reviewers

Suggested reviewer reply (short)
--------------------------------
We added a λ‑sensitivity analysis and a noise‑injection robustness proxy to explicitly demonstrate that (i) offline replay can mask open‑loop fragility and (ii) AdaStep provides a user‑configurable risk knob (λ) to trade robustness for efficiency; see Fig. X and Sec. 4.4. These additions, together with a concise Lipschitz-based derivation, directly address the reviewer’s concerns about theoretical grounding and robustness.

Checklist (for maintainers)
---------------------------
- [x] All modified LaTeX builds locally (checked)  
- [x] New figures saved under `experiments/sensitivity_results/`  
- [x] Repro commands added to PR body  

Notes
-----
If you want I can also open a draft PR on GitHub (I will attempt to use the `gh` CLI); if `gh` is unavailable I have pushed a branch and provided the PR body here for manual submission.
