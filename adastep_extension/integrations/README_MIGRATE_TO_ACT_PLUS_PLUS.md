Migration plan — integrate AdaStep (this repo) into `act-plus-plus`

Goal: port AdaStep (HorizonPredictor + experimental scripts) into the ACT codebase as a first-class, EDGE‑friendly option that uses ACT (not Diffusion). This document and the accompanying adapter/CI/scripts make the migration reproducible.

1) High-level mapping (this repo → act-plus-plus)
- predictor (AdaStep)  -> `act-plus-plus` plugin: `predictors/adastep.py`
- offline evaluation scripts -> `tools/eval_adastep.py`
- transport_simulation_validation.py -> `tools/sim_validation/transport_validation.py`
- configs/ -> add `configs/adastep_*_edge.yaml` (low-res / reduced cameras)

2) Required artifacts added here (in `integrations/`):
- `adastep_adapter.py` — minimal integration wrapper (load AdaStep predictor, call from ACT policy loop)
- `export_to_torchscript.py` — export AdaStep+ACT policy to TorchScript
- `edge_config/` — example configs tuned for Orin Nano (224x224, 1-2 cams, batch_size=1, fp16)
- `ci/edge-smoke.yml` — CI: export & run forward on synthetic data

3) Edge constraints & tests (what to validate before PR)
- TorchScript export successful on CPU/GPU
- Single-step forward latency < 40 ms (synthetic benchmark)
- Model size < 200 MB (or quantized to < 50 MB for TensorRT)
- Deterministic adapter: unit tests that compare predictor outputs on saved sample hdf5

4) Licensing & attribution
- Keep AdaStep license header in ported files
- Cite CSDN-derived integration notes in `docs/` (CC BY-SA)

How I suggest we proceed now (automatable):
- Create `integrations/adastep_adapter.py` + `edge_config` + export/quantize scripts and a CI smoke job.
- Add unit tests that run on synthetic HDF5 (already present in repo tests or create one).

