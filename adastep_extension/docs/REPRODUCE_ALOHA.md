Reproducibility checklist — ACT-Plus-Plus (Mobile ALOHA)

Quick start (smoke reproduction)

1. Create env and install deps

```bash
bash scripts/setup_aloha_env.sh  # creates conda env 'aloha' (then: conda activate aloha)
pip install -r scripts/requirements_aloha.txt
```

2. Run a smoke conversion (creates a tiny synthetic hdf5 so downstream code loads)

```bash
python3 scripts/convert_rosbag_to_hdf5.py --out ./tmp/sample_smoke.hdf5
python -c "import h5py; f=h5py.File('tmp/sample_smoke.hdf5','r'); print(list(f.keys()))"
```

3. (Optional) run training smoke

```bash
bash scripts/train_imitate.sh sim_transfer_cube_scripted ../ckp 2
```

What I added to the repo
- `docs/act-plus-plus-aloha_csdn.md` — extracted summary + runnable snippets
- `scripts/requirements_aloha.txt` — article dependency list
- `scripts/*` — conversion / train / eval helpers
- `.github/workflows/reproduce-aloha.yml` — CI smoke workflow (runs dependency install + smoke checks)

Caveats & next steps
- The repo may require specific `robomimic` branches or mujoco binaries; see the notes in `docs/act-plus-plus-aloha_csdn.md`.
- If you want, I can (A) add a tiny test HDF5 sample to `tests/data/` and wire up full CI, or (B) attempt a one‑shot download of the author's sample checkpoints into `data/` (you must confirm permission to download).