# QGIS Segmentation — Training Notebook (8-Class)

This folder contains `QGIS_Segmentation_8Class_run.ipynb`, a ready-to-run notebook for **semantic terrain segmentation** from aerial tiles + masks (TMS layout). The notebook includes a robust training workflow:

* per-run folders under `runs/`
* full checkpoints (`latest.pth`, `best.pth`)
* resume-from-checkpoint (Option A: resume in the same run folder)
* CSV logging + config snapshot
* early stopping
* STOP flag (finish epoch → save → exit)
* AMP + gradient accumulation
* safe DataLoader defaults for Windows/Jupyter

---

## Requirements

### Python / venv

* Python 3.9+ (virtualenv recommended)
* Windows: PowerShell

Activate your venv:

```powershell
& .venv\Scripts\Activate.ps1
```

### Install dependencies

Install general dependencies:

```powershell
pip install -r requirements.txt
```

> **GPU note (RTX 5050 / sm_120):** You must use the **nightly CUDA 12.8** PyTorch build (cu128). If you use a stable CPU build or an older CUDA build, CUDA may be unavailable or incompatible.

Recommended install (nightly cu128):

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

Quick verification:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Quick start

1. Open `QGIS_Segmentation_8Class_run.ipynb` and run cells in order.
2. Start training. A run folder is created under:

   * `runs/<RUN_NAME>/`
3. The training cell writes:

   * `latest.pth` — overwritten each epoch (resume point)
   * `best.pth` — saved only when validation IoU improves
   * `epoch_XXXX.pth` — optional snapshots (if enabled)
   * `train_log.csv` — per-epoch metrics (loss, IoU, LR, time)
   * `config.json` — run settings for reproducibility
   * `STOP` — optional flag file to request a clean stop

---

## STOP / Safe pause (laptop-friendly)

To stop training safely (finish current epoch → save → exit), create a file named `STOP` inside the **active** run folder.

PowerShell example:

```powershell
New-Item -Path runs\<RUN_NAME>\STOP -ItemType File
```

Training checks for `STOP` after each epoch, saves `latest.pth`, and exits cleanly.

---

## Resume (Option A — recommended)

**This notebook is configured for Option A: resume inside the same run folder.**

### How resume works

* The training cell loads `runs/<RUN_NAME>/latest.pth` **if it exists**.
* It resumes at `start_epoch = last_epoch + 1`.
* It continues writing logs/checkpoints into the **same** run folder.

### Steps to resume

1. Find the run folder you want to resume, e.g.:

   * `runs/20260219_231455/`
2. In the notebook, set:

   * `RESUME_RUN_NAME = "20260219_231455"` (example)
3. Increase `NUM_EPOCHS` to the total you want (e.g., 100).
4. Run the training cell again.

### Important notes

* Resume expects the **same model architecture** and compatible optimizer/scheduler configuration.
* If you change the model backbone or number of classes, you must start a new run.

---

## Logs and plotting

* Metrics are appended to:

  * `runs/<RUN_NAME>/train_log.csv`
* Use the plotting cell in the notebook to visualize:

  * `train_loss`, `val_loss`, `val_iou`, learning rate, epoch time

---

## Safe defaults (recommended for laptop)

These settings prioritize stability on Windows/Jupyter and 8GB VRAM:

* `BATCH_SIZE = 2`
* `ACCUM_STEPS = 4`
* `NUM_WORKERS = 0` in Windows notebooks (prevents random DataLoader hangs)
* `VAL_EVERY = 1` or `2`
* `SNAPSHOT_EVERY = 10`

If you later run training as a `.py` script (not Jupyter), you can try:

* `NUM_WORKERS = 2` (and optionally enable `prefetch_factor` / `persistent_workers`)

---

## Notes / Troubleshooting

### First-iteration “stall” (normal)

The first CUDA forward pass can be slow due to GPU warmup/kernel selection. After the first few batches, throughput stabilizes.

### CUDA OOM

If you hit out-of-memory:

* reduce `BATCH_SIZE`
* increase `ACCUM_STEPS`
* reduce image size / heavy augmentations

### Legacy output

The notebook may write `best_qgis_model.pth` (weights only) for backward compatibility with older scripts that expect that filename.

---
