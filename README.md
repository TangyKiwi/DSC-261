# Robustness of Vision Watermark Detection Against Adversarial Attacks

DSC 261 (Fall 2026) final project. A ResNet50 detector for a fixed ±λ invisible watermark on
CIFAR-10, attacked with FGSM and PGD, with FGSM/PGD adversarial training and a watermark-strength
sweep.

## Files

| Path | What it is |
|---|---|
| `final.ipynb` | **Submission notebook.** Loads saved results and reproduces every table and figure (no GPU needed). |
| `wmrobust.py` | All model, watermark, attack, training and evaluation code. |
| `run_experiments.py` | Trains and evaluates all 8 detectors; resumable. |
| `make_report_assets.py` | Builds report figures, LaTeX tables and number macros from a run. |
| `controls.py` | Control experiments on every saved detector: random ±ε noise and the oracle (worst-case) attacker. |
| `dsmlp_job.sh` | The DSMLP background job used for the full run (installs a Blackwell-capable PyTorch, then runs everything). |
| `runs/full/` | Results of the full run: `all_results.csv`, per-model CSVs, training histories, `log.txt`. |
| `report/` | Final report (PVLDB template). `report/main.pdf` is the compiled report. |
| `proposal/` | Project proposal. |
| `main revised Sam.ipynb`, `main.ipynb` | The team's exploratory notebooks (superseded by `final.ipynb`). |

## Reproduce

```bash
# 1. GPU machine (e.g. DSMLP scipy-ml-notebook with 1 GPU)
python run_experiments.py --quick        # ~5 min smoke test -> runs/quick/
python run_experiments.py --exp all      # full study -> runs/full/ (re-run the same command to resume)

# 2. Anywhere (needs runs/full/checkpoints for the control)
python controls.py --run runs/full
python make_report_assets.py --run runs/full --dest report
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Then open `final.ipynb` and run all cells.

## Experimental design (summary)

* Watermark: `x_W = clip(x + λ·W)`, `W ∈ {−1,+1}^{3×32×32}` fixed (seed 2026); main λ = 0.02.
* Detector: ImageNet-pretrained ResNet50, 2-way head; SGD lr 1e-3, momentum 0.9, batch 128, 10 epochs,
  best validation epoch kept.
* Attacks: FGSM; PGD with 10 steps, α = ε/4, random start; ε ∈ {0.0025, 0.005, 0.01, 0.02, 0.03}.
* Adversarial training: clean + adversarial batch, ε_train ∈ {0.005, 0.01}.
* Test set: paired — every test original evaluated clean and watermarked (20,000 images).
* For ε ≥ λ/2 no detector can classify both copies of an image correctly (report, Sec. 3.7).

LLM disclosure: the final code and report drafts were written with assistance from Claude (Anthropic).
