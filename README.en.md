# ASA Benchmark

[中文](README.md) · [English](README.en.md)

A **classifier-based artist-label consistency diagnostic** for generated anime illustrations. Kaloscope 2.0 (LSNet) scores images generated with target artist labels, allowing comparisons across models or prompt conditions. The scores measure association with a particular classifier; they do not establish authorship, successful style reproduction, or overall visual quality.

## Method and metrics

| Metric | Current implementation and interpretation |
| --- | --- |
| Top-1 / Top-5 hit rate | Whether the target label appears among the classifier's first 1 / 5 predictions, within its fixed label space |
| Consistency score | `1 − std(Top-5 rates across prompt groups)`; a heuristic summary of between-group dispersion |
| Weighted ASA | `mean group Top-5 × consistency score`; a composite without independent construct validation |
| Style resilience | Complex-background rate divided by standard-prompt rate; report numerator, denominator, and sample counts |
| Difference from baseline | Score difference between artist-conditioned and unconditioned generation under comparable model, content, seed, and sampling settings |

Classifier predictions are not ground truth. Artist labels, composition, characters, color, and training-data overlap can affect these scores. Shared models or data can bias evaluation. A Top-5 hit is not a human quality judgment.

## Usage

Obtain matching Kaloscope 2.0 weights and `class_mapping.csv` from the [model release](https://huggingface.co/heathcliff01/Kaloscope2.0). Derive the class count from the specific checkpoint and verify it against the mapping: the historical 39,260 figure and the release page's later 39,261 check must not be mixed. Weights and the complete evaluation dataset are not included here.

1. Install CUDA-compatible PyTorch and torchvision, plus NumPy, Pillow, timm, matplotlib, and tqdm. The LSNet SKA path also requires Triton.
2. Replace historical machine paths in [`benchmark_config.json`](benchmark_config.json) with local weights, mapping, images, and output paths.
3. Organize images by prompt type and retain target labels in the filename format parsed by the evaluator.

```bash
python batch_asa_eval.py --config benchmark_config.json
```

[`batch_asa_eval.py`](batch_asa_eval.py) performs scoring and aggregation. [`run_generation_task.py`](run_generation_task.py) requires a separately configured ComfyUI instance. Scoring includes `CenterCrop(448)` and does not evaluate the full image. Training checkpoints may use pickle-based loading; load only trusted files.

## Reproducibility and limitations

Publish the code commit, classifier and mapping hashes, prompts, seeds, sampling settings, per-image results, valid/missing sample counts, and label distribution. Report artist and prompt groups separately, add independent representations or blinded human review, and audit near-duplicates across training and evaluation. This repository does not establish a validated universal style measure or cross-model leaderboard.

## Attribution and license

Repository code is covered by [Apache-2.0](LICENSE). The classifier, upstream LSNet code, images, and generation models retain their respective terms. Attribute tool maintenance separately from model and method contributions.
