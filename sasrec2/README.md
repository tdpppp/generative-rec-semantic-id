# SASRec2 Temporal Baseline

This isolated baseline reuses the read-only SASRec model and TencentGR profile loader, but applies the same temporal train/validation/test protocol as the Qwen experiments. Outputs belong under `outputs/sasrec2/` and do not modify `sasrec/`.

Train the baseline with:

```bash
python -m sasrec2.train \
  --data_path data/TencentGR_1M \
  --profile_path data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82 \
  --output_dir outputs/sasrec2/temporal_10k_mm81_82 \
  --batch_size 256 --num_epochs 3 --lr 0.001 --device cuda:0
```

Evaluate a checkpoint with the same validation/test users used by Qwen:

```bash
python -m sasrec2.evaluate \
  --data_path data/TencentGR_1M \
  --profile_path data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82 \
  --checkpoint outputs/sasrec2/temporal_10k_mm81_82/checkpoints/best_model.pt \
  --split validation --output outputs/sasrec2/temporal_10k_mm81_82/validation_metrics.json \
  --predictions outputs/sasrec2/temporal_10k_mm81_82/validation_predictions.jsonl
```

The original `sasrec/` entry point remains unchanged. SASRec2 reports item-level ranking metrics; these must not be silently mixed with Qwen's Semantic-ID-level metrics.
