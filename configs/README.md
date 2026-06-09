# configs/

One JSON file per named experiment run. Notebooks load these by name rather than hardcoding parameters inline.

## File naming

Filename = `run_name` field with `-` replaced by `_`. The `run_name` is used by the library to search for matching experiment directories.

## Files

| File | Used in | Notes |
|---|---|---|
| `train_contrastive_positive_pairs.json` | `01-train.ipynb` | Full training config including lr, warmup, eval strategy |
| `contrastive_fixed_stratified_25k.json` | `02`, `03`, `04` | 25k stratified pairs, contrastive loss |
| `contrastive_fixed_stratified_50k.json` | `02`, `03`, `04` | 50k stratified pairs, contrastive loss |
| `contrastive_fixed_default_10.json` | `02`, `03` | Default (exhaustive) sampling |
| `positive_fixed_default_10.json` | `02` | Positive pairs only, default sampling |
| `positive_fixed_default_20.json` | `02` | Positive pairs only, default sampling |
| `positive_fixed_stratified_10.json` | `02` | Positive pairs only, stratified sampling |
| `positive_fixed_stratified_20.json` | `02` | Positive pairs only, stratified sampling |

## num_samples

`null` in `train_contrastive_positive_pairs.json` means exhaustive pairing — all pairwise combinations are used.  
Set to an integer (e.g. `25000`) to cap pairs and switch to stratified sampling instead.
