In this repository we are developing a deep learning model to detect and classify exoplanets. # AstroVision

## Recent pipeline enhancements

- **Balanced training** – the `exo_tabular` training commands now weight each sample inversely to its class frequency, helping the classifier pay more attention to the rare confirmed exoplanets.
- **Optional oversampling** – pass `--oversample` to `python -m src.exo_tabular ...` or `python -m src.export_predictions ...` to duplicate minority-class examples with an internal `RandomOverSampler` (requires `imbalanced-learn`).
- **Calibrated thresholds** – provide a target recall via `--recall-target` (default `0.6`). The training reports (`metrics_*.json`) include a recommended probability threshold that achieves at least that recall, and prediction modes can consume it with `--use-calibrated-buckets`/`--use-calibrated-thresholds` to adjust the candidate bucket.
- **Within-mission exports** – `python -m src.export_within_mission ...` now oversamples the positive class by default, records calibrated candidate thresholds that hit the desired recall, saves both JSON and text metric reports, and uses the calibrated threshold automatically unless `--keep-default-thresholds` is provided.

Make sure `imbalanced-learn` is installed to enable the oversampling option:

```bash
pip install imbalanced-learn
```
