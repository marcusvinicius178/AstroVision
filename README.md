In this repository we are developing a deep learning model to detect and classify exoplanets. # AstroVision

## Recent pipeline enhancements

- **Balanced training** – the `exo_tabular` training commands now weight each sample inversely to its class frequency, helping the classifier pay more attention to the rare confirmed exoplanets.
- **Optional oversampling** – pass `--oversample` to `python -m src.exo_tabular ...` or `python -m src.export_predictions ...` to duplicate minority-class examples with an internal `RandomOverSampler` (requires `imbalanced-learn`).
- **Calibrated thresholds** – provide a target recall via `--recall-target` (default `0.6`). The training reports (`metrics_*.json`) include a recommended probability threshold that achieves at least that recall, and prediction modes can consume it with `--use-calibrated-buckets`/`--use-calibrated-thresholds` to adjust the candidate bucket.
- **Within-mission exports** – `python -m src.export_within_mission ...` now oversamples the positive class by default, records calibrated candidate thresholds that hit the desired recall, saves both JSON and text metric reports, and uses the calibrated threshold automatically unless `--keep-default-thresholds` is provided.
- **Safe threshold fallback** – when a calibrated candidate cutoff is greater than or equal to the planet threshold (e.g., when the model cannot reach the requested recall before hitting 0.95), the tooling automatically keeps the default candidate threshold so that the candidate bucket never collapses.

## Rebuilding metrics, charts and exports

After installing the project requirements (and optionally `imbalanced-learn`), the following commands recreate the key artifacts from scratch. They assume you are in the repository root.

1. **Clean existing artifacts (optional but recommended):**
   ```bash
   rm -rf artifacts/*
   ```

2. **Cross-mission training (produces ROC/PR/confusion PNGs & metrics JSON):**
   ```bash
   for mission in tess k2 kepler; do
     python -m src.exo_tabular \
       --mode train \
       --split cross-mission \
       --test-mission "$mission" \
       --device cpu \
       --oversample \
       --recall-target 0.6
   done
   ```

3. **Cross-mission predictions with calibrated buckets:**
   ```bash
   for mission in tess k2 kepler; do
     python -m src.exo_tabular \
       --mode predict \
       --split cross-mission \
       --test-mission "$mission" \
       --use-calibrated-buckets
   done
   ```

4. **Automated export pipeline (optional convenience wrapper):**
   ```bash
   python -m src.export_predictions \
     --data-dir ./data \
     --artifacts-dir ./artifacts \
     --runs cross-mission \
     --oversample \
     --recall-target 0.6 \
     --use-calibrated-thresholds
   ```

5. **Within-mission exports (70/30 split, metrics + CSVs):**
   ```bash
   python -m src.export_within_mission \
     --data-dir ./data \
     --artifacts-dir ./artifacts \
     --missions kepler k2 tess \
     --test-size 0.30 \
     --seed 42 \
     --threshold-planet 0.95 \
     --threshold-candidate 0.50
   ```

Each command logs the selected thresholds, saves the confusion matrix/curve charts, and drops mission-specific CSVs (planets, candidates, non-planets, discrepancy tables) into the `artifacts/` tree.

Make sure `imbalanced-learn` is installed to enable the oversampling option:

```bash
pip install imbalanced-learn
```
