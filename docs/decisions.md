# Decision log

Short ADR-style entries. Newest last.

### D01 - Environment has 7.7 GB RAM, not 16 GB
The WSL2 VM on the build machine exposes 7.7 GB (i5-10210U, 4C/8T). All defaults are sized for
that: one LLM loaded at a time, bounded SHAP samples, a single API process.

### D02 - Pinned versions resolve without conflict
`scikit-learn 1.9.1`, `river 0.22.0`, `evidently 0.7.23`, `langgraph 1.2.20`, `mlflow 3.16.1`
resolve together on Python 3.12 under uv (`uv.lock` committed). No pin was changed. Unpinned
packages float to what uv resolved and are frozen by the lockfile.

### D03 - Reference window = validation split
Drift is measured against data the model was *not* fitted on, so reference score distributions
are not optimistically biased by training-set memorisation.

### D04 - Dropped columns
Adult: `fnlwgt` (sampling weight), `education` (duplicate of `education_num`).
Bank: `duration` - known only after the call outcome; using it is target leakage, as the dataset
authors note. Dropping it lowers headline F1 substantially and is the correct trade.

### D05 - Model selection by CV ROC-AUC with a simplicity margin
Accuracy is not used: majority-class accuracy is 76% (Adult) and 88% (Bank). A more complex
model must beat logistic regression by > 0.005 CV ROC-AUC. Threshold is tuned for F1 on
validation; test is touched once.

### D06 - Own file registry for integrity, MLflow for tracking
MLflow records experiments, metrics and model versions. The serving-integrity check (F10) needs
a content hash compared on every monitoring run, so a minimal registry stores
`record.json` + `model.joblib` and verifies SHA-256 on load. Two tools, two jobs.

### D07 - MLflow skops trusted types
MLflow 3.16 serialises scikit-learn models with skops and refuses unknown types. `numpy.dtype`
and HGB's `TreePredictor` are explicitly trusted for models produced by this repository's own
training code. Nothing downloaded is ever deserialised.
