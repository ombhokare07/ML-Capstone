# Freight Signal: Supply Chain Delivery Risk Prediction

## Executive summary

Freight Signal is a machine-learning capstone prototype for estimating whether
an order should receive additional delivery-risk attention. It uses order,
supplier, route, and logistics attributes, then exposes predictions through a
Flask web interface and JSON endpoints.

The checked-in example contains 438 balanced records and is synthetic. A tuned
RBF Support Vector Classifier (SVC) achieved 78.4% accuracy and 83.5% ROC-AUC
on an 88-record held-out test set. These figures demonstrate the workflow on
the supplied data; they are not evidence of expected performance on real
shipments. A real-world pilot requires representative historical data,
business-defined delivery labels, and validation over time.

## 1. Problem definition and business understanding

Late deliveries can disrupt inventory availability, customer commitments, and
transportation planning. Teams need a way to identify orders that may warrant
earlier review, such as checking supplier status, route conditions, port
congestion, or customs plans.

**Business question:** Given information available when an order is assessed,
can the project identify shipments that may merit extra attention?

**Intended users:** Supply-chain planners and logistics teams.

**Target:** `delivery_risk`, where `0` means low risk and `1` means high risk.

**Intended action:** Use a high-risk result as a human-reviewed alert. It must
not automatically penalize a supplier, reroute freight, or change a shipment.

Useful operational measures for a future real-data pilot include the share of
actual delays detected (high-risk recall), the share of alerts that correspond
to actual delays (high-risk precision), intervention workload, and delivery
outcomes after intervention. The current sample does not support estimating
financial savings or operational impact.

## 2. Dataset and data understanding

The supplied `data/supply_chain_data.csv` has 438 rows and 16 columns:
15 candidate input fields and the binary target. It contains four categorical
shipment descriptors (product category, supplier region, destination region,
and shipping mode), numeric order and logistics measures, the `order_id`
identifier, and `delivery_risk`.

The data is **synthetically generated** by `generate_sample_dataset` in
`main.py`. The target is constructed from a weighted risk score, noise, and a
threshold, then the classes are balanced by down-sampling. In simplified form,
the score weights lower supplier reliability, historical delay rate, weather
risk, port congestion, customs time, and distance. Thus, target relationships
are partly built into the generator and should not be generalized to real
freight operations.

After cleaning, the dataset has 219 examples in each target class. The stored
EDA summary reports zero duplicate rows removed, zero rows without a target,
and zero missing values in the supplied file. These are observations about
this sample, not assumptions made about future datasets.

## 3. Data preprocessing and exploratory analysis

The training workflow:

1. Removes exact duplicate records and rows without a target.
2. Separates a stratified 80/20 train/test split with a fixed random seed
   (`random_state=42`): 350 training and 88 test records.
3. Excludes identifier fields such as `order_id` from model inputs.
4. Learns numeric medians, categorical modes, category encoding, and numeric
   scaling from training data only. It reuses this fitted state for test and
   prediction data.
5. One-hot encodes categorical inputs, standardizes numeric features, and
   aligns transformed features to the fitted training columns.

This ordering prevents the held-out test data from influencing learned
preprocessing. During model search, preprocessing is also fit within each
cross-validation fold.

The generated EDA summary and plots include class distribution, numeric
descriptive statistics, an IQR outlier review, and numeric correlations.
The largest reported numeric correlations with the target are supplier
reliability score (-0.3876), historical delay rate (0.3391), customs
clearance time (0.2371), and distance (0.2013). These describe associations
in the synthetic sample; they do not establish causes.

IQR screening flagged 49 distance values, 19 customs-clearance values, 10
historical-delay-rate values, and a small number in other numeric columns.
The pipeline reports these for investigation rather than deleting them
automatically, since extreme route or clearance values may be legitimate.

## 4. Model selection and implementation

The training code supports a class-weighted Decision Tree as its default and
an optional `--tune` search comparing Decision Trees, RBF SVCs, and Gradient
Boosting classifiers. The checked-in metrics identify an RBF SVC with
`C=0.1` and no class weighting as the selected model. It was chosen by
five-fold stratified training cross-validation using accuracy as the search
score, with mean CV accuracy of 76.3%.

The model is stored separately from the fitted preprocessing state in
`models/delivery_risk_model.pkl` and `models/preprocessor.pkl`. The predictor
applies the saved transformations before producing a class prediction and
risk probability. The default classification threshold is the estimator's
standard threshold; it has not been optimized against stakeholder costs.

## 5. Evaluation and validation

The following values are reported in `models/metrics.json` for the currently
checked-in tuned model:

| Measure | Result | Interpretation |
| --- | ---: | --- |
| Held-out accuracy | 78.4% | 69 of 88 test predictions were correct. |
| Balanced accuracy | 78.4% | Average recall across the two equally represented classes. |
| ROC-AUC | 0.835 | Ranking discrimination across possible thresholds; not probability calibration. |
| High-risk precision | 77.8% | 35 of 45 high-risk alerts were truly high risk. |
| High-risk recall | 79.5% | 35 of 44 actual high-risk cases were detected. |
| High-risk F1 | 78.7% | Harmonic mean of high-risk precision and recall. |
| Five-fold training CV F1 | 0.777 +/- 0.019 | Fold mean and standard deviation on the training subset. |
| Majority-class baseline accuracy | 50.0% | Reference score on the balanced test split. |

Confusion matrix (rows are actual; columns are predicted):

|  | Predicted low risk | Predicted high risk |
| --- | ---: | ---: |
| Actual low risk | 34 | 10 |
| Actual high risk | 9 | 35 |

The 9 false negatives are high-risk cases not identified by the current
threshold; the 10 false positives are low-risk cases flagged for review.
Which error is more costly depends on operational capacity and delay impact.
Use stakeholder feedback and real outcomes to set any future alert threshold.

The 88-record holdout is small, synthetic, and has been reused during model
iteration. It should not be interpreted as an independent production estimate.
The 50% baseline is particularly simple because the example classes are
balanced; it is not necessarily an appropriate baseline for real operations.

## 6. Application and code organization

The project is organized by responsibility:

| Path | Responsibility |
| --- | --- |
| `main.py` | CLI entry point, data generation, EDA artifacts, and training/prediction orchestration. |
| `src/data_preprocessing.py` | Cleaning, train-only feature transformation, and persistence of preprocessing state. |
| `src/model_training.py` | Model fitting/search, cross-validation, evaluation, and plots. |
| `src/predict.py` | Single-record and batch prediction using saved artifacts. |
| `web/app.py` | Flask page and prediction API routes. |
| `web/templates/index.html` | Responsive user interface for the shipment-risk workflow. |
| `data/` | Example CSV input. |
| `models/` | Model, preprocessor, metrics, EDA summary, and generated evaluation visuals. |
| `docs/` | This report, presentation deck, and presentation notes/Q&A. |

The web application supports interactive shipment assessment and provides
`POST /predict`, `POST /predict_batch`, `POST /predict_batch_file`, and
`GET /health` endpoints.

## 7. Reproduction and use

The repository declares Python 3.14.x in `pyproject.toml` and
`.python-version`. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --mode train --tune
python main.py --mode predict --input sample_input.json
```

To start the browser application:

```powershell
Set-Location .\web
..\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`. Keep the server terminal open while using the
application. Training with `--tune` may regenerate model and metric artifacts;
the checked-in `models/metrics.json` records the results cited in this report.

## 8. Business insights and recommendations

For a controlled real-data pilot:

1. Define “late delivery” with business owners (for example, actual arrival
   later than the promised date by an agreed tolerance).
2. Ensure every input is known at the intended prediction time; remove
   post-outcome fields and review target leakage.
3. Prioritize human checks of supplier reliability, delay history, customs,
   congestion, weather, and route conditions only when supported by current
   operating data.
4. Select an alert threshold based on the relative cost of missed delays,
   unnecessary escalations, and available planner capacity.
5. Track precision, recall, calibration, alert volume, and outcomes by time,
   route, supplier, and shipping mode. Review drift and investigate disparate
   error rates before expanding use.

These are validation steps, not proven benefits from this synthetic example.

## 9. Limitations and next steps

- The dataset and labels are synthetic, not verified company shipment records.
- Only 438 examples are available, with an 88-record held-out test set.
- Labels are derived from a weighted generator formula and balancing, so the
  apparent signal is partly artificial.
- Repeated experiments use the same fixed holdout; an untouched temporal
  validation set is needed for stronger evidence.
- The project reports discrimination metrics but does not evaluate probability
  calibration, threshold economics, causal effects, or fairness.
- Human review and input validation should be part of any operational pilot.

Next steps are to obtain approved, representative, time-stamped shipment
history; document label and feature availability rules; conduct temporal and
segment-based validation; calibrate probabilities and choose an action
threshold with planners; and monitor outcomes after deployment.

## Conclusion

Freight Signal demonstrates an end-to-end delivery-risk workflow—from EDA and
training through evaluation to an interactive Flask prediction application.
Its current results are suitable for a capstone demonstration only. The
essential next step is validation on representative real shipment data before
making operational claims or decisions.
