# Supply Chain Delivery Risk Prediction

This capstone project estimates whether an order is at **high risk of delivery
delay** using order, supplier, route, and logistics information. It includes a
repeatable data-analysis and model-training workflow, saved evaluation
artifacts, a command-line prediction tool, and a Flask web application.

> **Important:** The checked-in CSV is a small sample dataset, not verified
> company shipment history. Treat the current scores as a demonstration of the
> workflow, not evidence of real-world performance. Replace it with approved,
> representative historical data before using predictions for operational
> decisions.

## Business question

Given information available when an order is assessed, can we identify
shipments that may need extra attention? The intended users are supply-chain
planners and logistics teams. A high-risk prediction can prompt a human to
check a supplier, route, port, weather conditions, or customs plan; it should
not automatically penalize a supplier or change a shipment.

The target is `delivery_risk`:

| Value | Meaning |
| --- | --- |
| `0` | Low delivery risk |
| `1` | High delivery risk |

For a production dataset, define the target from a documented business rule,
such as an order arriving more than a specified number of days after its
promised date. Confirm that all prediction features are known at the time the
prediction would be made. Do not include outcomes or post-delivery information.

## Rubric map

| Rubric area | Where this project demonstrates it |
| --- | --- |
| Problem and business understanding | This section states the business question, users, target, and intended action. |
| Data understanding | `data/supply_chain_data.csv`, the schema below, and `models/eda_summary.json` describe the example data. |
| Cleaning and preprocessing | `src/data_preprocessing.py` removes duplicates, excludes identifier columns, imputes missing feature values from training data, encodes categories, and scales numeric features. |
| Exploratory analysis | Training produces a class-distribution plot, correlation heatmap, and numeric/data-quality summary in `models/`. |
| Feature engineering and leakage | `order_id` is excluded; train/test splitting happens before learned preprocessing is fitted. |
| Model choice and implementation | `src/model_training.py` trains an interpretable Decision Tree by default; optional cross-validated selection compares decision trees, SVC, and gradient boosting. |
| Evaluation and validation | A stratified held-out test set, five-fold training-only cross-validation, confusion matrix, classification report, accuracy, balanced accuracy, and ROC-AUC are saved. |
| Business insights | Use the feature-importance file and high-risk precision/recall to decide which shipment checks to prioritize; see the guidance below. |
| Deployment | `web/app.py` serves a browser interface and JSON endpoints. |
| Documentation and presentation | This README contains setup steps, explanations, caveats, and a suggested presentation outline. |

## Data schema

The provided example has 438 rows and 16 columns. It currently has no missing
values or duplicate rows and has equal counts of low- and high-risk examples.
These properties describe this sample only; always inspect a replacement
dataset using the generated EDA summary.

| Column | Type | Description |
| --- | --- | --- |
| `order_id` | Identifier | Order reference; deliberately excluded from model features. |
| `product_category` | Categorical | Product group. |
| `supplier_region` | Categorical | Supplier location group. |
| `destination_region` | Categorical | Delivery location group. |
| `shipping_mode` | Categorical | Air, sea, road, or rail. |
| `order_quantity` | Numeric | Number of units in the order. |
| `unit_price` | Numeric | Unit price in the dataset's currency. |
| `order_month` | Integer | Order month, 1–12. |
| `order_day_of_week` | Integer | Day of week, 0–6. |
| `supplier_reliability_score` | Numeric | Supplier reliability score; larger values indicate more reliable suppliers. |
| `historical_delay_rate` | Numeric | Historical fraction of delayed orders. |
| `weather_risk_score` | Numeric | Weather-related risk score. |
| `port_congestion_level` | Numeric | Port congestion score. |
| `customs_clearance_time` | Numeric | Customs clearance time, in the source data's units. |
| `distance_km` | Numeric | Route distance in kilometres. |
| `delivery_risk` | Binary target | `0` = low risk; `1` = high risk. |

The example `delivery_risk` labels were generated from a weighted score and
noise. Therefore, apparent relationships learned from this sample may simply
reflect the synthetic label-generation formula.

## Setup and run

Use Python 3.8 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train using the included CSV:

```powershell
python main.py --mode train
```

Tune decision-tree hyperparameters for cross-validation accuracy and save the
best model:

```powershell
python main.py --mode train --tune
```

Generate a new synthetic dataset for experimentation (the output may contain
fewer than the requested number of rows because the generator balances the
classes):

```powershell
python main.py --mode generate --data data/supply_chain_data.csv
```

Predict one order:

```powershell
python main.py --mode predict --input sample_input.json
```

### Run the backend and frontend

From the project root, run `app.py` from the `web` folder:

```powershell
Set-Location .\web
..\.venv\Scripts\python.exe app.py
```

This starts the Flask backend, which also serves the HTML frontend; there is
no separate frontend server to start. Keep the terminal open, then open
`http://127.0.0.1:5000` in your browser. The page displays saved evaluation
metrics and supports interactive shipment predictions.

The backend API is available from the same server:

- `POST http://127.0.0.1:5000/predict` for one JSON prediction.
- `POST http://127.0.0.1:5000/predict_batch` for a JSON batch.
- `POST http://127.0.0.1:5000/predict_batch_file` for an uploaded CSV/JSON batch.
- `GET http://127.0.0.1:5000/health` for model/server status.

## What training does

1. Loads the CSV, removes exact duplicate rows, and removes rows without a
   target label.
2. Writes the data-quality summary, numeric summary, target-class chart, and
   numeric correlation heatmap.
3. Splits raw data into stratified training and test subsets.
4. Fits missing-value replacements, one-hot category encoding, and numeric
   scaling **using training data only**, then applies that saved state to test
   and future prediction data. Unseen categories become all-zero category
   columns rather than causing a prediction error.
5. Excludes identifier columns such as `order_id`, fits a class-weighted
   Decision Tree by default, and optionally compares decision trees, SVC, and
   gradient boosting using five-fold training cross-validation accuracy
   (`--tune`). Preprocessing is fitted separately inside each search fold.
   The held-out test set is not used to choose the model or hyperparameters.
6. Reports held-out test results and five-fold cross-validation on the
   training subset. Each cross-validation fold fits its own preprocessor.
7. Saves the model, preprocessor, metrics, feature importance, and plots under
   `models/`.

## Understanding the results

- **Confusion matrix:** rows are the actual class and columns are the
  predicted class. A false negative is a high-risk order predicted as low
  risk; a false positive is an order flagged high risk that was actually low
  risk.
- **High-risk recall:** the fraction of actual high-risk orders detected. Use
  this when missing a risky order is especially costly.
- **High-risk precision:** the fraction of high-risk alerts that were truly
  high risk. Use this when unnecessary interventions are costly.
- **F1-score:** balances precision and recall. Five-fold evaluation reports
  high-risk F1; when `--tune` is used, hyperparameters are selected by
  cross-validation accuracy.
- **Balanced accuracy:** averages recall for both classes, so it is more
  informative than plain accuracy when real-world classes are imbalanced.
- **ROC-AUC:** measures how well the model ranks positive examples above
  negative ones across probability thresholds; it is not a calibration score.
- **Cross-validation mean and standard deviation:** summarize training-set
  F1 across five folds. A large standard deviation suggests results vary
  depending on which examples are used.
- **Feature importance:** indicates how much the fitted tree used each
  feature. It does not prove that a feature causes delivery delays.

Use the high-risk precision/recall and business costs to choose an alert
threshold with logistics stakeholders. The current application uses the
classifier's default decision threshold and should be treated as decision
support, not an automatic decision-maker.

### Latest sample run

The latest tuned run selected an RBF SVC (`C=0.1`, no class weighting) using
five-fold cross-validation on 350 training rows. It achieved **76.3% mean CV
accuracy** and **78.4% accuracy** with **83.5% ROC-AUC** on the 88-row
held-out split. The earlier tuned decision tree scored 73.9% on that split.
The selected model and metrics are recorded in `models/metrics.json`.

This small synthetic dataset cannot guarantee better results on real
shipments. Repeated tuning attempts have reused the same held-out split, so
its score should not be treated as a fresh, independent estimate. Compare
with the majority-class baseline in `models/metrics.json`; on this balanced
sample, its accuracy is 50%.

## Generated artifacts

| Artifact | Purpose |
| --- | --- |
| `models/eda_summary.json` | Dataset size, duplicates, missingness, class balance, numeric summary, IQR outlier counts, and numeric correlations with the target. |
| `models/target_distribution.png` | Counts of low- and high-risk records. |
| `models/correlation_heatmap.png` | Correlations between numeric columns. Correlation is not causation. |
| `models/delivery_risk_model.pkl` | Fitted classifier selected by training (decision tree by default). |
| `models/preprocessor.pkl` | Training-derived imputation, encoding, scaling, and feature-column state. |
| `models/metrics.json` | Held-out metrics, classification report, confusion matrix, and cross-validation scores. |
| `models/feature_importance.csv` | Decision-tree feature importance ranking (tree models only). |
| `models/feature_importance.png` | Feature-importance chart (tree models only). |
| `models/confusion_matrix.png` | Held-out confusion-matrix chart. |
| `models/decision_tree.png` | First levels of the fitted tree for explanation (tree models only). |
These pickle files are Python artifacts and should only be loaded from a
trusted source. Retrain the model after changing the data or preprocessing.

## Practical recommendations to discuss

When a tree model is selected, use its feature-importance ranking together
with actual operating data to prioritize recommendations. For example:

1. Review supplier performance and contingency plans for suppliers with
   recurring delays or lower reliability scores.
2. Monitor congestion, adverse weather, and customs clearance for routes
   flagged as high risk; consider alternative routes or earlier dispatch when
   feasible.
3. Compare high-risk alerts with actual delay outcomes each month and adjust
   the intervention threshold to balance missed delays against unnecessary
   escalations.
4. Track performance by route, supplier, and shipping mode to identify groups
   where the model is unreliable or where operational interventions help.

These are hypotheses to validate with real shipment outcomes, not conclusions
established by the synthetic sample.

Potential IQR outliers are reported in `models/eda_summary.json`, not silently
removed. A very long route or customs clearance time can be a valid, important
shipment, so investigate flagged values against source records before deciding
whether to correct, cap, or exclude them.

## Suggested presentation flow

1. State the business problem and define what counts as a high-risk order.
2. Describe the data, source, time period, target definition, and limitations.
3. Show cleaning, the target distribution, and one or two EDA findings.
4. Explain how identifiers and leakage were handled, then justify the
   selected model.
5. Present test results with the confusion matrix, high-risk precision/recall,
   and cross-validation variability.
6. Demonstrate one prediction in the web application.
7. Recommend a human-reviewed pilot, explain monitoring, and state what data
   is still needed before deployment.
#   M L - C a p s t o n e  
 