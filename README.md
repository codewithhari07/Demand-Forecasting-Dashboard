# Retail Sales & Demand Forecasting Dashboard

An educational portfolio project using the supplied `sales_data.csv`. It trains two demand-regression models (LightGBM Regressor and Random Forest Regressor), selects the best regression model by validation MAE, and trains a separate Logistic Regression classifier to flag high-demand observations. Model pipelines are saved with Python `pickle` and used in the Streamlit dashboard. Deployment link: https://sales-demand-forecasting-dashboard.streamlit.app/

## Dataset

Put the supplied CSV at `data/sales_data.csv`. Expected columns:

- `Date`, `Store ID`, `Product ID`, `Category`, `Region`
- `Inventory Level`, `Units Sold`, `Units Ordered`
- `Price`, `Discount`, `Weather Condition`, `Promotion`
- `Competitor Pricing`, `Seasonality`, `Epidemic`, `Demand`

The regression target is `Demand`. The models do not use same-day `Units Sold`, `Inventory Level`, or `Units Ordered` as predictors because those may not be known when forecasting demand. Instead, lag and rolling features are generated from previous `Demand` observations for each store-product combination.

## Why there are two model tasks

- **LightGBM Regressor and Random Forest Regressor** predict a numeric demand value. Their validation results can be compared using MAE, RMSE, and R². The regressor with the lowest MAE is saved as the best demand forecasting model.
- **Logistic Regression Classifier** predicts a category, not a numeric demand value. In this project, it classifies a row as `High Demand` when its `Demand` is at or above the 75th percentile calculated on the training split. The threshold is kept fixed for validation and deployment. The classifier is evaluated separately with accuracy, precision, recall, F1, and ROC-AUC.

Do not compare Logistic Regression classification metrics directly with regression MAE/RMSE; these metrics measure different tasks.

## Setup on Windows (VS Code)

1. Extract the project ZIP and open this folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Train both tasks:

   ```powershell
   python train_model.py
   ```

   The script uses the latest 20% of unique dates as a chronological validation holdout. It trains LightGBM and Random Forest regressors, selects the lower-MAE model, evaluates Logistic Regression separately as a high-demand classifier, then refits the selected models on all eligible historical rows for the dashboard.

6. Launch the dashboard:

   ```powershell
   streamlit run app.py
   ```

## Generated model and result files

- `models/best_demand_model.pkl` — best complete regression pipeline (preprocessing + estimator) plus metadata.
- `models/high_demand_classifier.pkl` — complete Logistic Regression classification pipeline plus threshold and validation metrics.
- `outputs/model_comparison.csv` — LightGBM and Random Forest regression metrics.
- `outputs/classification_metrics.csv` — high-demand classification metrics.
- `outputs/validation_predictions.csv` — actual and predicted demand for the best regression model on the validation holdout.
- `outputs/classification_validation_predictions.csv` — actual and predicted high-demand labels and probabilities for the validation holdout.

## Dashboard pages

- **Sales Overview:** recorded demand trend and top products.
- **Demand Forecast:** recursively forecasts future demand and, where the classifier pickle is present, displays the estimated high-demand probability for each forecast day.
- **Inventory Insights:** demonstrates stock cover and inventory KPIs available from the supplied columns. Sell-through is labelled approximate because the dataset does not provide a full opening-stock/receipts/returns ledger.
- **Model Performance:** compares regression models by MAE and separately displays Logistic Regression classification metrics.

## Evaluation caveats

- The holdout is chronological, not randomly shuffled. It measures rolling one-day-ahead predictions where prior actual demand is available for later validation rows.
- A recursive 7–60 day forecast uses its own previous predictions as lag inputs, so errors may accumulate across the forecast horizon.
- The classifier's probability is a model estimate rather than a guarantee. It predicts whether the defined demand threshold will be met.
- Future price, discount, competitor price, promotion, weather, and epidemic assumptions are kept at the values selected in the dashboard. Seasonality and calendar features change based on forecast date.
- Do not describe the provided dataset as actual V-Mart internal data.
