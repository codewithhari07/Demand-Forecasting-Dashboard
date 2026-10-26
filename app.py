from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from project_utils import FEATURE_COLUMNS, make_future_feature_row

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
MODEL_PATH = ROOT / "models" / "best_demand_model.pkl"
CLASSIFIER_PATH = ROOT / "models" / "high_demand_classifier.pkl"


st.set_page_config(
    page_title="Retail Sales & Demand Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    data = data.dropna(subset=["Date", "Demand"]).copy()
    # Match categorical types used by training, especially numeric store/product IDs.
    for col in ["Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"]:
        data[col] = data[col].astype(str)
    return data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)


@st.cache_resource
def load_model_bundle():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_classifier_bundle():
    with open(CLASSIFIER_PATH, "rb") as file:
        return pickle.load(file)


def format_number(value, decimals=0):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


if not DATA_PATH.exists():
    st.error("Dataset not found. Put your CSV at data/sales_data.csv.")
    st.stop()

if not MODEL_PATH.exists():
    st.warning(
        "The trained model pickle was not found. Open a terminal in the project "
        "folder and run: python train_model.py"
    )
    st.stop()

df = load_data()
bundle = load_model_bundle()
model = bundle["model"]
classifier_bundle = load_classifier_bundle() if CLASSIFIER_PATH.exists() else None
classifier_model = classifier_bundle["model"] if classifier_bundle else None

st.title("Retail Sales & Demand Forecasting Dashboard")
st.caption(
    "Retail analytics prototype using historical store-product demand, "
    "forecasting models, and inventory indicators."
)

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Demand Forecast", "Inventory Insights", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Data through: {df['Date'].max().date()}")
st.sidebar.caption(f"Selected model: {bundle['best_model_name']}")


if page == "Sales Overview":
    st.subheader("Sales and Demand Overview")

    total_demand = df["Demand"].sum()
    total_units_sold = df["Units Sold"].sum()
    n_stores = df["Store ID"].nunique()
    n_products = df["Product ID"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Recorded Demand", format_number(total_demand))
    c2.metric("Total Units Sold", format_number(total_units_sold))
    c3.metric("Stores", format_number(n_stores))
    c4.metric("Products", format_number(n_products))

    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    chosen_category = st.selectbox("Filter by product category", categories)
    overview = df if chosen_category == "All" else df[df["Category"] == chosen_category]

    daily_demand = (
        overview.groupby("Date", as_index=False)["Demand"]
        .sum()
        .sort_values("Date")
    )
    fig = px.line(
        daily_demand,
        x="Date",
        y="Demand",
        title="Daily Demand Trend",
        labels={"Demand": "Recorded demand (units)", "Date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)

    top_items = (
        overview.groupby("Product ID", as_index=False)["Demand"]
        .sum()
        .sort_values("Demand", ascending=False)
        .head(10)
    )
    fig2 = px.bar(
        top_items,
        x="Product ID",
        y="Demand",
        title="Top 10 Products by Recorded Demand",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Recorded demand and units sold are dataset fields. They should not be "
        "interpreted as V-Mart's actual business results."
    )


elif page == "Demand Forecast":
    st.subheader("Forecast Demand for a Store and Product")
    st.write(
        "Choose a store-product combination, then set assumptions for future price, "
        "discount, promotion, competitor pricing, weather, and epidemic conditions. "
        "The model forecasts the selected combination recursively."
    )

    store_options = sorted(df["Store ID"].dropna().unique().tolist())
    store_id = st.selectbox("Store", store_options)

    products_for_store = sorted(
        df.loc[df["Store ID"] == store_id, "Product ID"].dropna().unique().tolist()
    )
    product_id = st.selectbox("Product", products_for_store)

    selected = (
        df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
        .sort_values("Date")
        .copy()
    )
    latest = selected.iloc[-1]
    history_default_weather = latest["Weather Condition"]
    weather_options = sorted(df["Weather Condition"].dropna().unique().tolist())
    weather_index = (
        weather_options.index(history_default_weather)
        if history_default_weather in weather_options
        else 0
    )

    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
            price = st.number_input(
                "Planned selling price",
                min_value=0.01,
                max_value=10000.0,
                value=float(latest["Price"]),
                step=1.0,
            )
        with col2:
            discount = st.number_input(
                "Discount (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(latest["Discount"]),
                step=1.0,
            )
            competitor_pricing = st.number_input(
                "Competitor price",
                min_value=0.01,
                max_value=10000.0,
                value=float(latest["Competitor Pricing"]),
                step=1.0,
            )
        with col3:
            promotion = st.checkbox("Promotion active", value=bool(latest["Promotion"]))
            weather = st.selectbox(
                "Assumed weather",
                weather_options,
                index=weather_index,
            )
            epidemic = st.checkbox(
                "Epidemic condition active", value=bool(latest["Epidemic"])
            )

        submitted = st.form_submit_button("Generate forecast")

    current_forecast_key = (
        str(store_id),
        str(product_id),
        int(forecast_days),
        round(float(price), 2),
        round(float(discount), 2),
        round(float(competitor_pricing), 2),
        int(promotion),
        str(weather),
        int(epidemic),
    )

    if submitted:
        if len(selected) < 28:
            st.error("At least 28 days of data are needed for this store-product combination.")
        else:
            history = selected["Demand"].astype(float).tolist()
            forecast_rows = []
            last_date = pd.Timestamp(selected["Date"].max())

            for offset in range(1, int(forecast_days) + 1):
                forecast_date = last_date + pd.Timedelta(days=offset)
                feature_dict = make_future_feature_row(
                    forecast_date=forecast_date,
                    store_id=str(store_id),
                    product_id=str(product_id),
                    category=str(latest["Category"]),
                    region=str(latest["Region"]),
                    price=float(price),
                    discount=float(discount),
                    competitor_pricing=float(competitor_pricing),
                    promotion=int(promotion),
                    weather_condition=str(weather),
                    epidemic=int(epidemic),
                    demand_history=history,
                )
                X_future = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)
                prediction = max(0.0, float(model.predict(X_future)[0]))
                forecast_row = {
                    "Date": forecast_date,
                    "Store ID": store_id,
                    "Product ID": product_id,
                    "Predicted Demand": prediction,
                }
                if classifier_model is not None:
                    high_demand_probability = float(classifier_model.predict_proba(X_future)[0, 1])
                    high_demand_class = int(classifier_model.predict(X_future)[0])
                    forecast_row["High Demand Probability (%)"] = high_demand_probability * 100
                    forecast_row["High Demand Alert"] = "High demand" if high_demand_class == 1 else "Normal demand"
                forecast_rows.append(forecast_row)
                # Recursive multi-step forecast: predicted demand feeds later lag features.
                history.append(prediction)

            forecast_df = pd.DataFrame(forecast_rows)
            st.session_state["forecast_df"] = forecast_df
            st.session_state["forecast_key"] = current_forecast_key

    saved_forecast = st.session_state.get("forecast_df")
    saved_key = st.session_state.get("forecast_key")
    if saved_forecast is not None and saved_key == current_forecast_key:
        forecast_df = saved_forecast
        average_forecast = forecast_df["Predicted Demand"].mean()
        total_forecast = forecast_df["Predicted Demand"].sum()

        if classifier_bundle is not None:
            m1, m2, m3, m4 = st.columns(4)
        else:
            m1, m2, m3 = st.columns(3)
        m1.metric("Forecast period", f"{len(forecast_df)} days")
        m2.metric("Total predicted demand", format_number(total_forecast, 1))
        m3.metric("Average daily predicted demand", format_number(average_forecast, 1))
        if classifier_bundle is not None:
            next_day_probability = float(forecast_df.iloc[0]["High Demand Probability (%)"])
            m4.metric("Day-1 high-demand probability", f"{next_day_probability:.1f}%")

        historical_plot = selected.tail(90)[["Date", "Demand"]].copy()
        historical_plot = historical_plot.rename(columns={"Demand": "Units"})
        historical_plot["Series"] = "Historical demand"

        future_plot = forecast_df[["Date", "Predicted Demand"]].rename(
            columns={"Predicted Demand": "Units"}
        )
        future_plot["Series"] = "Forecast demand"

        line_data = pd.concat([historical_plot, future_plot], ignore_index=True)
        fig = px.line(
            line_data,
            x="Date",
            y="Units",
            color="Series",
            title=f"Historical and Forecast Demand — {store_id} / {product_id}",
            labels={"Units": "Demand (units)", "Date": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            forecast_df.round({"Predicted Demand": 2, "High Demand Probability (%)": 1}),
            use_container_width=True,
        )

        st.download_button(
            "Download forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{store_id}_{product_id}.csv",
            mime="text/csv",
        )
        st.warning(
            "Forecast limitation: future price, discount, promotion, competitor price, "
            "weather, and epidemic status are held at your selected assumptions. "
            "Demand lags are updated using the model's own earlier predictions, so "
            "uncertainty can grow over a longer forecast horizon."
        )
    else:
        st.info("Choose your assumptions and click Generate forecast to view predictions.")


elif page == "Inventory Insights":
    st.subheader("Inventory and Replenishment Indicators")

    last_date = df["Date"].max()
    window_start = last_date - pd.Timedelta(days=29)
    recent = df[df["Date"].between(window_start, last_date)].copy()
    recent["Sales Value"] = recent["Units Sold"] * recent["Price"]

    group_keys = ["Store ID", "Product ID"]
    latest_stock = (
        df.sort_values("Date")
        .groupby(group_keys, as_index=False)
        .tail(1)
        .copy()
    )
    recent_sales = (
        recent.groupby(group_keys, as_index=False)
        .agg(
            units_sold_30d=("Units Sold", "sum"),
            sales_value_30d=("Sales Value", "sum"),
            orders_30d=("Units Ordered", "sum"),
        )
    )
    inventory = latest_stock.merge(recent_sales, on=group_keys, how="left")
    inventory["avg_daily_units_sold"] = inventory["units_sold_30d"] / 30.0
    inventory["inventory_value_at_list_price"] = (
        inventory["Inventory Level"] * inventory["Price"]
    )
    inventory["days_of_cover"] = np.where(
        inventory["avg_daily_units_sold"] > 0,
        inventory["Inventory Level"] / inventory["avg_daily_units_sold"],
        np.nan,
    )
    inventory["stock_to_sales_ratio"] = np.where(
        inventory["sales_value_30d"] > 0,
        inventory["inventory_value_at_list_price"] / inventory["sales_value_30d"],
        np.nan,
    )
    # Proxy only: opening stock and a reconciled movement ledger are not provided.
    denom = inventory["units_sold_30d"] + inventory["Inventory Level"]
    inventory["approx_sell_through_pct"] = np.where(
        denom > 0, inventory["units_sold_30d"] / denom * 100, np.nan
    )
    inventory["Status"] = np.select(
        [
            inventory["avg_daily_units_sold"] <= 0,
            inventory["days_of_cover"] < 7,
        ],
        ["No recent sales", "Low stock (<7 days cover)"],
        default="Stock cover >=7 days",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Products with low stock cover", int((inventory["Status"] == "Low stock (<7 days cover)").sum()))
    c2.metric("Average days of cover", format_number(inventory["days_of_cover"].replace([np.inf, -np.inf], np.nan).mean(), 1))
    c3.metric("Inventory value at list price", format_number(inventory["inventory_value_at_list_price"].sum(), 2))

    category_filter = st.selectbox(
        "Category filter",
        ["All"] + sorted(inventory["Category"].dropna().unique().tolist()),
    )
    store_filter = st.selectbox(
        "Store filter",
        ["All"] + sorted(inventory["Store ID"].dropna().unique().tolist()),
    )
    shown = inventory.copy()
    if category_filter != "All":
        shown = shown[shown["Category"] == category_filter]
    if store_filter != "All":
        shown = shown[shown["Store ID"] == store_filter]

    display_columns = [
        "Store ID", "Product ID", "Category", "Inventory Level", "Units Sold",
        "units_sold_30d", "Units Ordered", "orders_30d", "days_of_cover",
        "stock_to_sales_ratio", "approx_sell_through_pct", "Status",
    ]
    st.dataframe(
        shown[display_columns].rename(
            columns={
                "Inventory Level": "Latest inventory",
                "Units Sold": "Latest-day units sold",
                "units_sold_30d": "Units sold (30d)",
                "Units Ordered": "Latest-day units ordered",
                "orders_30d": "Units ordered (30d)",
                "days_of_cover": "Days of cover",
                "stock_to_sales_ratio": "Stock-to-sales ratio",
                "approx_sell_through_pct": "Approx. sell-through (%)",
            }
        ).round(2),
        use_container_width=True,
    )

    st.caption(
        f"Inventory snapshot date: {last_date.date()}. Stock-to-sales ratio uses latest "
        "inventory value at listed price divided by sales value over the previous 30 days. "
        "Sell-through is an approximation: a proper calculation requires opening stock "
        "and a reconciled record of receipts/returns. The 7-day cover threshold is a demo rule."
    )
    st.download_button(
        "Download inventory analysis CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="inventory_insights.csv",
        mime="text/csv",
    )


elif page == "Model Performance":
    st.subheader("Demand Forecasting — Regression Results")
    st.write(f"Selected forecasting model: **{bundle['best_model_name']}**")
    st.write(f"Validation period begins: **{bundle['validation_start_date']}**")
    st.write(f"Trained using data through: **{bundle['trained_through_date']}**")
    results = pd.DataFrame(bundle["metrics"]).sort_values("MAE")
    st.dataframe(results.round(4), use_container_width=True)

    if not results.empty:
        fig = px.bar(
            results,
            x="model",
            y="MAE",
            title="Regression Model Comparison — Lower MAE Is Better",
            labels={"model": "Model", "MAE": "Mean Absolute Error (units)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if classifier_bundle is not None:
        st.divider()
        st.subheader("High-Demand Classification — Logistic Regression")
        clf_metrics = classifier_bundle["metrics"]
        st.write(
            "This is a separate classification task, not a direct comparison with the "
            "regression models. High demand is defined as Demand >= "
            f"**{classifier_bundle['high_demand_threshold']:.2f}** based on the training-set 75th percentile."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{clf_metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{clf_metrics['precision']:.3f}")
        c3.metric("Recall", f"{clf_metrics['recall']:.3f}")
        c4.metric("F1 score", f"{clf_metrics['f1']:.3f}")
        c5.metric("ROC-AUC", f"{clf_metrics['roc_auc']:.3f}")
        st.caption(
            "The forecast page also shows Logistic Regression's estimated probability "
            "of high demand for each forecast day. This probability is model-estimated, "
            "not a guarantee."
        )

    st.markdown(
        """
        **Regression metrics**
        - **MAE:** average absolute difference between actual and predicted demand, in units.
        - **RMSE:** penalizes larger prediction errors more heavily.
        - **R²:** indicates how much variance is explained by the predictions; it can be negative.

        Regression models are ranked by MAE on a chronological holdout. Logistic Regression
        classifies high-demand vs. other days and is assessed with classification metrics,
        so its metrics should not be compared directly with regression MAE/RMSE.
        Validation is a rolling one-day-ahead evaluation; the recursive multi-day forecast
        may have larger errors.
        """
    )
