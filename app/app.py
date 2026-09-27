
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "APL_Logistics.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_random_forest.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_preprocessor.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_metadata.pkl"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )



    # ========================================================
    # CREATE ENGINEERED FEATURES
    # ========================================================

    # Shipping Pressure Index
    data["Shipping Pressure Index"] = (
        data["Order Item Quantity"]
        / (data["Days for shipment (scheduled)"] + 1)
    )

    # Shipping mode flags
    data["First_Class_Flag"] = (
        data["Shipping Mode"] == "First Class"
    ).astype(int)

    data["Second_Class_Flag"] = (
        data["Shipping Mode"] == "Second Class"
    ).astype(int)

    data["Same_Day_Flag"] = (
        data["Shipping Mode"] == "Same Day"
    ).astype(int)

    data["Standard_Class_Flag"] = (
        data["Shipping Mode"] == "Standard Class"
    ).astype(int)

    # Regional congestion
    region_counts = data["Order Region"].value_counts()

    data["Regional_Congestion_Indicator"] = (
        data["Order Region"].map(region_counts)
    )

    # Order complexity
    quantity_norm = (
        data["Order Item Quantity"]
        / data["Order Item Quantity"].max()
    )

    discount_norm = (
        data["Order Item Discount Rate"]
        / data["Order Item Discount Rate"].max()
    )

    pressure_norm = (
        data["Shipping Pressure Index"]
        / data["Shipping Pressure Index"].max()
    )

    data["Order_Complexity_Score"] = (
        quantity_norm
        + discount_norm
        + pressure_norm
    ) / 3

    return data

@st.cache_data
def load_dashboard_predictions():

    predictions_path = os.path.join(
        BASE_DIR,
        "data",
        "dashboard_predictions.csv"
    )

    return pd.read_csv(predictions_path)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = joblib.load(METADATA_PATH)

    return model, preprocessor, metadata


df = load_data()
model, preprocessor, metadata = load_model()
dashboard_predictions = load_dashboard_predictions()

model_columns = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "model_input_columns.pkl"
    )
)

# ============================================================
# TITLE
# ============================================================

st.title("📦 Late Delivery Risk Prediction")
st.markdown(
    """
    **Machine Learning–based Late Delivery Risk Prediction
    in Global Supply Chain Operations**
    """
)

st.divider()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

shipping_modes = sorted(
    df["Shipping Mode"].dropna().unique()
)

markets = sorted(
    df["Market"].dropna().unique()
)

customer_segments = sorted(
    df["Customer Segment"].dropna().unique()
)

selected_shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    shipping_modes,
    default=shipping_modes
)

selected_markets = st.sidebar.multiselect(
    "Market",
    markets,
    default=markets
)

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    customer_segments,
    default=customer_segments
)

risk_threshold = st.sidebar.slider(
    "High-Risk Probability Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    df["Shipping Mode"].isin(selected_shipping_modes)
    & df["Market"].isin(selected_markets)
    & df["Customer Segment"].isin(selected_segments)
].copy()

# ============================================================
# OVERVIEW
# ============================================================

st.header("1. Delay Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Orders",
        f"{len(filtered_df):,}"
    )

with col2:
    actual_late_rate = (
        filtered_df["Late_delivery_risk"].mean()
        if len(filtered_df) > 0
        else 0
    )

    st.metric(
        "Historical Late Rate",
        f"{actual_late_rate:.1%}"
    )

with col3:
    avg_scheduled_days = (
        filtered_df["Days for shipment (scheduled)"].mean()
        if len(filtered_df) > 0
        else 0
    )

    st.metric(
        "Avg Scheduled Days",
        f"{avg_scheduled_days:.2f}"
    )

with col4:
    st.metric(
        "Model ROC-AUC",
        f"{metadata['roc_auc']:.3f}"
    )

st.subheader("Shipping Mode Distribution")

if len(filtered_df) > 0:
    mode_counts = (
        filtered_df["Shipping Mode"]
        .value_counts()
    )

    st.bar_chart(mode_counts)

# ============================================================
# HISTORICAL REGION & MODE ANALYSIS
# ============================================================

st.header("2. Region & Mode Risk Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Late Delivery Rate by Shipping Mode")

    mode_risk = (
        filtered_df
        .groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(mode_risk)

with col2:

    st.subheader("Late Delivery Rate by Market")

    market_risk = (
        filtered_df
        .groupby("Market")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(market_risk)

st.subheader("Late Delivery Rate by Order Region")

region_risk = (
    filtered_df
    .groupby("Order Region")["Late_delivery_risk"]
    .mean()
    .sort_values(ascending=False)
)

st.dataframe(
    region_risk.reset_index().rename(
        columns={
            "Late_delivery_risk": "Historical Late Rate"
        }
    ),
    use_container_width=True
)

# ============================================================
# ORDER-LEVEL PREDICTION
# ============================================================

st.header("3. Order-Level Risk Prediction")

st.info(
    "Enter the characteristics of an order before shipment "
    "to estimate its late-delivery risk."
)

st.caption(
    "The model uses the operational information entered below. "
    "For model features that are not directly entered, typical "
    "values from the dataset are used."
)

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------

with st.form("order_prediction_form"):

    st.subheader("Order Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        input_type = st.selectbox(
            "Order Type",
            sorted(df["Type"].dropna().unique())
        )

        input_shipping_mode = st.selectbox(
            "Shipping Mode",
            sorted(df["Shipping Mode"].dropna().unique())
        )

        input_market = st.selectbox(
            "Market",
            sorted(df["Market"].dropna().unique())
        )

        input_customer_segment = st.selectbox(
            "Customer Segment",
            sorted(df["Customer Segment"].dropna().unique())
        )

    with col2:

        input_region = st.selectbox(
            "Order Region",
            sorted(df["Order Region"].dropna().unique())
        )

        input_scheduled_days = st.number_input(
            "Days for Shipment (Scheduled)",
            min_value=0,
            max_value=10,
            value=3,
            step=1
        )

        input_quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )

        input_product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    with col3:

        input_discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        input_discount_rate = st.number_input(
            "Order Item Discount Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

        input_profit_ratio = st.number_input(
            "Order Item Profit Ratio",
            min_value=-1.0,
            max_value=1.0,
            value=0.20,
            step=0.01
        )

        input_benefit = st.number_input(
            "Benefit per Order",
            value=20.0,
            step=1.0
        )

    input_sales_customer = st.number_input(
        "Sales per Customer",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    predict_button = st.form_submit_button(
        "Predict Late Delivery Risk",
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE BASELINE INPUT
    # --------------------------------------------------------

    prediction_input = {}

    # Numerical model features
    numerical_model_features = (
        df[model_columns]
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    # Categorical model features
    categorical_model_features = (
        df[model_columns]
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    # Use typical values for numerical variables
    for feature in numerical_model_features:

        if feature in df.columns:

            prediction_input[feature] = (
                df[feature].median()
            )

    # Use most common values for categorical variables
    for feature in categorical_model_features:

        if feature in df.columns:

            prediction_input[feature] = (
                df[feature].mode()[0]
            )

    # --------------------------------------------------------
    # OVERWRITE WITH USER INPUTS
    # --------------------------------------------------------

    prediction_input["Type"] = input_type

    prediction_input["Shipping Mode"] = input_shipping_mode

    prediction_input["Market"] = input_market

    prediction_input["Order Region"] = input_region

    prediction_input["Customer Segment"] = input_customer_segment

    prediction_input["Days for shipment (scheduled)"] = (
        input_scheduled_days
    )

    prediction_input["Order Item Quantity"] = (
        input_quantity
    )

    prediction_input["Order Item Product Price"] = (
        input_product_price
    )

    prediction_input["Product Price"] = (
        input_product_price
    )

    prediction_input["Order Item Discount"] = (
        input_discount
    )

    prediction_input["Order Item Discount Rate"] = (
        input_discount_rate
    )

    prediction_input["Order Item Profit Ratio"] = (
        input_profit_ratio
    )

    prediction_input["Benefit per order"] = (
        input_benefit
    )

    prediction_input["Sales per customer"] = (
        input_sales_customer
    )

    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

    prediction_input["Shipping Pressure Index"] = (
        input_quantity /
        (input_scheduled_days + 1)
    )

    prediction_input["First_Class_Flag"] = int(
        input_shipping_mode == "First Class"
    )

    prediction_input["Second_Class_Flag"] = int(
        input_shipping_mode == "Second Class"
    )

    prediction_input["Same_Day_Flag"] = int(
        input_shipping_mode == "Same Day"
    )

    prediction_input["Standard_Class_Flag"] = int(
        input_shipping_mode == "Standard Class"
    )

    # Regional congestion
    region_counts = df["Order Region"].value_counts()

    prediction_input["Regional_Congestion_Indicator"] = (
        region_counts.get(
            input_region,
            region_counts.median()
        )
    )

    # Order complexity
    quantity_norm = (
        input_quantity /
        df["Order Item Quantity"].max()
    )

    discount_norm = (
        input_discount_rate /
        df["Order Item Discount Rate"].max()
    )

    pressure_norm = (
        prediction_input["Shipping Pressure Index"] /
        df["Shipping Pressure Index"].max()
    )

    prediction_input["Order_Complexity_Score"] = (
        quantity_norm
        + discount_norm
        + pressure_norm
    ) / 3

    # --------------------------------------------------------
    # CONVERT TO DATAFRAME
    # --------------------------------------------------------

    prediction_input = pd.DataFrame(
        [prediction_input]
    )

    # Ensure exact model column order
    prediction_input = prediction_input[
        model_columns
    ]

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    processed_input = preprocessor.transform(
        prediction_input
    )

    probability = model.predict_proba(
        processed_input
    )[0, 1]

    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if probability < 0.40:

        risk_category = "Low Risk"

    elif probability <= 0.70:

        risk_category = "Medium Risk"

    else:

        risk_category = "High Risk"

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Late Delivery Probability",
            f"{probability:.1%}"
        )

    with result_col2:

        st.metric(
            "Risk Category",
            risk_category
        )

    # --------------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------------

    if risk_category == "High Risk":

        st.error(
            "⚠️ High-risk order: proactive operational "
            "review is recommended."
        )

    elif risk_category == "Medium Risk":

        st.warning(
            "⚠️ Medium-risk order: monitor the order "
            "and review available capacity."
        )

    else:

        st.success(
            "✓ Low-risk order: continue standard "
            "operational monitoring."
        )

    # --------------------------------------------------------
    # KEY INPUT FACTORS
    # --------------------------------------------------------

    st.subheader("Key Order Characteristics")

    explanation_col1, explanation_col2 = st.columns(2)

    with explanation_col1:

        st.write(
            f"**Shipping Mode:** "
            f"{input_shipping_mode}"
        )

        st.write(
            f"**Scheduled Shipping Days:** "
            f"{input_scheduled_days}"
        )

        st.write(
            f"**Shipping Pressure Index:** "
            f"{prediction_input['Shipping Pressure Index'].iloc[0]:.3f}"
        )

        st.write(
            f"**Order Quantity:** "
            f"{input_quantity}"
        )

    with explanation_col2:

        st.write(
            f"**Order Complexity Score:** "
            f"{prediction_input['Order_Complexity_Score'].iloc[0]:.3f}"
        )

        st.write(
            f"**Discount Rate:** "
            f"{input_discount_rate:.1%}"
        )

        st.write(
            f"**Order Profit Ratio:** "
            f"{input_profit_ratio:.2f}"
        )

        st.write(
            f"**Order Region:** "
            f"{input_region}"
        )

    st.caption(
        "The prediction represents model output, not a causal "
        "determination that any individual feature will cause "
        "a delivery delay."
    )

# ============================================================
# OPERATIONS ACTION PANEL
# ============================================================

st.header("4. Operations Action Panel")

st.markdown(
    """
    **Suggested operational interpretation**

    - **High Risk:** Review the order and consider proactive intervention.
    - **Medium Risk:** Monitor the order and review available capacity.
    - **Low Risk:** Continue normal processing while maintaining standard monitoring.

    These recommendations are operational guidelines and should be
    combined with current logistics conditions and human judgment.
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Late Delivery Risk Prediction | Unified Mentor Project"
)
