# Machine Learning–Based Late Delivery Risk Prediction in Global Supply Chain Operations

## Project Overview

This project develops a machine learning system to predict the risk of late delivery in global supply chain operations **before an order is shipped**.

The project uses historical logistics and order data to identify patterns associated with delivery delays and provides:

- Late delivery probability
- Low / Medium / High risk classification
- Key operational risk drivers
- High-risk order identification
- Shipping mode and regional risk analysis
- An interactive Streamlit dashboard for operational analysis

The project was developed as part of the **Unified Mentor / APL Logistics (KWE Group)** project work.

---

## Business Problem

Late deliveries can affect customer satisfaction, operational efficiency, service-level performance, and supply chain planning.

The objective of this project is to build a predictive system that can help operations teams identify potentially high-risk orders **before shipment**, allowing earlier intervention and better prioritization.

### Main Question

> Can historical order, shipping, product, and regional information be used to predict whether an order is at risk of late delivery?

---

## Project Objectives

1. Analyze historical logistics and delivery data.
2. Identify patterns associated with late deliveries.
3. Perform data preprocessing and feature engineering.
4. Develop machine learning classification models.
5. Compare Logistic Regression, Random Forest, and XGBoost.
6. Generate late delivery probabilities.
7. Classify orders into Low, Medium, and High risk.
8. Identify important operational risk drivers.
9. Build an interactive Streamlit dashboard.
10. Provide operational insights that can support proactive decision-making.

---

## Dataset

The project uses the `APL_Logistics.csv` dataset.

### Dataset Size

- **Rows:** 180,519
- **Columns:** 40

The dataset contains information related to:

- Orders
- Customers
- Products
- Shipping modes
- Markets
- Regions
- Sales
- Discounts
- Profitability
- Scheduled shipping time
- Delivery status
- Late delivery risk

### Important Variables

Examples include:

- Shipping Mode
- Market
- Customer Segment
- Order Region
- Order Country
- Category Name
- Product Name
- Days for shipment (scheduled)
- Order Item Quantity
- Product Price
- Sales
- Discount Rate
- Profit Ratio
- Benefit per order
- Late Delivery Risk

---

## Methodology

The project follows a structured machine learning workflow.

### 1. Data Preprocessing

The preprocessing stage included:

- Missing value handling
- Duplicate checking
- Categorical value cleaning
- Categorical encoding
- Numerical feature scaling
- Train/test splitting
- Class imbalance assessment

Two missing-value fields were handled using the mode:

- Customer Lname
- Customer Zipcode

No duplicate rows were found.

The target variable was:

```text
Late_delivery_risk
```

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| 0 | 81,542 | 45.17% |
| 1 | 98,977 | 54.83% |

Because the imbalance was relatively mild, class weighting was used instead of SMOTE.

---

## 2. Feature Engineering

Several operational features were created to improve the model.

### Shipping Pressure Index

Measures order quantity relative to scheduled shipping time:

```text
Shipping Pressure Index =
Order Item Quantity / (Scheduled Shipping Days + 1)
```

### Shipping Mode Risk Flags

Binary indicators were created for:

- First Class
- Second Class
- Same Day
- Standard Class

### Regional Congestion Indicator

The number of orders associated with each order region was used as a regional congestion indicator.

### Order Complexity Score

A combined normalized measure based on:

- Order quantity
- Discount rate
- Shipping pressure

These features were designed to capture operational characteristics that may be associated with delivery risk.

---

## 3. Model Development

Three machine learning approaches were evaluated:

### Logistic Regression

Used as the baseline classification model.

### Random Forest

Used as the main tree-based ensemble model.

### XGBoost

Used as an advanced gradient boosting model.

---

## Model Performance

The initial model comparison produced the following results:

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7430 | 0.8463 | 0.5439 | 0.6622 |
| Random Forest | 0.8171 | 0.8483 | 0.6158 | 0.7136 |
| XGBoost | 0.7843 | 0.8362 | 0.5755 | 0.6817 |

During model review, identifier and location-proxy variables were removed to produce a more operationally defensible model.

The cleaned Random Forest model achieved:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.7598 |
| Precision | 0.7947 |
| Recall | 0.6075 |
| F1 Score | 0.6886 |

The cleaned model was retained as the operational model because it avoids relying heavily on customer IDs, geographic coordinates, postal codes, and other identifier/location-proxy variables.

---

## Risk Classification

The model produces a probability between 0 and 1.

The current operational risk bands are:

| Probability | Risk Category |
|---|---|
| `< 0.40` | Low Risk |
| `0.40 – 0.70` | Medium Risk |
| `> 0.70` | High Risk |

These thresholds are initial operational thresholds and were not statistically optimized.

---

## Explainability

Model interpretation was included to help explain why an order receives a high-risk prediction.

Important operational features in the cleaned Random Forest included:

- Benefit per order
- Order Profit Per Order
- Order Item Profit Ratio
- Days for shipment (scheduled)
- Order Item Total
- Sales per customer
- Order Item Discount
- Standard Class indicator
- Order Complexity Score
- Shipping Mode
- Discount Rate
- Regional Congestion Indicator
- Shipping Pressure Index

The dashboard can provide order-level risk information and characteristics associated with the prediction.

> Feature importance and sensitivity analysis describe predictive relationships in the model. They should not be interpreted as proof that a feature causally causes delivery delays.

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for operational analysis.

### Dashboard Sections

#### Delay Risk Overview

Provides a high-level view of:

- Historical late delivery rate
- Average scheduled shipping days
- Model ROC-AUC
- Shipping mode distribution

#### Region & Mode Risk Analysis

Allows analysis of delivery risk across:

- Shipping modes
- Markets
- Order regions
- Customer segments

#### Order-Level Risk Prediction

Users can enter pre-shipment order information such as:

- Order type
- Shipping mode
- Market
- Customer segment
- Order region
- Scheduled shipping days
- Quantity
- Product price
- Discount
- Discount rate
- Profit ratio
- Benefit
- Sales per customer

The system then generates:

- Late delivery probability
- Risk category
- Key order characteristics

#### Operations Action Panel

The dashboard provides an operational view of potentially high-risk orders and risk categories.

---

## Project Structure

```text
late-delivery-risk-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── APL_Logistics.csv
│   ├── dashboard_predictions.csv
│   ├── high_risk_order_list.csv
│   ├── key_risk_drivers.csv
│   └── risk_summary.csv
│
├── models/
│   ├── final_preprocessor.pkl
│   ├── final_random_forest.pkl
│   ├── model_input_columns.pkl
│   └── model_metadata.pkl
│
├── notebooks/
│   └── late_delivery_risk_prediction.ipynb
│
└── README.md
```

---

## Technologies Used

### Programming

- Python 3.11

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn

### Model Management

- Joblib

### Dashboard

- Streamlit

### Development Environment

- Anaconda
- Jupyter Notebook
- GitHub
- GitHub Desktop

---

## Installation

Create and activate the project environment:

```bash
conda activate unified-mentor
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib streamlit openpyxl jupyter
```

---

## Running the Streamlit Dashboard

Navigate to the project directory:

```bash
cd "D:\LEARNING\unified mentor\project 1\late-delivery-risk-prediction"
```

Run:

```bash
streamlit run app\app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Machine Learning Pipeline

The overall pipeline is:

```text
Raw Logistics Data
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Handling
        │
        ▼
Feature Engineering
        │
        ▼
Train / Test Split
        │
        ▼
Categorical Encoding
        │
        ▼
Numerical Scaling
        │
        ▼
Random Forest Model
        │
        ▼
Late Delivery Probability
        │
        ▼
Risk Classification
        │
        ├── Low Risk
        ├── Medium Risk
        └── High Risk
        │
        ▼
Streamlit Dashboard
```

---

## Model Artifacts

The trained model and preprocessing components are stored in the `models/` directory.

### Files

`final_random_forest.pkl`

Trained Random Forest classification model.

`final_preprocessor.pkl`

Preprocessing pipeline containing transformations used before prediction.

`model_input_columns.pkl`

List of model input columns required for prediction.

`model_metadata.pkl`

Stores model information, evaluation metrics, thresholds, and feature information.

---

## Key Outputs

The project generates operational outputs including:

### Risk Summary

```text
data/risk_summary.csv
```

Contains summary-level risk information.

### Key Risk Drivers

```text
data/key_risk_drivers.csv
```

Contains important model-derived risk drivers.

### High-Risk Orders

```text
data/high_risk_order_list.csv
```

Contains orders classified as high risk for further operational review.

### Dashboard Predictions

```text
data/dashboard_predictions.csv
```

Contains model predictions used by the dashboard for analysis.

---

## Important Modeling Considerations

### Pre-shipment Prediction

Post-shipment variables such as:

```text
Days for shipping (real)
Delivery Status
Order Status
```

were excluded from the final predictive model because they would not be available at the required pre-shipment decision point.

### Identifier Reduction

The final operational model excludes variables such as:

- Customer ID
- Order Customer ID
- Category ID
- Department ID
- Customer Zipcode
- Latitude
- Longitude

This reduces reliance on identifiers and location proxies.

### Probability Interpretation

A high predicted probability indicates that the model estimates a higher likelihood of late delivery based on learned historical patterns.

It does not establish causation.

---

## Limitations

This project has several limitations:

1. The dataset represents historical operations and may not perfectly represent future conditions.
2. The risk thresholds are initial operational thresholds rather than statistically optimized decision thresholds.
3. Regional congestion is derived from historical order counts.
4. Feature importance should not be interpreted as causal evidence.
5. Model performance may change when applied to new operational environments.
6. Additional real-time logistics information could potentially improve future versions of the system.
7. The current project is a predictive decision-support system and does not automatically execute operational interventions.

---

## Future Improvements

Possible future development includes:

- Hyperparameter optimization
- Cross-validation
- Probability calibration
- Threshold optimization based on operational costs
- SHAP-based explainability
- Real-time logistics data integration
- Time-based validation
- Model monitoring
- Data drift detection
- Automated retraining
- Integration with supply chain management systems
- More detailed intervention recommendations

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for late delivery risk prediction in supply chain operations.

The solution combines:

- Data preprocessing
- Feature engineering
- Machine learning classification
- Model evaluation
- Risk probability estimation
- Explainability
- KPI generation
- Interactive Streamlit visualization

The resulting system is designed to support proactive identification of potentially high-risk orders before shipment and provide an analytical foundation for operational decision-making.

---

## Author

**Unified Mentor Project**

**Project:** Machine Learning–Based Late Delivery Risk Prediction in Global Supply Chain Operations
