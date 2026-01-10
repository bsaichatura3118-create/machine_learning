"""
Streamlit app for Online Shoppers Purchasing Intention Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Shopping Intention Predictor",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Online Shoppers Purchasing Intention Predictor")
st.write("Predict whether a customer will make a purchase based on their browsing behavior")

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load scaler
    scaler = None
    if os.path.exists('model/scaler.pkl'):
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    return models, scaler

# Check if models exist
if not os.path.exists('model/logistic_regression.pkl'):
    st.error("❌ Models not found! Please run `python train.py` first to train and save the models.")
    st.stop()

models, scaler = load_models()

if not models or scaler is None:
    st.error("❌ Failed to load models or scaler. Please check the model files.")
    st.stop()

# Load saved encoders (fallback to fitting on dataset if not present)
le_month = None
le_visitor = None
le_weekend = None
enc_month_path = 'model/encoder_month.pkl'
enc_visitor_path = 'model/encoder_visitor.pkl'
enc_weekend_path = 'model/encoder_weekend.pkl'

if os.path.exists(enc_month_path) and os.path.exists(enc_visitor_path) and os.path.exists(enc_weekend_path):
    with open(enc_month_path, 'rb') as f:
        le_month = pickle.load(f)
    with open(enc_visitor_path, 'rb') as f:
        le_visitor = pickle.load(f)
    with open(enc_weekend_path, 'rb') as f:
        le_weekend = pickle.load(f)
else:
    # fallback: fit from dataset (not preferred but ensures app still works)
    try:
        data_for_encoding = pd.read_csv('dataset.csv')
        le_month = LabelEncoder(); le_month.fit(data_for_encoding['Month'].astype(str))
        le_visitor = LabelEncoder(); le_visitor.fit(data_for_encoding['VisitorType'].astype(str))
        le_weekend = LabelEncoder(); le_weekend.fit(data_for_encoding['Weekend'].astype(str))
    except Exception:
        # If even fallback fails, create simple encoders mapping
        le_month = LabelEncoder(); le_month.fit(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        le_visitor = LabelEncoder(); le_visitor.fit(["Returning_Visitor","New_Visitor","Other"]) 
        le_weekend = LabelEncoder(); le_weekend.fit(["False","True"]) 

# Sidebar for model selection
st.sidebar.header("⚙️ Configuration")
selected_model = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys()),
    help="Choose the model to use for prediction"
)

st.sidebar.info(
    "**Model Information:**\n\n"
    "- **Logistic Regression**: Linear classifier\n"
    "- **Decision Tree**: Tree-based classifier\n"
    "- **KNN**: Distance-based classifier\n"
    "- **Naive Bayes**: Probabilistic classifier\n"
    "- **Random Forest**: Ensemble method\n"
    "- **XGBoost**: Gradient boosting ensemble"
)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Features")
    
    # Create input fields organized by category
    st.write("**Administrative Page Features**")
    administrative = st.slider("Administrative Pages Visited", 0, 27, 1)
    administrative_duration = st.slider("Time on Admin Pages (seconds)", 0, 3400, 100)
    
    st.write("**Informational Page Features**")
    informational = st.slider("Informational Pages Visited", 0, 24, 0)
    informational_duration = st.slider("Time on Info Pages (seconds)", 0, 2550, 0)
    
    st.write("**Product Page Features**")
    product_related = st.slider("Product Related Pages Visited", 0, 705, 1)
    product_related_duration = st.slider("Time on Product Pages (seconds)", 0, 63973, 500)

with col2:
    st.subheader("📈 Additional Features")
    
    bounce_rate = st.slider("Bounce Rate (%)", 0.0, 100.0, 50.0)
    exit_rate = st.slider("Exit Rate (%)", 0.0, 100.0, 50.0)
    page_values = st.slider("Page Values", 0.0, 365.0, 0.0)
    special_day = st.slider("Special Day (closeness to holiday)", 0, 6, 0)
    
    st.write("**Categorical Features**")
    month = st.selectbox(
        "Month",
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        ["Returning_Visitor", "New_Visitor", "Other"],
    )

    weekend = st.selectbox(
        "Is Weekend?",
        ["False", "True"],
    )

    st.write("**System / Session Info**")
    operating_systems = st.number_input("Operating Systems (numeric code)", min_value=0, max_value=50, value=3)
    browser = st.number_input("Browser (numeric code)", min_value=0, max_value=50, value=10)
    region = st.number_input("Region (numeric code)", min_value=0, max_value=50, value=1)
    traffic_type = st.number_input("Traffic Type (numeric code)", min_value=0, max_value=50, value=1)

# Prediction section
st.divider()
st.subheader("🔮 Prediction")

if st.button("🚀 Make Prediction", type="primary"):
    # Encode categorical inputs using encoders fitted on training data
    try:
        month_enc = le_month.transform([str(month)])[0]
    except Exception:
        month_enc = 0
    try:
        visitor_enc = le_visitor.transform([str(visitor_type)])[0]
    except Exception:
        visitor_enc = 0
    try:
        weekend_enc = le_weekend.transform([str(weekend)])[0]
    except Exception:
        weekend_enc = 0

    # Prepare input data in the same feature order as training:
    # Administrative, Administrative_Duration, Informational, Informational_Duration,
    # ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
    # SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend
    input_array = [
        administrative,
        administrative_duration,
        informational,
        informational_duration,
        product_related,
        product_related_duration,
        bounce_rate,
        exit_rate,
        page_values,
        special_day,
        month_enc,
        operating_systems,
        browser,
        region,
        traffic_type,
        visitor_enc,
        weekend_enc
    ]

    input_data = np.array([input_array])

    # Build DataFrame with original feature names so scaler sees feature names
    feature_cols = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
        'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    input_df = pd.DataFrame([input_array], columns=feature_cols)

    # Scale input data (use DataFrame to avoid sklearn feature-name warning)
    input_scaled = scaler.transform(input_df)
    
    # Get prediction
    model = models[selected_model]
    prediction = model.predict(input_scaled)[0]
    
    # Get probability if available
    probability = None
    prob_unavailable = False
    try:
        prob_array = model.predict_proba(input_scaled)[0]
        probability = prob_array[1]  # Probability of purchase
    except Exception:
        prob_unavailable = True
    
    # Display results
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.success("✅ **LIKELY TO PURCHASE**", icon="🛒")
            st.write("This visitor is predicted to generate revenue")
        else:
            st.warning("⚠️ **UNLIKELY TO PURCHASE**", icon="👤")
            st.write("This visitor is predicted to NOT generate revenue")
    
    with result_col2:
        if probability is not None:
            st.metric("Purchase Probability", f"{probability*100:.2f}%")
            # Confidence indicator
            confidence = max(probability, 1-probability) * 100
            st.metric("Model Confidence", f"{confidence:.2f}%")
        elif prob_unavailable:
            st.info("Purchase Probability: Not available for this model")

# Information section
st.divider()
st.subheader("ℹ️ About This Project")

with st.expander("View Dataset Information"):
    st.write("""
    **Dataset:** Online Shoppers Purchasing Intention Dataset
    
    **Features:** 13 input features + 1 target variable
    - Administrative, Informational, ProductRelated pages visited
    - Time spent on each category
    - Bounce Rate and Exit Rate metrics
    - Page Values and Special Day indicator
    - Month, Visitor Type, Weekend status
    
    **Target Variable:** Revenue (Purchase made: Yes/No)
    
    **Class Distribution:** Imbalanced (≈85% No Purchase, ≈15% Purchase)
    """)

with st.expander("View Model Comparison"):
    comparison_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.5000, 0.7457, 0.8268, 0.8487, 0.8487, 0.5191],
        'AUC': [0.4746, 0.4999, 0.4891, 0.4733, 0.4893, 0.5045],
        'Precision': [0.8424, 0.8487, 0.8480, 0.8487, 0.8487, 0.8512],
        'Recall': [0.5055, 0.8524, 0.9699, 1.0000, 1.0000, 0.5251],
        'F1 Score': [0.6318, 0.8505, 0.9048, 0.9182, 0.9182, 0.6495],
        'MCC': [-0.0182, -0.0002, -0.0127, 0.0000, 0.0000, 0.0074]
    }
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

with st.expander("View Project Structure"):
    st.write("""
    ```
    ML_ASSIGN2/
    ├── app.py                      # Streamlit web application
    ├── train.py                    # Model training script
    ├── main.py                     # Original evaluation script
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project documentation
    ├── dataset.csv                 # Training dataset
    └── model/
        ├── scaler.pkl
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        ├── random_forest.pkl
        └── xgboost.pkl
    ```
    """)

st.divider()
st.caption("🔬 Machine Learning Model Comparison | Online Shoppers Purchasing Intention Prediction")
