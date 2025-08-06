import streamlit as st
import pandas as pd
import joblib
import time

# Load model
model_path = r'D:/Portfolio/waste_management/models/random_forest_tuned_model.pkl'
model = joblib.load(model_path)

# Configure page
st.set_page_config(page_title="♻️ Recycling Rate Predictor ♻️", layout="centered")

# Custom CSS styling for a gentle UI
st.markdown("""
    <style>
    /* Gradient Title */
    .title {
        font-size: 44px;
        text-align: center;
        background: linear-gradient(to right, #4caf50, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555555;
        margin-top: 0;
        margin-bottom: 25px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Input box styling */
    div.stNumberInput > label, div.stSelectbox > label {
        font-weight: 600;
        margin-bottom: 4px;
    }
    .stNumberInput > div, .stSelectbox > div {
        background-color: #f7f9f8 !important;
        border-radius: 8px;
        padding: 8px 10px;
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #4caf50;  /* green */
        color: white;
        border: none;
        padding: 0.6em 1.7em;
        font-size: 18px;
        border-radius: 10px;
        font-weight: 700;
        transition: 0.25s ease-in-out;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #81c784;  /* lighter green */
        color: black;
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(129, 199, 132, 0.4);
    }
    /* Prediction Result Box */
    .result-box {
        border: 2px solid #4caf50;
        border-radius: 15px;
        background-color: #e8f5e9;
        padding: 24px 32px;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        color: #2e7d32;
        margin-top: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 8px rgba(46, 125, 50, 0.15);
    }
    /* Green progress bar fill */
    .stProgress > div > div > div > div {
        background-color: #4caf50 !important;
    }
    </style>
""", unsafe_allow_html=True)



# Titles
st.markdown('<div class="title">♻️ Waste Management Recycling Rate Predictor ♻️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Provide urban waste management details to predict Recycling Rate (%) for any year</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    st.subheader("📊 Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("🏙️ City/District", [
            "Mumbai", "Delhi", "Bengaluru", "Kolkata", "Chennai"
        ])
        waste_type = st.selectbox("🗑️ Waste Type", [
            "Plastic", "Organic", "E-Waste", "Construction", "Hazardous"
        ])
        waste_generated = st.number_input(
            "♨️ Waste Generated (Tons/Day)", 
            min_value=0.0, 
            step=0.1, 
            format="%.2f", 
            value=0.0
        )
        population_density = st.number_input(
            "👥 Population Density (People/km²)", 
            min_value=0, 
            step=1, 
            format="%d", 
            value=0
        )
        municipal_efficiency = st.slider("🏛️ Municipal Efficiency Score (1-10)", 1, 10, 5)

    with col2:
        cost_per_ton = st.number_input(
            "💰 Cost of Waste Management (₹/Ton)", 
            min_value=0.0, 
            step=10.0, 
            format="%.2f", 
            value=0.0
        )
        awareness_campaigns = st.number_input(
            "📢 Awareness Campaigns Count", 
            min_value=0, 
            step=1, 
            format="%d", 
            value=0
        )
        landfill_capacity = st.number_input(
            "🚮 Landfill Capacity (Tons)", 
            min_value=0.0, 
            step=100.0, 
            format="%.2f", 
            value=0.0
        )
        year = st.number_input(
            "📅 Year",
            min_value=2010,
            max_value=2100,
            value=2023,
            step=1,
            help="Select the year for prediction; can be future years beyond dataset"
        )

    submitted = st.form_submit_button("🔍 Predict Recycling Rate")

def prepare_input():
    data = {
        'City/District_' + city: 1,
        'Waste Type_' + waste_type: 1,
        'Waste Generated (Tons/Day)': waste_generated,
        'Population Density (People/km²)': population_density,
        'Municipal Efficiency Score (1-10)': municipal_efficiency,
        'Cost of Waste Management (₹/Ton)': cost_per_ton,
        'Awareness Campaigns Count': awareness_campaigns,
        'Landfill Capacity (Tons)': landfill_capacity,
        'Year': year
    }

    # Initialize all features expected by the model to zero
    features = model.feature_names_in_.tolist()
    input_dict = dict.fromkeys(features, 0)

    # Insert actual input values
    for key, value in data.items():
        if key in input_dict:
            input_dict[key] = value

    return pd.DataFrame([input_dict])

if submitted:
    input_df = prepare_input()
    progress_bar = st.progress(0)  # Initialize progress bar at 0%
    try:
        for pct in range(0, 101, 5):   # Increase progress in steps of 5%
            time.sleep(0.1)             # Total delay ~2 seconds (0.1 * 20 steps)
            progress_bar.progress(pct)  # Update progress bar
        prediction = model.predict(input_df)[0]
        progress_bar.empty()            # Remove progress bar after completion
        st.markdown(f'<div class="result-box">🔮 Predicted Recycling Rate: <br><span>{prediction:.2f}%</span></div>', unsafe_allow_html=True)
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error in prediction: {e}")

