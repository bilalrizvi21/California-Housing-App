import streamlit as st
import numpy as np
import joblib

# Load model and features
model = joblib.load("forest_pipeline.pkl")
features = joblib.load("features.pkl")

# App config
st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠", layout="wide")

# Theme toggle
theme = st.radio("Choose Theme:", ["🌞 Light", "🌙 Dark"], horizontal=True)

# Theme-based styling variables
background_gradient = (
    "linear-gradient(135deg, #fdfbfb, #ebedee)"  # Light gradient
    if theme == "🌞 Light"
    else "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"  # Dark gradient
)

text_color = "#000000" if theme == "🌞 Light" else "#ffffff"
box_bg = "rgba(255, 255, 255, 0.8)" if theme == "🌞 Light" else "rgba(0, 0, 0, 0.3)"
card_bg = "rgba(255, 255, 255, 0.3)" if theme == "🌞 Light" else "rgba(0, 0, 0, 0.35)"

#css styling

st.markdown(
    f"""
    <style>
        body {{
            color: {text_color};
        }}

        .stApp {{
            background: {background_gradient};
            background-attachment: fixed;
            background-size: cover;
            animation: fadeIn 1s ease-in-out;
        }}

        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}

        /* Input, select, and text fields */
        input, select, textarea {{
            color: {text_color} !important;
            background-color: {box_bg} !important;
        }}

        /* Ocean proximity dropdown fix */
        div[data-baseweb="select"] > div {{
            background-color: {box_bg} !important;
            color: {text_color} !important;
        }}
        div[data-baseweb="select"] * {{
            color: {text_color} !important;
        }}

        /* Focus highlight on inputs */
        input:focus, select:focus, textarea:focus {{
            outline: none;
            border: 2px solid #007acc;
            box-shadow: 0 0 5px #007acc;
            transition: 0.3s ease;
        }}

        /* Input containers */
        .stTextInput, .stNumberInput, .stSelectbox {{
            background-color: {box_bg} !important;
            border-radius: 10px;
            padding: 10px;
        }}

        /* Labels */
        label, .stTextInput > label, .stNumberInput > label, .stSelectbox > label {{
            font-weight: bold !important;
            font-size: 1rem !important;
            color: {text_color} !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: #007acc;
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: #005b99;
            transform: scale(1.05);
        }}

        /* Animated title */
        h1 {{
            color: #007acc;
            text-align: center;
            animation: slideIn 0.8s ease;
        }}

        @keyframes slideIn {{
            0% {{ transform: translateY(-30px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}

        /* General text fix */
        div, p, span {{
            color: {text_color} !important;
        }}

        ::placeholder {{
            color: #aaa !important;
        }}

        .glass-card {{
            max-width: 900px;
            margin: 30px auto;
            padding: 2rem;
            border-radius: 20px;
            background: {card_bg};
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
    </style>
    """,
    unsafe_allow_html=True
)





st.title("🏠 California Housing Price Predictor")

# Optional: add logo
# st.image("your_logo.png", width=100)

# Glass card container with 2-column layout
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### Enter Housing Details:")

    col1, col2 = st.columns(2)

    with col1:
        median_income = st.number_input("Median Income", 0.0, 20.0, 3.0)
        housing_median_age = st.number_input("Housing Median Age", 1.0, 100.0, 30.0)
        total_rooms = st.number_input("Total Rooms", 1.0, 50000.0, 3000.0)
        total_bedrooms = st.number_input("Total Bedrooms", 1.0, 10000.0, 500.0)
        population = st.number_input("Population", 1.0, 30000.0, 1500.0)

    with col2:
        households = st.number_input("Households", 1.0, 5000.0, 500.0)
        latitude = st.number_input("Latitude", 32.0, 42.0, 34.0)
        longitude = st.number_input("Longitude", -125.0, -114.0, -118.0)
        ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

    st.markdown('</div>', unsafe_allow_html=True)

# One-hot encode ocean proximity
ocean_encoded = {
    '<1H OCEAN': [1, 0, 0, 0, 0],
    'INLAND': [0, 1, 0, 0, 0],
    'ISLAND': [0, 0, 1, 0, 0],
    'NEAR BAY': [0, 0, 0, 1, 0],
    'NEAR OCEAN': [0, 0, 0, 0, 1],
}[ocean_proximity]

# Feature engineering
total_rooms_log = np.log(total_rooms + 1)
total_bedrooms_log = np.log(total_bedrooms + 1)
population_log = np.log(population + 1)
households_log = np.log(households + 1)
bedroom_ratio = total_bedrooms_log / total_rooms_log
per_household_rooms = total_rooms_log / households_log

# Final input order
input_data = [
    median_income,
    housing_median_age,
    total_rooms_log,
    total_bedrooms_log,
    population_log,
    households_log,
    latitude,
    longitude,
    *ocean_encoded,
    bedroom_ratio,
    per_household_rooms
]

input_array = np.array([input_data])

# Prediction button
if st.button("💡 Predict"):
    prediction = model.predict(input_array)[0]
    st.success(f"🏡 Estimated Median House Value: **${prediction:,.2f}**")
