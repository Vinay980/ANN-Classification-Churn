
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import plotly.graph_objects as go

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    .hero {
        padding: 26px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #164e63);
        border: 1px solid #256477;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 34px;
        margin: 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
        margin-top: 10px;
    }

    .metric-card {
        background: #151f30;
        padding: 22px;
        border: 1px solid #2a374a;
        border-radius: 15px;
        min-height: 135px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 700;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin: 12px 0;
        border: 1px solid #334155;
        background: #151f30;
    }

    .result-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .muted {
        color: #9ca3af;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 12px;
        font-weight: 700;
        background: #0891b2;
        color: white;
    }

    div.stButton > button:hover {
        background: #0e7490;
        color: white;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 25px 0 5px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL & ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_geo.pkl", "rb") as f:
        geo_encoder = pickle.load(f)

    with open("label_encoder_gender.pkl", "rb") as f:
        gender_encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, geo_encoder, gender_encoder, scaler


try:
    model, geo_encoder, gender_encoder, scaler = load_artifacts()
except Exception as e:
    st.error("Unable to load the model or preprocessing files.")
    st.exception(e)
    st.stop()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption("AI-powered customer analytics")
    st.divider()

    st.markdown("### 👤 Customer Profile")

    geography = st.selectbox(
        "🌍 Geography",
        geo_encoder.categories_[0]
    )

    gender = st.selectbox(
        "⚧ Gender",
        gender_encoder.classes_
    )

    st.markdown("### 💳 Financial Details")

    credit_score = st.number_input(
        "Credit Score", min_value=0, max_value=1000,
        value=650, step=1
    )

    balance = st.number_input(
        "Account Balance", min_value=0.0,
        value=50000.0, step=1000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary", min_value=0.0,
        value=50000.0, step=1000.0
    )

    st.markdown("### 📈 Account Activity")

    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

    has_cr_card = st.selectbox(
        "Has Credit Card?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_active_member = st.selectbox(
        "Active Member?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    st.divider()
    predict_clicked = st.button(
        "🔍 Analyze Customer",
        use_container_width=True
    )


# -------------------- MAIN HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>Customer Churn Intelligence</h1>
    <p>
        Understand customer retention risk using your trained
        Artificial Neural Network (ANN).
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📌 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL</div>
        <div class="metric-value">ANN</div>
        <div class="muted">Neural network classification</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ANALYSIS</div>
        <div class="metric-value">12 Features</div>
        <div class="muted">Customer information inputs</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">PREDICTION TYPE</div>
        <div class="metric-value">Churn Risk</div>
        <div class="muted">Estimated probability of leaving</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------- PREDICTION --------------------
st.markdown("### 🧠 Customer Risk Analysis")

if predict_clicked:
    try:
        gender_encoded = gender_encoder.transform([gender])[0]

        geo_encoded = geo_encoder.transform(
            [[geography]]
        ).toarray()

        geo_columns = geo_encoder.get_feature_names_out(
            ["Geography"]
        )

        geo_df = pd.DataFrame(
            geo_encoded,
            columns=geo_columns
        )

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        })

        input_data = pd.concat(
            [input_data, geo_df], axis=1
        )

        expected_columns = [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain",
        ]

        input_data = input_data[expected_columns]
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(
            input_scaled, verbose=0
        )
        probability = float(prediction[0][0])

        if not 0 <= probability <= 1:
            st.error("The model returned an invalid probability.")
            st.stop()

        churn_percent = probability * 100

        if probability >= 0.5:
            result_title = "High Churn Risk"
            result_icon = "⚠️"
            result_color = "#f87171"
            explanation = (
                "The model predicts that this customer is likely to leave."
            )
        else:
            result_title = "Lower Churn Risk"
            result_icon = "✅"
            result_color = "#34d399"
            explanation = (
                "The model predicts that this customer is less likely to leave."
            )

        left, right = st.columns([1.1, 1])

        with left:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        {result_icon} {result_title}
                    </div>
                    <p>{explanation}</p>
                    <p class="muted">
                        This result is a model estimate, not a guarantee.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Estimated Churn Probability",
                f"{churn_percent:.1f}%"
            )

            st.progress(probability)

            st.caption(
                "The classification threshold is 50%. "
                "This threshold may need adjustment based on model evaluation."
            )

        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_percent,
                    number={"suffix": "%", "font": {"size": 38}},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "ticksuffix": "%"
                        },
                        "bar": {"color": result_color},
                        "bgcolor": "#151f30",
                        "bordercolor": "#334155",
                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#123b36"
                            },
                            {
                                "range": [50, 100],
                                "color": "#48252e"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "#f8fafc",
                                "width": 3
                            },
                            "thickness": 0.8,
                            "value": 50
                        }
                    }
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="#0b1120",
                font={"color": "#e5e7eb"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with st.expander("🔎 View encoded model input"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Check the input preprocessing and model files.")
        st.exception(e)

else:
    st.info(
        "👈 Enter the customer's details in the sidebar, "
        "then click **Analyze Customer** to view the prediction."
    )

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
    Built with Python, TensorFlow, Scikit-learn & Streamlit
    <br>
    Educational machine learning project · Predictions are estimates
</div>
""", unsafe_allow_html=True)
