
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Prediction | ANN",
    page_icon="💰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.hero {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(120deg, #172554, #164e63);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #cbd5e1;
}
.metric-card {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}
.result-card {
    background: #123b36;
    border: 1px solid #16876d;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
}
.result-card h2 {
    color: #6ee7b7;
}
div.stButton > button {
    width: 100%;
    background: #0891b2;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 12px;
    border: none;
}
div.stButton > button:hover {
    background: #0e7490;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND FILES ----------------
@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        "regression_model.h5"
    )

    with open("label_encoder_gender.pkl", "rb") as file:
        gender_encoder = pickle.load(file)

    # This file contains the OneHotEncoder for Geography
    with open("onehot_encoder_geo.pkl", "rb") as file:
        geo_encoder = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, gender_encoder, geo_encoder, scaler


try:
    model, gender_encoder, geo_encoder, scaler = load_artifacts()

    # Exact features used during training
    expected_columns = list(scaler.feature_names_in_)

except Exception as e:
    st.error("Unable to load model or preprocessing files.")
    st.exception(e)
    st.stop()


# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:

    st.title("💰 Salary Predictor")
    st.caption("ANN Regression Application")
    st.divider()

    st.subheader("Customer Information")

    geography = st.selectbox(
        "Geography",
        geo_encoder.categories_[0].tolist()
    )

    gender = st.selectbox(
        "Gender",
        gender_encoder.classes_.tolist()
    )

    st.subheader("Financial Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=1000,
        value=650
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0
    )

    st.subheader("Account Information")

    age = st.slider(
        "Age", 18, 100, 35
    )

    tenure = st.slider(
        "Tenure", 0, 10, 5
    )

    num_of_products = st.slider(
        "Number of Products", 1, 4, 1
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_active_member = st.selectbox(
        "Is Active Member?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # IMPORTANT: Exited is required by your saved scaler
    exited = st.selectbox(
        "Customer Exited?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    predict_button = st.button(
        "💡 Predict Estimated Salary"
    )


# ---------------- MAIN DASHBOARD ----------------
st.markdown("""
<div class="hero">
    <h1>Customer Salary Prediction</h1>
    <p>
        Artificial Neural Network powered regression model
        to estimate customer salary.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🧠 ANN</h3>
        <p>Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>12 Features</h3>
        <p>Customer Input Variables</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Regression</h3>
        <p>Numerical Prediction</p>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.subheader("📊 Prediction Results")


# ---------------- PREDICTION ----------------
if predict_button:

    try:
        # 1. Encode Gender
        gender_encoded = gender_encoder.transform(
            [gender]
        )[0]

        # 2. One-hot encode Geography
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

        # 3. Create customer input DataFrame
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "Exited": [exited]
        })

        # 4. Add Geography columns
        input_data = pd.concat(
            [input_data, geo_df],
            axis=1
        )

        # 5. Check for missing or extra features
        missing_columns = [
            col for col in expected_columns
            if col not in input_data.columns
        ]

        extra_columns = [
            col for col in input_data.columns
            if col not in expected_columns
        ]

        if missing_columns or extra_columns:

            st.error("Input features do not match training features.")

            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")

            if extra_columns:
                st.error(f"Extra columns: {extra_columns}")

                st.stop()

        # 6. Arrange features in exact training order
        input_data = input_data[expected_columns]

        # 7. Scale input data
        input_scaled = scaler.transform(input_data)

        # 8. Predict salary
        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        predicted_salary = float(prediction[0][0])

        # 9. Display prediction
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Estimated Customer Salary</h3>
                <h2>₹ {predicted_salary:,.2f}</h2>
                <p>Predicted by your trained ANN regression model</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Salary prediction completed successfully!")

        with st.expander("View Customer Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:
        st.error("Prediction failed. Please check the input data.")
        st.exception(e)

else:
    st.info(
        "Enter customer information in the sidebar "
        "and click Predict Estimated Salary."
    )


# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
<p style="color:gray;">
Built using Python | TensorFlow | Keras | Scikit-learn | Streamlit
<br>
ANN Regression Project
</p>
</center>
""", unsafe_allow_html=True)
