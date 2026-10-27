# 🏦 ANN Customer Churn Classification

An end-to-end **Artificial Neural Network (ANN)** project that predicts whether a bank customer is likely to leave the bank. The project uses machine learning preprocessing techniques and a trained deep learning model to generate churn predictions through an interactive Streamlit web application.

## 🚀 Project Overview

Customer churn is a major challenge for banks and financial institutions. Identifying customers who are likely to leave helps businesses understand customer behavior and plan retention strategies.

This project uses customer information such as credit score, age, account balance, geography, and account activity to predict the probability of a customer leaving the bank.

The trained ANN model is integrated into a Streamlit application where users can enter customer details and receive a churn prediction along with the predicted probability.

## ✨ Features

- **Customer Churn Prediction:** Predict whether a customer is likely to leave the bank.
- **Artificial Neural Network:** Use a trained deep learning model for binary classification.
- **Data Preprocessing:** Encode categorical variables and scale numerical features.
- **Interactive Web Application:** Enter customer details through a Streamlit interface.
- **Prediction Probability:** Display the estimated probability of customer churn.
- **Model Persistence:** Load the trained model, encoders, and scaler without retraining the model every time the application starts.
- **Jupyter Notebooks:** Explore data preprocessing, model development, and prediction workflows.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Utilities:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Serialization:** Pickle
- **Development Environment:** Jupyter Notebook

## 📂 Project Structure

```text
ANN-Classification-Churn/
│
├── app.py                       # Streamlit web application
├── Churn_Modelling.csv          # Customer churn dataset
├── experiment.ipynb             # Data preprocessing and model experiments
├── prediction.ipynb             # Prediction workflow
│
├── model.h5                     # Trained ANN model
├── model.keras                  # Saved Keras model
├── scaler.pkl                   # Fitted feature scaler
├── label_encoder_gender.pkl     # Gender label encoder
├── label_encoder_geo.pkl        # Geography one-hot encoder
├── onehot_encoder_geo.pkl       # Additional encoder artifact
│
├── requirement.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset, which contains customer information and a target variable indicating whether the customer exited the bank.

The dataset contains **10,000 customer records**.

### Input Features

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years the customer has been with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |

### Target Variable

- `Exited = 1`: The customer left the bank.
- `Exited = 0`: The customer remained with the bank.

The columns `RowNumber`, `CustomerId`, and `Surname` are excluded from model training because they are not used as predictive input features.

## 🧠 Machine Learning Workflow

The project follows these major steps:

1. **Data Loading and Exploration:** Load the dataset using Pandas and inspect its structure.
2. **Data Cleaning:** Remove irrelevant columns and prepare the input features and target variable.
3. **Categorical Encoding:** Use Label Encoding for gender and One-Hot Encoding for geography.
4. **Feature Scaling:** Apply a fitted `StandardScaler` to scale the model inputs.
5. **Model Training:** Train an Artificial Neural Network using TensorFlow and Keras for binary classification.
6. **Model Saving:** Save the trained model and preprocessing objects for later use.
7. **Prediction:** Transform new customer details using the saved preprocessing objects and generate churn probabilities.
8. **Deployment Interface:** Use Streamlit to provide an interactive interface for customer churn prediction.

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Vinay980/ANN-Classification-Churn.git
```

### 2. Navigate to the Project Directory

```bash
cd ANN-Classification-Churn
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

The repository currently uses a file named `requirement.txt`.

```bash
pip install -r requirement.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Once the application starts, open the local URL displayed in your terminal, typically:

```text
http://localhost:8501
```

## 🖥️ How to Use the Application

1. Start the Streamlit application.
2. Select the customer's geography and gender.
3. Enter the customer's credit score, age, tenure, balance, estimated salary, and other account details.
4. View the predicted churn probability.
5. Check the prediction message to determine whether the model classifies the customer as likely to churn.

The prediction threshold in the current application is `0.5`:

- Probability greater than `0.5`: The customer is classified as likely to churn.
- Probability less than or equal to `0.5`: The customer is classified as unlikely to churn.

## 🔮 Sample Prediction

For example, a customer can be evaluated using the following input:

```python
input_data = {
    "CreditScore": 700,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 10000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000
}
```

The trained model processes the customer's information and returns a churn probability.

**Note:** The actual result depends on the trained model and its preprocessing pipeline. The example input does not imply a guaranteed prediction.

## 📈 Model Output

The application displays:

- **Churn Probability:** The model's estimated probability that the customer will leave.
- **Churn Classification:** Whether the predicted probability exceeds the configured threshold.

The probability is a model estimate, not a guarantee of future customer behavior.

## 🎯 Learning Objectives

This project demonstrates practical experience with:

- Artificial Neural Networks and deep learning.
- Binary classification problems.
- Exploratory data analysis and feature preprocessing.
- Label Encoding and One-Hot Encoding.
- Feature scaling with StandardScaler.
- Saving and loading trained machine learning models.
- Using a trained TensorFlow model for inference.
- Building interactive machine learning applications with Streamlit.

## 🔧 Future Improvements

- Add model evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Visualize customer churn patterns and feature distributions.
- Improve the user interface with clearer prediction summaries and probability charts.
- Add model performance monitoring and input validation.
- Deploy the application to a cloud hosting platform.
- Evaluate an appropriate classification threshold based on business requirements.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions are based on the available dataset and the trained model. They should not be treated as definitive financial or business decisions.

## 👨‍💻 Author

**Vinay Kumar Mahto**

- GitHub: [@Vinay980](https://github.com/Vinay980)
- Project Repository: [ANN-Classification-Churn](https://github.com/Vinay980/ANN-Classification-Churn)

---

⭐ If you find this project useful, consider giving the repository a star!
