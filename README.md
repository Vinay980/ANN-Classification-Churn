# 🏦 ANN Customer Churn Classification

An end-to-end **Artificial Neural Network (ANN)** project that predicts the likelihood of a bank customer leaving the bank. The trained TensorFlow/Keras model is served through an interactive **Streamlit** dashboard called **ChurnIQ**.

The application accepts customer details, applies the same preprocessing used during training, and displays a churn probability with a risk classification.

## 🚀 Live Demo

**[Open the Streamlit App](https://requirementtxt-fi2bkypw4v7mvtegdexvhq.streamlit.app/)**

> The live application requires the model and preprocessing artifacts to be available in its deployment environment.

## ✨ Features

- **Customer churn prediction:** Estimates whether a customer is likely to leave the bank.
- **ANN binary-classification model:** Uses a trained TensorFlow/Keras neural network.
- **Consistent preprocessing:** Encodes gender and geography and scales the input features using saved Scikit-learn objects.
- **Interactive Streamlit dashboard:** Collects customer details through sidebar controls.
- **Probability visualization:** Shows the estimated churn probability with a Plotly gauge and progress indicator.
- **Risk classification:** Uses a 0.5 probability threshold to label the result as higher or lower churn risk.
- **Saved model artifacts:** Loads the trained model, encoders, and scaler at application startup rather than retraining for every prediction.

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow, Keras |
| Data processing | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Web application | Streamlit |
| Model/artifact storage | HDF5/Keras model files, Pickle |
| Development | Jupyter Notebook |

## 📂 Project Structure

The main application and required runtime artifacts are:

```text
ANN-Classification-Churn/
├── app.py
├── Churn_Modelling.csv
├── model.h5
├── model.keras
├── scaler.pkl
├── label_encoder_gender.pkl
├── label_encoder_geo.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
└── README.md
```

The Streamlit app currently loads `model.h5`, `label_encoder_geo.pkl`, `label_encoder_gender.pkl`, and `scaler.pkl`. The Geography encoder is a fitted **OneHotEncoder**, despite the legacy filename `label_encoder_geo.pkl`.

## 📊 Dataset

This project uses the `Churn_Modelling.csv` bank customer dataset, containing 10,000 customer records.

### Input features

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |
| Geography_France | One-hot encoded France indicator |
| Geography_Germany | One-hot encoded Germany indicator |
| Geography_Spain | One-hot encoded Spain indicator |

The original `Geography` column is transformed into three one-hot encoded columns. The columns `RowNumber`, `CustomerId`, and `Surname` are excluded from model inputs.

### Target variable

- `Exited = 1`: The customer left the bank.
- `Exited = 0`: The customer stayed with the bank.

## 🧠 Model Development Workflow

1. **Load and explore data** using Pandas.
2. **Clean the dataset** by removing identifier and non-predictive columns.
3. **Encode categorical variables:** Label-encode `Gender` and one-hot encode `Geography`.
4. **Separate features and target:** Use the customer attributes as input features and `Exited` as the classification target.
5. **Split the data** into training and testing sets.
6. **Scale the features** with `StandardScaler`; fit the scaler on training data and apply it to test data.
7. **Train the ANN** with TensorFlow/Keras for binary classification.
8. **Save the model and preprocessing artifacts** for reuse.
9. **Serve predictions** through the Streamlit dashboard.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vinay980/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

The dependency file is named **`requirements.txt`**.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

## 🖥️ How to Use

1. Open the application.
2. Select the customer's geography and gender.
3. Enter the customer's credit score, balance, estimated salary, age, tenure, number of products, and account activity details.
4. Click **Analyze Customer**.
5. Review the predicted churn probability and risk classification.
6. Expand **View encoded model input** to inspect the feature values passed to the model.

## 📈 Understanding the Prediction

The application displays the model's output as a probability between 0 and 1, converted to a percentage.

| Model probability | Application classification |
|---|---|
| 0.50 or higher | Higher churn risk |
| Below 0.50 | Lower churn risk |

The 0.5 threshold is the threshold currently configured in the app. It is a modelling choice and may need to be tuned using validation results and business requirements.

**Important:** A predicted probability is an estimate from the trained model, not a guarantee that a customer will or will not leave.

## 🎯 What This Project Demonstrates

- Building and using an Artificial Neural Network for binary classification.
- Preparing tabular data for deep learning.
- Label encoding and one-hot encoding.
- Feature scaling with `StandardScaler`.
- Saving and loading TensorFlow/Keras models.
- Reusing fitted preprocessing objects during inference.
- Building and deploying an interactive Streamlit application.
- Communicating classification results with probability visualizations.

## 🔭 Possible Improvements

- Report evaluation metrics such as precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Tune the classification threshold based on validation performance and the cost of false positives/false negatives.
- Add model explainability to help interpret factors associated with churn.
- Add input validation and automated tests for the preprocessing pipeline.
- Track model performance and data drift after deployment.

## ⚠️ Disclaimer

This project is for learning and demonstration purposes. Predictions depend on the dataset, preprocessing pipeline, and trained model. They should not be used as the sole basis for financial or business decisions.

## 👨‍💻 Author

**Vinay Kumar Mahto**

- GitHub: [@Vinay980](https://github.com/Vinay980)
- Repository: [ANN-Classification-Churn](https://github.com/Vinay980/ANN-Classification-Churn)
- Live Demo: [Streamlit App](https://requirementtxt-fi2bkypw4v7mvtegdexvhq.streamlit.app/)

---

If you find this project useful, please consider giving the repository a ⭐.
